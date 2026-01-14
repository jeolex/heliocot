#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


# ============================================================
# Helper: image resolver
# ============================================================
def resolve_image_path(image_id: str, image_dir: Path) -> Path | None:
    image_id = str(image_id)
    p = image_dir / image_id
    if p.exists():
        return p
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p2 = image_dir / f"{image_id}{ext}"
        if p2.exists():
            return p2
    cands = list(image_dir.glob(f"{image_id}*"))
    return cands[0] if cands else None


# ============================================================
# Geometry utils
# ============================================================
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # TL
    rect[2] = pts[np.argmax(s)]       # BR
    rect[1] = pts[np.argmin(diff)]    # TR
    rect[3] = pts[np.argmax(diff)]    # BL
    return rect

def quad_area_px(ordered_pts: np.ndarray) -> float:
    return float(cv2.contourArea(ordered_pts.reshape(-1, 1, 2)))

def roi_bbox_from_points(pts: np.ndarray, pad=10, img_shape=None):
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1 = int(np.floor(xs.min() - pad))
    y1 = int(np.floor(ys.min() - pad))
    x2 = int(np.ceil(xs.max() + pad))
    y2 = int(np.ceil(ys.max() + pad))
    if img_shape is not None:
        H, W = img_shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return x1, y1, w, h


# ============================================================
# MARKER DETECTION (HEX proximity + saturation)
# ============================================================
def hex_to_bgr(hex_color: str) -> np.ndarray:
    hex_color = hex_color.strip().lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)

def detect_markers_by_hex(
    image_bgr: np.ndarray,
    target_hex: str,
    lab_dist_thr: float,
    sat_min: int,
    min_area: int,
    expected_n: int = 4,
    circ_min: float = 0.35
) -> np.ndarray:
    target_bgr = hex_to_bgr(target_hex)
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0, :]
    dist = np.linalg.norm(img_lab - tgt_lab, axis=2)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    mask = ((dist <= lab_dist_thr) & (sat >= sat_min)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        per = cv2.arcLength(cnt, True)
        if per > 0:
            circ = 4 * np.pi * (area / (per * per))
            if circ < circ_min:
                continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        blobs.append((area, cx, cy))

    blobs.sort(reverse=True, key=lambda t: t[0])
    if len(blobs) < expected_n:
        raise ValueError(f"{expected_n} marker bulunamadı. Bulunan: {len(blobs)}")

    pts = np.array([[b[1], b[2]] for b in blobs[:expected_n]], dtype=np.float32)
    return pts


# ============================================================
# C112 Postprocess on union mask
# ============================================================
def postprocess_union_mask(union01: np.ndarray, params: dict) -> np.ndarray | None:
    if union01 is None:
        return None

    min_area_ratio = float(params["min_area_ratio"])
    ksize = int(params["kernel_size"])
    use_open = bool(params["use_open"])
    use_close = bool(params["use_close"])

    m = (union01.astype(np.uint8) * 255)

    if use_close:
        k = ksize if ksize % 2 == 1 else ksize + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    if use_open:
        k = ksize if ksize % 2 == 1 else ksize + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    roi_area_px = union01.shape[0] * union01.shape[1]
    min_area_px = int(min_area_ratio * roi_area_px)

    if min_area_px > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(m, dtype=np.uint8)
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_area_px:
                keep[labels == lab] = 255
        m = keep

    return (m > 0).astype(np.uint8)


def leaf_px_post_c112(
    model: YOLO,
    roi_bgr: np.ndarray,
    params: dict,
    imgsz: int,
    conf_th: float,
    device: str,
    keep_mask: np.ndarray | None = None
):
    res = model.predict(
        roi_bgr,
        imgsz=imgsz,
        conf=conf_th,
        device=device,
        verbose=False
    )[0]

    if res.masks is None or res.masks.data is None:
        return 0, None

    masks = res.masks.data.cpu().numpy()  # (n, mh, mw)
    thr = float(params["mask_threshold"])
    union = np.any(masks > thr, axis=0).astype(np.uint8)  # 0/1

    H, W = roi_bgr.shape[:2]
    if union.shape != (H, W):
        union = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST)

    post = postprocess_union_mask(union, params)
    if post is None:
        return 0, None

    if keep_mask is not None:
        km = keep_mask
        if km.shape != post.shape:
            km = cv2.resize(km, (post.shape[1], post.shape[0]), interpolation=cv2.INTER_NEAREST)
        post = (post.astype(np.uint8) & (km > 0).astype(np.uint8))

    px_post = int(post.sum())
    return px_post, post


# ============================================================
# Mixed model helpers (ONLY NORM)
# ============================================================
def fit_mixed_robust(d: pd.DataFrame):
    methods = ["lbfgs", "powell", "nm"]

    formula_A = "y ~ C(time) + C(rep)"
    try:
        vc = {"cultivar": "0 + C(cultivar)"}
        for m in methods:
            res = smf.mixedlm(formula_A, data=d, groups=d["parsel_no"], vc_formula=vc, re_formula="1").fit(
                reml=False, method=m
            )
            return res, "MixedLM(vc=cultivar_random)", formula_A
    except Exception:
        pass

    formula_B = "y ~ C(time) + C(rep) + C(cultivar)"
    try:
        for m in methods:
            res = smf.mixedlm(formula_B, data=d, groups=d["parsel_no"], re_formula="1").fit(reml=False, method=m)
            return res, "MixedLM(cultivar_fixed)", formula_B
    except Exception:
        pass

    formula_C = "y ~ C(time) + C(rep)"
    try:
        for m in methods:
            res = smf.mixedlm(formula_C, data=d, groups=d["parsel_no"], re_formula="1").fit(reml=False, method=m)
            return res, "MixedLM(simple)", formula_C
    except Exception:
        pass

    # fallback: OLS cluster-robust
    try:
        ols = smf.ols(formula_B, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["parsel_no"]})
        return ols, "OLS(cluster_plot)", formula_B
    except Exception:
        ols = smf.ols(formula_C, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["parsel_no"]})
        return ols, "OLS(cluster_plot_simple)", formula_C


def compact_letter_display(levels, sig_matrix):
    letters = []
    groups = {lv: "" for lv in levels}

    for lv in levels:
        placed = False
        for letter_set in letters:
            ok = True
            for other in letter_set:
                i = levels.index(lv)
                j = levels.index(other)
                if sig_matrix[i, j] or sig_matrix[j, i]:
                    ok = False
                    break
            if ok:
                letter_set.add(lv)
                placed = True
                break
        if not placed:
            letters.append(set([lv]))

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for k, letter_set in enumerate(letters):
        ch = alphabet[k] if k < len(alphabet) else f"g{k}"
        for lv in letter_set:
            groups[lv] += ch
    return groups


def run_norm_mixedmodel_from_wide(wide: pd.DataFrame, time_levels, time_decimal_hr):
    # plot-level aggregation
    px_cols = [f"leaf_px_std_{t}" for t in time_levels]
    need = ["parsel_no", "cultivar", "rep", "plant_no"] + px_cols
    miss = [c for c in need if c not in wide.columns]
    if miss:
        raise ValueError(f"Wide tabloda eksik kolonlar: {miss}")

    df0 = wide.copy()
    df0["parsel_no"] = df0["parsel_no"].astype(str)
    df0["cultivar"]  = df0["cultivar"].astype(str)
    df0["rep"]       = df0["rep"].astype(str)
    df0["plant_no"]  = df0["plant_no"].astype(str)

    for c in px_cols:
        df0[c] = pd.to_numeric(df0[c], errors="coerce")

    plot = (df0
            .groupby(["parsel_no", "cultivar", "rep"], dropna=False)
            .agg(
                n_plants=("plant_no", "count"),
                **{c: (c, "mean") for c in px_cols}
            )
            .reset_index())

    # long
    long = plot.melt(
        id_vars=["parsel_no", "cultivar", "rep", "n_plants"],
        value_vars=px_cols,
        var_name="time",
        value_name="pixel"
    )
    long["time"] = long["time"].str.replace("leaf_px_std_", "", regex=False)
    long["time"] = pd.to_numeric(long["time"], errors="coerce").astype("Int64")
    long = long[long["time"].isin(time_levels)].copy()
    long["pixel"] = pd.to_numeric(long["pixel"], errors="coerce")
    long = long.dropna(subset=["pixel"]).copy()

    # NORM: y = pixel / mean(pixel over times within each plot)
    denom = long.groupby("parsel_no")["pixel"].transform(lambda x: np.nanmean(x.values.astype(float)))
    long["y"] = long["pixel"].astype(float) / denom.replace(0, np.nan)
    long = long.dropna(subset=["y"]).copy()

    # categories
    time_cats = [str(t) for t in time_levels]
    long["time"] = long["time"].astype(int).astype(str)
    long["time"] = pd.Categorical(long["time"], categories=time_cats, ordered=True)

    rep_cats = sorted(long["rep"].astype(str).unique(), key=lambda x: (len(x), x))
    cult_cats = sorted(long["cultivar"].astype(str).unique(), key=lambda x: (len(x), x))

    long["rep"] = pd.Categorical(long["rep"].astype(str), categories=rep_cats, ordered=True)
    long["cultivar"] = pd.Categorical(long["cultivar"].astype(str), categories=cult_cats, ordered=False)

    # fit
    res, model_type, formula_used = fit_mixed_robust(long)

    # fixed params + cov
    if hasattr(res, "fe_params"):  # MixedLM
        beta = res.fe_params
        fe_names = list(beta.index)
        cov_all = res.cov_params()
        if hasattr(cov_all, "loc"):
            cov_fe = cov_all.loc[fe_names, fe_names].to_numpy()
        else:
            cov_fe = np.array(cov_all)[:len(fe_names), :len(fe_names)]
        design_info = res.model.data.design_info
    else:  # OLS
        beta = res.params
        fe_names = list(beta.index)
        cov_fe = res.cov_params().to_numpy()
        design_info = res.model.data.design_info

    has_cult_fixed = any("C(cultivar)" in str(p) for p in fe_names)

    def Xbar_for_time(t_level_str: str):
        if has_cult_fixed:
            grid = pd.DataFrame(
                [(t_level_str, r, c) for r in rep_cats for c in cult_cats],
                columns=["time", "rep", "cultivar"]
            )
            grid["cultivar"] = pd.Categorical(grid["cultivar"], categories=cult_cats, ordered=False)
        else:
            grid = pd.DataFrame([(t_level_str, r) for r in rep_cats], columns=["time", "rep"])

        grid["time"] = pd.Categorical(grid["time"], categories=time_cats, ordered=True)
        grid["rep"]  = pd.Categorical(grid["rep"], categories=rep_cats, ordered=True)
        X = build_design_matrices([design_info], grid, return_type="dataframe")[0]
        X = X.reindex(columns=fe_names, fill_value=0.0)
        return X.mean(axis=0).to_numpy(dtype=float)

    # EMM
    emm_rows = []
    for tlev in time_cats:
        xb = Xbar_for_time(tlev)
        mu = float(np.dot(xb, np.asarray(beta, dtype=float)))
        se = float(np.sqrt(np.dot(xb, np.dot(cov_fe, xb))))
        emm_rows.append({
            "mode": "NORM",
            "time": int(tlev),
            "time_hr": time_decimal_hr.get(int(tlev), np.nan),
            "EMM_mean": mu,
            "EMM_SE": se
        })
    emm = pd.DataFrame(emm_rows).sort_values("time").reset_index(drop=True)

    # Pairwise Holm
    pairs = []
    for i, t1 in enumerate(time_cats):
        xb1 = Xbar_for_time(t1)
        for j, t2 in enumerate(time_cats):
            if j <= i:
                continue
            xb2 = Xbar_for_time(t2)
            L = xb1 - xb2
            diff = float(np.dot(L, np.asarray(beta, dtype=float)))
            se_d = float(np.sqrt(np.dot(L, np.dot(cov_fe, L))))
            if se_d == 0 or np.isnan(se_d):
                z = np.nan
                p = np.nan
            else:
                z = diff / se_d
                p = 2 * (1 - norm.cdf(abs(z)))
            pairs.append({
                "mode": "NORM",
                "time1": int(t1),
                "time2": int(t2),
                "diff(time1-time2)": diff,
                "SE_diff": se_d,
                "z": z,
                "p_value": p,
                "LSD_like_0.05(=1.96*SE)": 1.96 * se_d if np.isfinite(se_d) else np.nan
            })
    pairwise = pd.DataFrame(pairs)
    if pairwise["p_value"].notna().any():
        pvals = pairwise["p_value"].fillna(1.0).to_numpy()
        rej, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        pairwise["p_holm"] = p_holm
        pairwise["sig_holm_0.05"] = rej.astype(int)
    else:
        pairwise["p_holm"] = np.nan
        pairwise["sig_holm_0.05"] = 0

    # CLD
    emm_rank = emm.sort_values("EMM_mean", ascending=False).reset_index(drop=True)
    levs = [int(x) for x in emm_rank["time"].tolist()]
    sig_matrix = np.zeros((len(levs), len(levs)), dtype=bool)
    for _, r in pairwise.iterrows():
        if int(r["sig_holm_0.05"]) == 1:
            a = int(r["time1"]); b = int(r["time2"])
            ia = levs.index(a); ib = levs.index(b)
            sig_matrix[ia, ib] = True
            sig_matrix[ib, ia] = True

    cld = compact_letter_display(levs, sig_matrix)
    emm_rank["group_holm"] = emm_rank["time"].map(cld)
    emm_rank["rank_desc"] = np.arange(1, len(emm_rank) + 1)

    # observed means (NORM y)
    obs = (long.groupby("time")["y"].agg(n="count", mean="mean", sd="std").reset_index())
    obs["time"] = obs["time"].astype(str).astype(int)
    obs["se"] = obs["sd"] / np.sqrt(obs["n"].replace(0, np.nan))
    obs["mode"] = "NORM"
    obs = obs.sort_values("time").reset_index(drop=True)

    info = pd.DataFrame([{
        "mode": "NORM",
        "model_type_used": model_type,
        "formula_used": formula_used,
        "n_obs": len(long),
        "n_plots": long["parsel_no"].nunique()
    }])

    return plot, long, info, obs, emm, emm_rank, pairwise


# ============================================================
# MAIN
# ============================================================
def main():
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description="Heliocot — C112 helio pipeline (marker ROI + seg + std + NORM mixed model)")

    ap.add_argument("--workflow-dir", type=Path, default=Path("workflow"))
    ap.add_argument("--stage", type=str, default="E")
    ap.add_argument("--device", type=str, default="cpu")  # cpu / mps / cuda:0
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.6)

    ap.add_argument("--weights", type=Path, default=None, help="Default: first *.pt under workflow/models")
    ap.add_argument("--out-xlsx", type=Path, default=None, help="Default: workflow/outputs/helio_E_C112_NORM.xlsx")

    # marker tuning
    ap.add_argument("--red-hex", type=str, default="#FF0000")
    ap.add_argument("--blue-hex", type=str, default="#0000FF")
    ap.add_argument("--red-lab", type=float, default=22.0)
    ap.add_argument("--blue-lab", type=float, default=22.0)
    ap.add_argument("--red-sat", type=int, default=150)
    ap.add_argument("--blue-sat", type=int, default=150)
    ap.add_argument("--red-min-area", type=int, default=60)
    ap.add_argument("--blue-min-area", type=int, default=60)
    ap.add_argument("--circ-min", type=float, default=0.35)

    # outputs
    ap.add_argument("--save-viz", action="store_true", help="Save *_viz.jpg into workflow/outputs/qc")
    ap.add_argument("--save-debug", action="store_true", help="Save *_dbg.jpg into workflow/outputs/qc")
    ap.add_argument("--export-masks", action="store_true", help="Save final post masks into workflow/outputs/masks")

    args = ap.parse_args()

    wf = args.workflow_dir
    image_dir = wf / "images" / "helio"
    meta_path = wf / "plot_metadata.xlsx"
    models_dir = wf / "models"

    out_xlsx = args.out_xlsx or (wf / "outputs" / f"helio_{args.stage.upper()}_C112_NORM.xlsx")
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    qc_dir = wf / "outputs" / "qc"
    masks_dir = wf / "outputs" / "masks"
    if args.save_viz or args.save_debug:
        qc_dir.mkdir(parents=True, exist_ok=True)
    if args.export_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    # pick weights
    if args.weights is None:
        pts = sorted(models_dir.glob("*.pt"))
        if not pts:
            raise RuntimeError(f"No .pt found in {models_dir}")
        weights = pts[0]
    else:
        weights = args.weights
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

    # C112 params (fixed)
    C112_PARAMS = {
        "mask_threshold": 0.6,
        "min_area_ratio": 0.003,
        "kernel_size": 5,
        "use_open": True,
        "use_close": True,
    }

    # time levels and reporting hours
    TIME_LEVELS = [1, 2, 3, 4, 5]
    TIME_DECIMAL_HR = {
        1: 8 + 10/60,
        2: 10 + 15/60,
        3: 13 + 15/60,
        4: 16 + 5/60,
        5: 18 + 15/60
    }

    # model
    model = YOLO(str(weights))

    # metadata
    meta = pd.read_excel(meta_path)

    # stage filter
    stage = args.stage.upper()
    if "stage" in meta.columns:
        meta_s = meta[meta["stage"].astype(str).str.upper() == stage].copy()
    else:
        meta_s = meta.copy()

    # image_available filter if exists
    if "image_available" in meta_s.columns:
        meta_s = meta_s[meta_s["image_available"].astype(str).isin(["1", "True", "TRUE", "true"])].copy()

    print(f"Rows after stage/image_available filter: {len(meta_s)}")

    # marker detectors bound to args
    def detect_red(img_bgr):
        return detect_markers_by_hex(
            image_bgr=img_bgr,
            target_hex=args.red_hex,
            lab_dist_thr=args.red_lab,
            sat_min=args.red_sat,
            min_area=args.red_min_area,
            expected_n=4,
            circ_min=args.circ_min
        )

    def detect_blue(img_bgr):
        return detect_markers_by_hex(
            image_bgr=img_bgr,
            target_hex=args.blue_hex,
            lab_dist_thr=args.blue_lab,
            sat_min=args.blue_sat,
            min_area=args.blue_min_area,
            expected_n=4,
            circ_min=args.circ_min
        )

    # 1) REF_AREA_PX_TARGET = median(ref_area_px) over usable images
    ref_areas = []
    for _, r in meta_s.iterrows():
        ip = resolve_image_path(r["image_id"], image_dir)
        if ip is None:
            continue
        im = cv2.imread(str(ip))
        if im is None:
            continue
        try:
            rp = detect_red(im)
            rr = order_points(rp)
            ref_areas.append(quad_area_px(rr))
        except Exception:
            continue

    if len(ref_areas) == 0:
        raise RuntimeError("ref_area_px bulunamadı. Kırmızı marker tespiti ayarlarını kontrol et.")
    REF_AREA_PX_TARGET = float(np.median(ref_areas))
    print("REF_AREA_PX_TARGET (median):", REF_AREA_PX_TARGET)

    # 2) Process images -> image-level table
    rows = []
    for _, r in meta_s.iterrows():
        image_id = r["image_id"]
        img_path = resolve_image_path(image_id, image_dir)

        out = r.to_dict()
        out.update({
            "img_path": str(img_path) if img_path else None,
            "ref_area_px": np.nan,
            "roi_x": np.nan, "roi_y": np.nan, "roi_w": np.nan, "roi_h": np.nan,
            "leaf_px_post": np.nan,
            "leaf_px_std": np.nan,
            "status": "OK",
            "error": ""
        })

        if img_path is None:
            out["status"] = "SKIP"; out["error"] = "image not found"
            rows.append(out); continue

        img = cv2.imread(str(img_path))
        if img is None:
            out["status"] = "SKIP"; out["error"] = "cv2.imread failed"
            rows.append(out); continue

        try:
            red_pts = detect_red(img)
            red_rect = order_points(red_pts)
            ref_area_px = quad_area_px(red_rect)
            out["ref_area_px"] = ref_area_px

            blue_pts = detect_blue(img)
            blue_rect = order_points(blue_pts)

            x, y, w, h = roi_bbox_from_points(blue_rect, pad=10, img_shape=img.shape)
            out["roi_x"], out["roi_y"], out["roi_w"], out["roi_h"] = x, y, w, h

            roi = img[y:y+h, x:x+w].copy()

            blue_poly_roi = (blue_rect - np.array([[x, y]], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
            keep_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(keep_mask, [blue_poly_roi], 255)

            roi_masked = roi.copy()
            roi_masked[keep_mask == 0] = (0, 0, 0)

            leaf_px_post, post_mask = leaf_px_post_c112(
                model=model,
                roi_bgr=roi_masked,
                params=C112_PARAMS,
                imgsz=args.imgsz,
                conf_th=args.conf,
                device=args.device,
                keep_mask=keep_mask
            )
            out["leaf_px_post"] = leaf_px_post

            if ref_area_px > 0:
                out["leaf_px_std"] = leaf_px_post * (REF_AREA_PX_TARGET / ref_area_px)

            # QC / debug / masks
            stem = Path(str(img_path)).stem

            if args.export_masks and post_mask is not None:
                # place ROI mask into full image canvas
                big = np.zeros(img.shape[:2], dtype=np.uint8)
                pm = post_mask.astype(np.uint8)
                if pm.shape != (h, w):
                    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                big[y:y+h, x:x+w] = (pm * 255).astype(np.uint8)
                cv2.imwrite(str(masks_dir / f"{stem}.png"), big)

            if args.save_viz:
                viz = img.copy()
                overlay = viz.copy()
                red_poly = red_rect.astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [red_poly], color=(0, 0, 255))
                blue_poly_full = blue_rect.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [blue_poly_full], isClosed=True, color=(255, 0, 0), thickness=4)

                if post_mask is not None:
                    big = np.zeros(viz.shape[:2], dtype=np.uint8)
                    pm = post_mask.astype(np.uint8)
                    if pm.shape != (h, w):
                        pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                    big[y:y+h, x:x+w] = (pm * 255).astype(np.uint8)
                    overlay[big > 0] = (255, 0, 0)

                viz = cv2.addWeighted(overlay, 0.35, viz, 0.65, 0)
                cv2.putText(viz, f"ref_area_px={ref_area_px:.1f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(viz, f"leaf_px_post={leaf_px_post}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                cv2.imwrite(str(qc_dir / f"{stem}_viz.jpg"), viz)

            if args.save_debug:
                dbg = img.copy()
                for (px, py) in red_rect:
                    cv2.drawMarker(dbg, (int(px), int(py)), (0, 255, 0),
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                for (px, py) in blue_rect:
                    cv2.drawMarker(dbg, (int(px), int(py)), (255, 255, 0),
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.imwrite(str(qc_dir / f"{stem}_dbg.jpg"), dbg)

        except Exception as e:
            out["status"] = "ERR"
            out["error"] = str(e)

        rows.append(out)

    df_img = pd.DataFrame(rows)
    print(df_img["status"].value_counts(dropna=False))

    # 3) Build WIDE table (no intermediate Excel)
    df_ok = df_img.copy()
    if "status" in df_ok.columns:
        df_ok = df_ok[df_ok["status"].astype(str).str.upper() == "OK"].copy()

    if "time_of_day" not in df_ok.columns:
        raise ValueError("'time_of_day' kolonu bulunamadı (plot_metadata.xlsx kontrol).")

    # time clean
    df_ok["time_of_day"] = pd.to_numeric(df_ok["time_of_day"], errors="coerce").astype("Int64")
    df_ok = df_ok[df_ok["time_of_day"].isin(TIME_LEVELS)].copy()

    required = ["plot_no", "variety", "rep", "plant_no", "time_of_day", "leaf_px_std"]
    missing = [c for c in required if c not in df_ok.columns]
    if missing:
        raise ValueError(f"Excel için eksik kolonlar: {missing}")

    index_cols = ["plot_no", "variety", "rep", "plant_no"]
    wide_px = df_ok.pivot_table(index=index_cols, columns="time_of_day", values="leaf_px_std", aggfunc="mean")

    # rename cols to leaf_px_std_1..5 (ensure missing times present)
    wide_px.columns = [int(c) for c in wide_px.columns]
    for t in TIME_LEVELS:
        if t not in wide_px.columns:
            wide_px[t] = np.nan
    wide_px = wide_px[TIME_LEVELS]
    wide_px.columns = [f"leaf_px_std_{t}" for t in TIME_LEVELS]

    wide = wide_px.reset_index()
    wide = wide.rename(columns={"plot_no": "parsel_no", "variety": "cultivar"})
    wide = wide.sort_values(["parsel_no", "cultivar", "rep", "plant_no"]).reset_index(drop=True)

    # 4) Mixed model / EMM (ONLY NORM)
    plot_level, long_used, info, obs, emm, emm_rank, pairwise = run_norm_mixedmodel_from_wide(
        wide=wide,
        time_levels=TIME_LEVELS,
        time_decimal_hr=TIME_DECIMAL_HR
    )

    # 5) Write ONE Excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_img.to_excel(w, sheet_name="image_level_C112", index=False)
        wide.to_excel(w, sheet_name="wide_by_time", index=False)

        plot_level.to_excel(w, sheet_name="NORM_plot_level_input", index=False)
        long_used.to_excel(w, sheet_name="NORM_long_input", index=False)

        info.to_excel(w, sheet_name="NORM_model_info", index=False)
        obs.to_excel(w, sheet_name="NORM_observed_means", index=False)
        emm.to_excel(w, sheet_name="NORM_emm_numeric", index=False)
        emm_rank.to_excel(w, sheet_name="NORM_time_ranking_EMM", index=False)
        pairwise.to_excel(w, sheet_name="NORM_pairwise_Holm", index=False)

    print("OK -> Wrote:", out_xlsx.resolve())


if __name__ == "__main__":
    main()
