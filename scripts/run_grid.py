#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def compute_leaf_area_from_result(
    result,
    mask_threshold: float,
    min_area_ratio: float,
    kernel_size: int,
    use_open: bool,
    use_close: bool,
    return_mask: bool = False,
):
    """Compute raw/post leaf area in px from a single Ultralytics Result (one image)."""
    if result.masks is None:
        out = {"raw_area_px": 0, "post_area_px": 0}
        if return_mask:
            out["post_mask"] = None
        return out

    masks = result.masks.data.cpu().numpy()  # (N, h, w), float/0-1
    H_yolo, W_yolo = masks.shape[1:]

    orig_img = result.orig_img
    H_orig, W_orig = orig_img.shape[:2]

    # RAW union (thresholded)
    raw_union = np.max((masks > mask_threshold).astype(np.uint8), axis=0)
    raw_union = cv2.resize(raw_union, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
    raw_area = int(raw_union.sum())

    # POST process per instance then union
    min_area = float(min_area_ratio) * float(H_yolo * W_yolo)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    refined_masks = []
    for m in masks:
        if m.sum() < min_area:
            continue

        m = (m > mask_threshold).astype(np.uint8)

        if use_open:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

        if use_close:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        m = cv2.resize(m, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        refined_masks.append(m)

    if refined_masks:
        final_mask = np.max(refined_masks, axis=0).astype(np.uint8)
        post_area = int(final_mask.sum())
       
    else:
        final_mask = None
        post_area = 0
       

    out = {"raw_area_px": raw_area, "post_area_px": post_area}
    if return_mask:
        out["post_mask"] = final_mask
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Heliocot — Full grid evaluation (model × conf × postprocess params).")

    # Paths (workflow layout)
    p.add_argument("--workflow-dir", type=Path, default=Path("workflow"))
    p.add_argument("--image-dir", type=Path, default=None, help="Default: workflow/images/resized")
    p.add_argument("--meta-path", type=Path, default=None, help="Default: workflow/plot_metadata.xlsx")
    p.add_argument("--models-dir", type=Path, default=None, help="Default: workflow/models")
    p.add_argument("--out-xlsx", type=Path, default=None, help="Default: workflow/outputs/grid_results.xlsx")
    p.add_argument("--masks-outdir", type=Path, default=None, help="Default: workflow/outputs/masks")

    # Runtime
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--device", type=str, default="cpu")  # cpu / mps / cuda:0 etc.
    p.add_argument("--image-ext", type=str, default=".jpg")

    # Save masks for ONE combo only (optional)
    p.add_argument("--save-masks-combo", type=str, default="", help="Example: C112 (saves post mask PNGs only for this combo)")
    p.add_argument("--overwrite-masks", action="store_true")

    args = p.parse_args()

    wf = args.workflow_dir
    image_dir = args.image_dir or (wf / "images" / "resized")
    meta_path = args.meta_path or (wf / "plot_metadata.xlsx")
    models_dir = args.models_dir or (wf / "models")

    out_xlsx = args.out_xlsx or (wf / "outputs" / "grid_results.xlsx")
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    masks_outdir = args.masks_outdir or (wf / "outputs" / "masks")
    if args.save_masks_combo:
        masks_outdir.mkdir(parents=True, exist_ok=True)

    # Parameter grid (your original 144 combos)
    param_grid = {
        "conf_th": [0.4, 0.5, 0.6, 0.7],
        "mask_threshold": [0.5, 0.6, 0.7],
        "min_area_ratio": [0.002, 0.003, 0.005],
        "kernel_size": [3, 5],
        "use_open": [False, True],
        "use_close": [True],
    }

    # Build combos with stable IDs: C001..Cxxx over (model × params)
    conf_list = param_grid["conf_th"]
    other_keys = ["mask_threshold", "min_area_ratio", "kernel_size", "use_open", "use_close"]
    other_combos = list(itertools.product(*(param_grid[k] for k in other_keys)))

    model_paths = sorted(models_dir.glob("*.pt"))
    if not model_paths:
        raise RuntimeError(f"No .pt files found in: {models_dir}")

    # Load models once (important)
    models = [(mp, YOLO(mp)) for mp in model_paths]

    combo_rows = []
    combo_defs = []  # (combo_id, model_path, conf_th, other_param_dict)
    combo_counter = 1

    for model_path, _ in models:
        for conf_th in conf_list:
            for values in other_combos:
                combo_id = f"C{combo_counter:03d}"
                combo_counter += 1
                param_dict = dict(zip(other_keys, values))
                combo_rows.append(
                    {
                        "combo_id": combo_id,
                        "model_id": model_path.stem,
                        "conf_th": conf_th,
                        **param_dict,
                    }
                )
                combo_defs.append((combo_id, model_path, conf_th, param_dict))

    df_combinations = pd.DataFrame(combo_rows)

    # Metadata
    meta = pd.read_excel(meta_path)

    # Keep your filters
    meta_eval = meta[(meta["image_available"] == 1) & (meta["used_for_training"] == 0)].copy()

    prediction_rows = []
    image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"

    for _, meta_row in meta_eval.iterrows():
        image_id = str(meta_row["image_id"])
        image_path = image_dir / f"{image_id}{image_ext}"

        if not image_path.exists():
            print(f"Missing image, skipped: {image_path}")
            continue

        prediction_row = {
            "image_id": image_id,
            "variety": meta_row.get("variety", None),
            "stage": meta_row.get("stage", None),
            "time_of_day": meta_row.get("time_of_day", None),
            "rep": meta_row.get("rep", None),
            "plant_no": meta_row.get("plant_no", None),
            "plot_no": meta_row.get("plot_no", None),
        }

        # Iterate combos
        for combo_id, model_path, conf_th, params in combo_defs:
            # find loaded model object
            model = next(m for (mp, m) in models if mp == model_path)

            results = model.predict(
                source=str(image_path),
                conf=float(conf_th),
                imgsz=int(args.imgsz),
                device=args.device,
                save=False,
                show=False,
                verbose=False,
            )

            want_mask = bool(args.save_masks_combo) and (combo_id == args.save_masks_combo)
            area = compute_leaf_area_from_result(
                results[0],
                mask_threshold=float(params["mask_threshold"]),
                min_area_ratio=float(params["min_area_ratio"]),
                kernel_size=int(params["kernel_size"]),
                use_open=bool(params["use_open"]),
                use_close=bool(params["use_close"]),
                return_mask=want_mask,
            )

            prediction_row[f"{combo_id}_raw_px"] = area["raw_area_px"]
            prediction_row[f"{combo_id}_post_px"] = area["post_area_px"]
   

            if want_mask:
                mask = area["post_mask"]
                if mask is not None:
                    out_mask = masks_outdir / f"{image_id}.png"
                    if out_mask.exists() and not args.overwrite_masks:
                        pass
                    else:
                        cv2.imwrite(str(out_mask), (mask * 255).astype(np.uint8))

        prediction_rows.append(prediction_row)
        print(f"Done: {image_id}")

    df_predictions = pd.DataFrame(prediction_rows)

    # Write Excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_predictions.to_excel(writer, sheet_name="predictions", index=False)
        df_combinations.to_excel(writer, sheet_name="combinations", index=False)

    print("Finished.")
    print(f"Images processed: {len(df_predictions)}")
    print(f"Total combos: {len(df_combinations)}")
    print(f"Excel: {out_xlsx.resolve()}")


if __name__ == "__main__":
    main()
