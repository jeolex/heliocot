#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


COMBO_RE = re.compile(r"^(C\d{3})_(raw|post)_px$")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.astype(float)
    yp = y_pred.astype(float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = y_true.astype(float)
    yp = y_pred.astype(float)

    n = int(len(yt))
    if n == 0:
        return {
            "n": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias_px": np.nan,
            "Bias_pct": np.nan,
            "NRMSE_mean": np.nan,
        }

    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias_px = float(np.mean(err))

    mean_true = float(np.mean(yt))
    if mean_true != 0:
        bias_pct = float((bias_px / mean_true) * 100.0)
        nrmse_mean = float(rmse / mean_true)
    else:
        bias_pct = np.nan
        nrmse_mean = np.nan

    return {
        "n": n,
        "R2": float(r2_score(yt, yp)) if n >= 2 else np.nan,
        "RMSE": rmse,
        "MAE": mae,
        "Bias_px": bias_px,
        "Bias_pct": bias_pct,
        "NRMSE_mean": nrmse_mean,
    }


def gt_distribution(y: pd.Series) -> pd.DataFrame:
    x = y.astype(float)
    return pd.DataFrame([{
        "n": int(x.shape[0]),
        "mean_true_px": float(x.mean()),
        "sd_true_px": float(x.std(ddof=1)) if x.shape[0] > 1 else np.nan,
        "median_true_px": float(x.median()),
        "p25_true_px": float(x.quantile(0.25)),
        "p75_true_px": float(x.quantile(0.75)),
        "IQR_true_px": float(x.quantile(0.75) - x.quantile(0.25)),
        "min_true_px": float(x.min()),
        "max_true_px": float(x.max()),
        "range_true_px": float(x.max() - x.min()),
        "p5_true_px": float(x.quantile(0.05)),
        "p95_true_px": float(x.quantile(0.95)),
    }])


def main() -> None:
    p = argparse.ArgumentParser(description="Heliocot — Validation metrics over all combo columns in for_validation.xlsx")
    p.add_argument("--in-xlsx", type=Path, required=True, help="Input Excel (e.g., workflow/for_validation.xlsx)")
    p.add_argument("--sheet", type=str, default=None, help="Sheet name (default: first sheet)")
    p.add_argument("--stage", type=str, default="", help="Filter by stage (e.g., E). Leave empty for no filter.")
    p.add_argument("--gt-col", type=str, default="real_leaf_px", help="Ground-truth column name")

    p.add_argument("--primary-sort", type=str, default="post_NRMSE_mean",
                   help="Primary sort for Top20 (e.g., post_NRMSE_mean or post_R2)")
    p.add_argument("--secondary-sort", type=str, default="post_R2",
                   help="Secondary sort for Top20")

    p.add_argument("--topk", type=int, default=20, help="Top-k combos to export")
    p.add_argument("--write-tables", action="store_true", help="Also write GT distribution + TopK sheets")

    p.add_argument("--out-xlsx", type=Path, required=True, help="Output Excel for summary/tables")
    args = p.parse_args()

    # Load
    if args.sheet is None:
        df = pd.read_excel(args.in_xlsx, sheet_name=0)  # first sheet
    else:
        df = pd.read_excel(args.in_xlsx, sheet_name=args.sheet)

    # Optional stage filter
    if args.stage:
        if "stage" not in df.columns:
            raise ValueError("Requested --stage filter but 'stage' column not found in input.")
        df = df[df["stage"].astype(str).str.upper() == args.stage.upper()].copy()

    # GT filter
    if args.gt_col not in df.columns:
        raise ValueError(f"Ground-truth column not found: {args.gt_col}")
    df = df[df[args.gt_col].notna()].copy()

    if df.empty:
        raise RuntimeError("No rows after filtering (stage/GT).")

    y_true_all = df[args.gt_col].astype(float)

    # Discover combo columns
    combo_map = {}  # combo_id -> {"raw": colname, "post": colname}
    for c in df.columns:
        m = COMBO_RE.match(str(c))
        if not m:
            continue
        combo_id, kind = m.group(1), m.group(2)
        combo_map.setdefault(combo_id, {})[kind] = c

    if not combo_map:
        raise RuntimeError("No combo columns found. Expected columns like C001_raw_px or C001_post_px.")

    # Compute summary
    rows = []
    for combo_id in sorted(combo_map.keys()):
        entry = {"combo": combo_id}

        for kind in ("raw", "post"):
            col = combo_map[combo_id].get(kind, None)
            if col is None:
                # not present
                entry[f"{kind}_n"] = 0
                entry[f"{kind}_R2"] = np.nan
                entry[f"{kind}_RMSE"] = np.nan
                entry[f"{kind}_MAE"] = np.nan
                entry[f"{kind}_Bias_px"] = np.nan
                entry[f"{kind}_Bias_pct"] = np.nan
                entry[f"{kind}_NRMSE_mean"] = np.nan
                continue

            y_pred = df[col].astype(float)
            mask = y_pred.notna() & y_true_all.notna()
            yt = y_true_all[mask].to_numpy()
            yp = y_pred[mask].to_numpy()

            met = compute_metrics(yt, yp)
            entry[f"{kind}_n"] = met["n"]
            entry[f"{kind}_R2"] = met["R2"]
            entry[f"{kind}_RMSE"] = met["RMSE"]
            entry[f"{kind}_MAE"] = met["MAE"]
            entry[f"{kind}_Bias_px"] = met["Bias_px"]
            entry[f"{kind}_Bias_pct"] = met["Bias_pct"]
            entry[f"{kind}_NRMSE_mean"] = met["NRMSE_mean"]

        rows.append(entry)

    summary = pd.DataFrame(rows).set_index("combo")

    # Optional tables
    S1 = gt_distribution(y_true_all)

    primary = args.primary_sort
    secondary = args.secondary_sort

    if primary not in summary.columns:
        raise ValueError(f"PRIMARY_SORT not found in summary columns: {primary}")
    if secondary and secondary not in summary.columns:
        raise ValueError(f"SECONDARY_SORT not found in summary columns: {secondary}")

    # Sort direction: for R2 higher better, for error metrics lower better
    asc_primary = not primary.endswith("_R2")
    asc_secondary = not secondary.endswith("_R2") if secondary else True

    sort_cols = [primary] + ([secondary] if secondary else [])
    sort_asc = [asc_primary] + ([asc_secondary] if secondary else [])

    topk = summary.sort_values(sort_cols, ascending=sort_asc).head(int(args.topk)).copy()

    # Export
    args.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=True)
        if args.write_tables:
            S1.to_excel(w, sheet_name="Table_S1_GT_Distribution", index=False)
            topk.to_excel(w, sheet_name=f"Table_S2_Top{args.topk}", index=True)

    print("Finished.")
    print(f"Rows used (GT available): {len(df)}")
    print(f"Combos found: {len(summary)}")
    print(f"Wrote: {args.out_xlsx.resolve()}")


if __name__ == "__main__":
    main()
