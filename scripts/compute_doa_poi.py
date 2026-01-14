#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PX5_COLS = [f"leaf_px_std_{i}" for i in [1, 2, 3, 4, 5]]
PX3_COLS = ["leaf_px_std_1", "leaf_px_std_3", "leaf_px_std_5"]


def safe_norm_vec(y: np.ndarray, mode: str = "mean"):
    """
    y: (n, k) float array
    mode:
      - "mean": denom = nanmean(row)
      - "t1"  : denom = first element (column 0)
    """
    y = y.astype(float)
    if mode == "mean":
        denom = np.nanmean(y, axis=1)
    elif mode == "t1":
        denom = y[:, 0]
    else:
        raise ValueError("mode must be 'mean' or 't1'")

    out = np.full_like(y, np.nan, dtype=float)
    ok = np.isfinite(denom) & (denom != 0)
    out[ok, :] = y[ok, :] / denom[ok, None]
    return out, denom


def main() -> None:
    p = argparse.ArgumentParser(
        description="Heliocot — compute DOA (Daily Orientation Amplitude) and POI (Peak Orientation Index) from helio_E_C112_NORM.xlsx."
    )

    # Defaults set to your confirmed file/sheet
    p.add_argument(
        "--in-xlsx",
        type=Path,
        default=Path("workflow/outputs/helio_E_C112_NORM.xlsx"),
        help="Default: workflow/outputs/helio_E_C112_NORM.xlsx",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default="wide_by_time",
        help="Default: wide_by_time",
    )
    p.add_argument(
        "--out-xlsx",
        type=Path,
        default=Path("workflow/outputs/helio_E_C112_DOA_POI.xlsx"),
        help="Default: workflow/outputs/helio_E_C112_DOA_POI.xlsx",
    )

    # Normalization modes (keep as options)
    p.add_argument("--norm5", type=str, default="mean", choices=["mean", "t1"], help="DOA için 5-pt normalize modu.")
    p.add_argument("--norm3", type=str, default="mean", choices=["mean", "t1"], help="POI için 3-pt normalize modu.")

    args = p.parse_args()

    in_xlsx = args.in_xlsx
    sheet = args.sheet
    out_xlsx = args.out_xlsx
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(in_xlsx, sheet_name=sheet)

    # Required checks
    missing = [c for c in PX5_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sheet '{sheet}': {missing}")

    # Numeric conversion
    for c in PX5_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # =========================
    # DOA (Daily Orientation Amplitude) = max-min of 5-pt normalized series
    # (old name: HAI_amp_5pt)
    # =========================
    Y5 = df[PX5_COLS].to_numpy(dtype=float)  # (n,5)
    Y5n, _ = safe_norm_vec(Y5, mode=args.norm5)

    finite_counts_5 = np.sum(np.isfinite(Y5n), axis=1)
    y5_max = np.nanmax(Y5n, axis=1)
    y5_min = np.nanmin(Y5n, axis=1)
    doa = np.where(finite_counts_5 >= 2, y5_max - y5_min, np.nan)

    # =========================
    # POI (Peak Orientation Index) = y3 - 0.5*(y1+y5) on 3-pt normalized series
    # (old name: HI_peak_3pt)
    # =========================
    Y3 = df[PX3_COLS].to_numpy(dtype=float)  # (n,3)
    Y3n, _ = safe_norm_vec(Y3, mode=args.norm3)

    y1n = Y3n[:, 0]
    y3n = Y3n[:, 1]
    y5n = Y3n[:, 2]

    poi = y3n - 0.5 * (y1n + y5n)
    ok3 = np.isfinite(y1n) & np.isfinite(y3n) & np.isfinite(y5n)
    poi = np.where(ok3, poi, np.nan)

    # Append only two new columns
    out = df.copy()
    out["DOA"] = doa
    out["POI"] = poi

    # Optional summaries (only if keys exist)
    base_cols = [c for c in ["parsel_no", "cultivar", "rep"] if c in out.columns]
    have_plant = "plant_no" in out.columns

    plot = pd.DataFrame()
    cult = pd.DataFrame()

    if base_cols:
        plot = (out
                .groupby(base_cols, dropna=False)
                .agg(
                    n_plants=("plant_no", "count") if have_plant else ("DOA", "count"),
                    DOA=("DOA", "mean"),
                    POI=("POI", "mean"),
                )
                .reset_index())

        if "cultivar" in plot.columns:
            def mean_se(s: pd.Series):
                s = pd.to_numeric(s, errors="coerce")
                n = int(s.notna().sum())
                mu = float(s.mean()) if n > 0 else np.nan
                sd = float(s.std(ddof=1)) if n > 1 else np.nan
                se = float(sd / np.sqrt(n)) if n > 1 else np.nan
                return pd.Series({"n_plot": n, "mean": mu, "se": se})

            cult = (plot.groupby("cultivar", dropna=False)
                    .apply(lambda g: pd.concat([
                        mean_se(g["DOA"]).add_prefix("DOA_"),
                        mean_se(g["POI"]).add_prefix("POI_"),
                    ]), include_groups=False)
                    .reset_index())

    defs = pd.DataFrame([
        {"column": "DOA", "new_name": "Daily Orientation Amplitude (DOA)", "old_name": "HAI_amp_5pt",
         "definition": "max-min of 5-pt normalized leaf_px_std series"},
        {"column": "POI", "new_name": "Peak Orientation Index (POI)", "old_name": "HI_peak_3pt",
         "definition": "y3 - 0.5*(y1+y5) computed on 3-pt normalized series (t1,t3,t5)"},
        {"input_file": str(in_xlsx), "sheet": sheet, "norm5": args.norm5, "norm3": args.norm3},
    ])

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="wide_by_time_plus_DOA_POI", index=False)
        if not plot.empty:
            plot.to_excel(w, sheet_name="plot_level_DOA_POI", index=False)
        if not cult.empty:
            cult.to_excel(w, sheet_name="cultivar_mean_SE", index=False)
        defs.to_excel(w, sheet_name="definitions", index=False)

    print("OK -> Wrote:", out_xlsx.resolve())
    print("Input:", in_xlsx, "| sheet:", sheet)
    print("Rows:", len(out))


if __name__ == "__main__":
    main()
