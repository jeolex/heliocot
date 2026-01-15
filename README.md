# Heliocot

Heliocot is an open, reproducible segmentation-to-PLA workflow for top-down field RGB images of cotton. It converts instance segmentation masks into reference-area standardized projected leaf area (PLA) time series and supports quantifying diurnal canopy orientation dynamics.

## What this repository contains
- Modular scripts for dataset preparation, model training, inference/post-processing, and PLA computation
- Utilities to generate reference-area standardized PLA time series and selected derived indices
- Statistical analysis scripts (mixed-effects models / EMMs)

## Data and code availability
Large research assets (images, annotations, and derived outputs) are shared via Zenodo, while this GitHub repository hosts the workflow code and documentation.

- Zenodo (dataset archive: images/annotations/derived outputs) DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18243285.svg)](https://doi.org/10.5281/zenodo.18243285)  
- Zenodo (workflow release: versioned repository snapshot) DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18241920.svg)](https://doi.org/10.5281/zenodo.18241920)

## Installation

### Option A: conda (recommended)
```bash
conda env create -f environment.yml
conda activate heliocot
```

### Option B: pip
```bash
pip install -r requirements.txt
```

## Expected repository structure
This repository uses a `workflow/` directory for inputs and outputs.

- `scripts/` : runnable scripts (modular steps)  
- `workflow/`  
  - `images/`  
    - `raw/`        : original RGB images (optional; typically stored on Zenodo)  
    - `resized/`    : resized images used for inference/grid evaluation  
    - `helio/`      : images with reference (red) and ROI (blue) markers for the helio step  
  - `models/`  
    - `*.pt`        : trained YOLO segmentation model weights (e.g., `best.pt`)  
  - `plot_metadata.xlsx`  
  - `outputs/`      : generated outputs (created by scripts)

## Scripts overview (what each script produces)

- `scripts/prepare_dataset.py`  
  Splits a pooled CVAT YOLO export into `train/val/test` using group-aware splitting (e.g., by `variety,stage`). Writes a fresh `data.yaml`, a split manifest (`splits_manifest.csv`), and optionally file lists (`train.txt/val.txt/test.txt`).

- `scripts/train_yolo.py`  
  Trains a YOLO segmentation model using `model/data.yaml` and produces trained weights (e.g., `best.pt`) under a `runs/` directory (Ultralytics default). Run with `--help` to see training options (epochs, imgsz, device, etc.).

- `scripts/resize_images.py`  
  Downscales RAW RGB images so the long edge is 3000 px (aspect ratio preserved). Writes resized images to `workflow/images/resized/` for faster, consistent downstream inference.

- `scripts/run_grid.py`  
  Runs inference across a full parameter grid (model × confidence × post-processing settings) and exports per-image raw/post mask pixel areas for each combination. Writes `workflow/outputs/grid_results.xlsx` and can optionally export post masks for one selected combo.

- `scripts/validate_combos.py`  
  Computes validation metrics (R², RMSE, MAE, Bias, NRMSE_mean) for all combo prediction columns against ground truth (`real_leaf_px`). Writes a summary workbook (and optional tables such as GT distribution and Top-K combos).

- `scripts/run_c112_helio_norm_emm.py`  
  Performs the core Heliocot helio step: marker-based ROI extraction (red reference + blue ROI markers), YOLO segmentation (C112 post-processing), reference-area standardization (`leaf_px_std`), and NORM mixed-model/EMM inference for the time effect. Writes `workflow/outputs/helio_<STAGE>_C112_NORM.xlsx` and can optionally export QC images and masks.

- `scripts/compute_doa_poi.py`  
  Computes two derived indices from the standardized wide table: DOA (Daily Orientation Amplitude) and POI (Peak Orientation Index). Writes `workflow/outputs/helio_E_C112_DOA_POI.xlsx` including a definitions sheet.

## Reproducible workflow (step-by-step)
The workflow is modular by design: each script produces a defined output that is used by subsequent steps.

### Step 0 — (Optional) Prepare YOLO dataset splits from a CVAT YOLO export
Use this step if you start from a CVAT YOLO export where all images/labels are pooled under `model/images/raw` and `model/labels/raw`.

Inputs:
- `model/images/raw/`
- `model/labels/raw/`
- `model/metadata.xlsx`

Outputs:
- `model/images/{train,val,test}/`
- `model/labels/{train,val,test}/`
- `model/data.yaml`
- `model/splits_manifest.csv`
- (optional) `model/train.txt`, `model/val.txt`, `model/test.txt`

Run (defaults):
```bash
python scripts/prepare_dataset.py
```

Run (explicit, recommended):
```bash
python scripts/prepare_dataset.py   --base-dir model   --metadata model/metadata.xlsx   --src-images model/images/raw   --src-labels model/labels/raw   --dst-images model/images   --dst-labels model/labels   --group-cols variety,stage   --train-size 0.70 --val-size 0.15 --test-size 0.15   --seed 42
```

Optional:
```bash
python scripts/prepare_dataset.py --overwrite
python scripts/prepare_dataset.py --write-filelists
```

### Step 0b — (Optional) Train the YOLO model
Train a segmentation model and place the resulting `*.pt` (e.g., `best.pt`) under `workflow/models/` for downstream steps.

Run (example):
```bash
python scripts/train_yolo.py --help
```

### Step 1 — Resize RAW images (optional but recommended)
Resizes images so that the long edge is 3000 px (aspect ratio preserved).

Input:
- `workflow/images/raw/*.jpg`

Output:
- `workflow/images/resized/*.jpg`

Run:
```bash
python scripts/resize_images.py
```

### Step 2 — Full grid evaluation (model × confidence × post-processing)
Runs inference across a parameter grid and exports raw/post pixel areas for each combination.

Inputs:
- `workflow/images/resized/`
- `workflow/plot_metadata.xlsx`
- `workflow/models/*.pt`

Outputs:
- `workflow/outputs/grid_results.xlsx`  
  Sheets: `predictions`, `combinations`

Run (defaults):
```bash
python scripts/run_grid.py
```

Run (explicit):
```bash
python scripts/run_grid.py --workflow-dir workflow
```

Optional (export masks only for one combo, e.g., C112):
```bash
python scripts/run_grid.py --save-masks-combo C112
```

### Step 3 — Model validation against ground-truth pixels
Computes validation metrics (R2, RMSE, MAE, Bias, NRMSE_mean) for all combo columns against `real_leaf_px`.

Input:
- `workflow/for_validation.xlsx`  
  Must include:
  - ground truth column: `real_leaf_px`
  - prediction columns like `C001_raw_px`, `C001_post_px`, ...

Output:
- `workflow/outputs/validation_summary.xlsx` (user-defined)

Run (recommended):
```bash
python scripts/validate_combos.py   --in-xlsx workflow/for_validation.xlsx   --out-xlsx workflow/outputs/validation_summary.xlsx   --stage E   --gt-col real_leaf_px   --write-tables   --topk 20   --primary-sort post_NRMSE_mean   --secondary-sort post_R2
```

### Step 4 — C112 helio processing (marker-based ROI) + NORM + EMM (time effect)
Performs marker-based ROI extraction, YOLO segmentation (C112 settings), reference-area standardization, and time-effect inference (NORM + EMM).

Inputs:
- `workflow/images/helio/` (images containing red reference markers and blue ROI markers)
- `workflow/plot_metadata.xlsx`
- `workflow/models/*.pt` (weights; default = first `*.pt` under `workflow/models`)

Outputs:
- `workflow/outputs/helio_E_C112_NORM.xlsx`  
  Includes `wide_by_time` sheet with `leaf_px_std_1..5`

Run (minimal):
```bash
python scripts/run_c112_helio_norm_emm.py --stage E
```

Run (recommended; also saves QC images and/or masks):
```bash
python scripts/run_c112_helio_norm_emm.py   --workflow-dir workflow   --stage E   --device cpu   --imgsz 768   --conf 0.6   --save-viz   --save-debug   --export-masks
```

Notes on QC outputs:
- `--save-viz` writes `*_viz.jpg` into `workflow/outputs/qc/`
- `--save-debug` writes `*_dbg.jpg` into `workflow/outputs/qc/`
- `--export-masks` writes final post masks into `workflow/outputs/masks/`

### Step 5 — Derived indices (DOA, POI)
Computes two indices from the standardized wide table:
- DOA: Daily Orientation Amplitude (max–min of the 5-point normalized series)  
- POI: Peak Orientation Index (3-point peak deviation; t1–t3–t5)

Input:
- `workflow/outputs/helio_E_C112_NORM.xlsx` (Sheet: `wide_by_time`)

Output:
- `workflow/outputs/helio_E_C112_DOA_POI.xlsx`

Run (defaults):
```bash
python scripts/compute_doa_poi.py
```

Run (explicit):
```bash
python scripts/compute_doa_poi.py   --in-xlsx workflow/outputs/helio_E_C112_NORM.xlsx   --sheet wide_by_time   --out-xlsx workflow/outputs/helio_E_C112_DOA_POI.xlsx
```

## Notes
- Device selection depends on your setup (`cpu`, `mps`, `cuda:0`). Use `--device` where supported (e.g., in Step 4).
- Large files (images, trained weights, derived outputs) are typically stored on Zenodo rather than GitHub.

## Citation
Please use the repository’s `CITATION.cff` for citing this workflow. If you reuse the code and/or data, cite the corresponding Zenodo records listed above.
