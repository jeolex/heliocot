# Heliocot

Heliocot is an open, reproducible segmentation-to-PLA workflow for top-down field RGB images of cotton. It converts instance segmentation masks into reference-area standardized projected leaf area (PLA) time series and supports quantifying diurnal canopy orientation dynamics.

## What this repository contains
- Modular scripts for dataset preparation, training, inference/post-processing, and PLA computation  
- Utilities to generate reference-area standardized PLA time series and selected derived indices (DOA, POI)  
- Statistical analysis outputs for time effects (mixed-effects models / estimated marginal means; EMMs)

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
This repository uses a `workflow/` directory for inputs/outputs and a `model/` directory for training assets.

- `scripts/` : runnable scripts (modular steps)
- `model/`   : training assets (YOLO format)
  - `metadata.xlsx`
  - `images/raw/` and `labels/raw/` (pooled CVAT YOLO export before splitting)
  - `images/train|val|test/` and `labels/train|val|test/` (created by `prepare_dataset.py`)
  - `data.yaml` (created by `prepare_dataset.py`)
- `workflow/`
  - `images/`
    - `raw/`        : original RGB images (optional; typically stored on Zenodo)
    - `resized/`    : resized images used for inference/grid evaluation
    - `helio/`      : images with reference (red) and ROI (blue) markers for the helio step
  - `models/`
    - `*.pt`        : trained YOLO segmentation model weights (e.g., `best.pt`)
  - `plot_metadata.xlsx`
  - `outputs/`      : generated outputs (Excel tables, QC images, masks; created by scripts)

## Reproducible workflow (step-by-step)
The workflow is modular by design: each script produces a defined output that is used by subsequent steps. Each step below includes a brief description (1–2 sentences) and a recommended run command.

---

### Step 0 — (Optional) Prepare YOLO dataset splits from a CVAT YOLO export
**Script:** `scripts/prepare_dataset.py`  
Splits a pooled CVAT YOLO export into train/val/test subsets using group-aware splitting (default grouping: `variety,stage`). It writes `model/data.yaml` plus an audit manifest (`model/splits_manifest.csv`), and optionally `train.txt/val.txt/test.txt`.

**Inputs (defaults):**
- `model/metadata.xlsx`
- `model/images/raw/` and `model/labels/raw/` (pooled images/labels)

**Outputs:**
- `model/images/train|val|test/` and `model/labels/train|val|test/`
- `model/data.yaml`
- `model/splits_manifest.csv`
- (optional) `model/train.txt`, `model/val.txt`, `model/test.txt`

**Run (recommended):**
```bash
python scripts/prepare_dataset.py   --base-dir model   --metadata model/metadata.xlsx   --src-images model/images/raw   --src-labels model/labels/raw   --group-cols variety,stage   --train-size 0.70 --val-size 0.15 --test-size 0.15   --seed 42   --class-name leaf   --write-filelists
```

---

### Step 0b — (Optional) Train a YOLOv8 segmentation model
**Script:** `scripts/train_yolo.py`  
Trains a YOLOv8 segmentation model using Ultralytics. Outputs are written under `--project/--name` (defaults: `runs_leaf_seg/yolov8m_leafseg_final`) including `weights/best.pt`.

**Inputs (defaults):**
- `model/data.yaml` (from Step 0)
- Pretrained checkpoint: `yolov8m-seg.pt`

**Outputs (Ultralytics default structure):**
- `runs_leaf_seg/yolov8m_leafseg_final/weights/best.pt`
- training logs and plots under the same run directory

**Run (default parameters):**
```bash
python scripts/train_yolo.py
```

**Run (explicit, commonly used settings):**
```bash
python scripts/train_yolo.py   --data model/data.yaml   --pretrained yolov8m-seg.pt   --epochs 120   --imgsz 768   --batch 4   --project runs_leaf_seg   --name yolov8m_leafseg_final
```

Notes:
- `--device` can be set explicitly (e.g., `cpu`, `mps`, `cuda`). If omitted, the script auto-detects a device.
- After training, copy/select the intended weights (e.g., `best.pt`) into `workflow/models/` for downstream inference.

---

### Step 1 — Resize RAW images (optional but recommended for efficiency)
**Script:** `scripts/resize_images.py`  
Resizes images by fixing the long edge (default 3000 px) while preserving aspect ratio. Images smaller than the target long edge are left unchanged.

**Inputs (defaults):**
- `workflow/images/raw/`

**Outputs (defaults):**
- `workflow/images/resized/`

**Run (default parameters):**
```bash
python scripts/resize_images.py
```

**Run (explicit, recursive scan and overwrite enabled):**
```bash
python scripts/resize_images.py   --raw-dir workflow/images/raw   --out-dir workflow/images/resized   --long-edge 3000   --recursive   --overwrite
```

---

### Step 2 — Full grid evaluation (model × confidence × post-processing)
**Script:** `scripts/run_grid.py`  
Runs inference over all images flagged as evaluable in `plot_metadata.xlsx` (default filter: `image_available==1` and `used_for_training==0`). For each parameter combination, it exports per-image raw/post pixel areas to `grid_results.xlsx` (plus a combinations table).

**Inputs (defaults):**
- `workflow/images/resized/`
- `workflow/plot_metadata.xlsx`
- `workflow/models/*.pt`

**Outputs (defaults):**
- `workflow/outputs/grid_results.xlsx` (Sheets: `predictions`, `combinations`)

**Run (default parameters):**
```bash
python scripts/run_grid.py
```

**Run (example: save post masks only for one combo, e.g., C112):**
```bash
python scripts/run_grid.py   --save-masks-combo C112   --overwrite-masks
```

---

### Step 3 — Model validation against ground-truth pixels
**Script:** `scripts/validate_combos.py`  
Computes validation metrics between `real_leaf_px` (ground truth) and each predicted combo column (`C###_raw_px`, `C###_post_px`). It exports a full `summary` sheet and (optionally) GT distribution + Top-K ranking tables.

**Inputs:**
- `workflow/for_validation.xlsx` (must include `real_leaf_px` and combo columns such as `C001_post_px`)

**Outputs (example default path):**
- `workflow/outputs/validation_summary.xlsx`

**Run (recommended):**
```bash
python scripts/validate_combos.py   --in-xlsx workflow/for_validation.xlsx   --out-xlsx workflow/outputs/validation_summary.xlsx   --stage E   --gt-col real_leaf_px   --write-tables   --topk 20   --primary-sort post_NRMSE_mean   --secondary-sort post_R2
```

Notes:
- If the ground-truth sheet is not the first sheet, add `--sheet <sheetname>`.

---

### Step 4 — C112 helio processing (marker-based ROI) + NORM + EMM (time effect)
**Script:** `scripts/run_c112_helio_norm_emm.py`  
Performs marker-based ROI extraction (red reference markers; blue ROI markers), YOLO segmentation using fixed C112 post-processing parameters, reference-area standardization, then time-effect inference (NORM + EMM) from the `wide_by_time` table.

**Inputs (defaults):**
- `workflow/images/helio/`
- `workflow/plot_metadata.xlsx`
- `workflow/models/*.pt` (weights; default = first `*.pt` under `workflow/models/`)

**Outputs (default):**
- `workflow/outputs/helio_E_C112_NORM.xlsx`  
  Includes `wide_by_time` with `leaf_px_std_1..5` plus model/EMM tables.

**Run (recommended; with QC and mask exports enabled):**
```bash
python scripts/run_c112_helio_norm_emm.py   --workflow-dir workflow   --stage E   --device cpu   --imgsz 768   --conf 0.6   --save-viz   --save-debug   --export-masks
```

Notes on QC outputs:
- `--save-viz` writes `*_viz.jpg` into `workflow/outputs/qc/`
- `--save-debug` writes `*_dbg.jpg` into `workflow/outputs/qc/`
- `--export-masks` writes final post masks into `workflow/outputs/masks/`

---

### Step 5 — Derived indices (DOA, POI)
**Script:** `scripts/compute_doa_poi.py`  
Computes two indices from the standardized wide table: DOA (Daily Orientation Amplitude) and POI (Peak Orientation Index). The indices are appended to the table and exported to a new Excel file with definitions.

**Input (default):**
- `workflow/outputs/helio_E_C112_NORM.xlsx` (Sheet: `wide_by_time`)

**Output (default):**
- `workflow/outputs/helio_E_C112_DOA_POI.xlsx`

**Run (default parameters):**
```bash
python scripts/compute_doa_poi.py
```

**Run (explicit; recommended defaults):**
```bash
python scripts/compute_doa_poi.py   --in-xlsx workflow/outputs/helio_E_C112_NORM.xlsx   --sheet wide_by_time   --out-xlsx workflow/outputs/helio_E_C112_DOA_POI.xlsx   --norm5 mean   --norm3 mean
```

---

## Notes
- For Apple Silicon, `--device mps` can accelerate Ultralytics where supported. For NVIDIA GPUs, use `--device cuda`.
- Ultralytics may create `runs/` directories during training/inference. These are not required for the workflow and can be ignored (and are typically excluded via `.gitignore`).

## Citation
This repository includes a `CITATION.cff` file for GitHub citation support. Please also cite the Zenodo records listed above when using the workflow and/or dataset.
