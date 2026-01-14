# Heliocot

Heliocot is an open, reproducible segmentation-to-PLA workflow for top-down field RGB images of cotton. It converts instance segmentation masks into reference-area standardized projected leaf area (PLA) time series and supports quantifying diurnal canopy orientation dynamics.

## What this repository contains
- Modular scripts for dataset preparation, inference/post-processing, and PLA computation
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
This repository uses a `workflow/` directory for inputs and outputs. A typical layout is:

- `scripts/` : runnable scripts (modular steps)
- `workflow/`
  - `images/`
    - `raw/`        : original RGB images (optional; usually stored on Zenodo)
    - `resized/`    : resized images used for inference/grid evaluation
    - `helio/`      : images with reference (red) and ROI (blue) markers for the helio step
  - `models/`
    - `*.pt`        : trained YOLO segmentation model weights (e.g., `best.pt`)
  - `plot_metadata.xlsx`
  - `outputs/`      : generated Excel outputs (created by scripts)

## Reproducible workflow (step-by-step)
The workflow is modular by design: each script produces a defined output that is used by subsequent steps.

### Step 1 — Resize RAW images (optional but recommended)
Input:
- `workflow/images/raw/*.jpg`

Output:
- `workflow/images/resized/*.jpg`

Run:
```bash
python scripts/resize_images.py
```

### Step 2 — Full grid evaluation (model × confidence × post-processing)
Input:
- `workflow/images/resized/`
- `workflow/plot_metadata.xlsx`
- `workflow/models/*.pt`

Output:
- `workflow/outputs/grid_results.xlsx`  
  (Sheets: `predictions`, `combinations`)

Run:
```bash
python scripts/run_grid.py
```

### Step 3 — Model validation against ground-truth pixels
Input:
- `workflow/outputs/for_validation.xlsx`  
  (must include `real_leaf_px` and predicted columns for all evaluated combinations)

Output:
- `workflow/outputs/validation_summary.xlsx` (or your configured output name)

Run:
```bash
python scripts/validate_combos.py
```

### Step 4 — C112 helio processing (marker-based ROI) + NORM + EMM (time effect)
Input:
- `workflow/images/helio/` (images containing red reference markers and blue ROI markers)
- `workflow/plot_metadata.xlsx`
- `workflow/models/model_4.pt` (or your chosen weights)

Output:
- `workflow/outputs/helio_E_C112_NORM.xlsx`  
  (Sheet: `wide_by_time`, including `leaf_px_std_1..5`)

Run:
```bash
python scripts/run_c112_helio_norm_emm.py
```

### Step 5 — Derived indices (DOA, POI)
Input:
- `workflow/outputs/helio_E_C112_NORM.xlsx` (Sheet: `wide_by_time`)

Output:
- `workflow/outputs/helio_E_C112_DOA_POI.xlsx`

Run:
```bash
python scripts/compute_doa_poi.py
```

## Notes
- For Apple Silicon, you may switch YOLO device to `mps` where supported (script-specific).
- Large files (images, trained weights, intermediate outputs) are typically stored on Zenodo rather than GitHub.

## Citation
A `CITATION.cff` file will be provided upon final release. Until then, please cite the associated manuscript (TBA) and the Zenodo DOIs (TBA).
