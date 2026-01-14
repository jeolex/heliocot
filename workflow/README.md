# workflow/ (local working directory)

This folder is a **local working directory** used to run the Heliocot pipeline.  
It is **intentionally excluded from GitHub** via `.gitignore` because it contains large data (images, models, outputs).

## Where to get the data
All required inputs (images, annotations/exports, metadata) and the produced outputs referenced in the manuscript are archived on Zenodo.

- **Dataset DOI:** TBA  
- **Workflow/code DOI (GitHub release archived on Zenodo):** TBA

After downloading the Zenodo dataset, place its contents into this `workflow/` folder following the structure below.

## Expected folder structure

```
workflow/
  images/
    raw/            # original high-resolution RGB images (optional if not shared)
    resized/        # resized RGB images used for inference (e.g., long edge = 3000 px)
    helio/          # marked ROI/reference images (red/blue markers) for helio analysis
  models/
    best.pt      # trained YOLO segmentation model used in the manuscript (example name)
  metadata/
    plot_metadata.xlsx
    # (optional) any additional metadata files used in the study
  outputs/
    grid_results.xlsx
    for_validation.xlsx
    validation_summary.xlsx
    helio_E_C112_NORM.xlsx
    helio_E_C112_DOA_POI.xlsx
    # other intermediate/derived outputs (as produced by scripts)
  outputs/
    masks/          # optional: saved masks for QC (if enabled)
```

Notes:
- If your dataset provides `plot_metadata.xlsx` at the top level, move/copy it into `workflow/metadata/`.
- The scripts in `scripts/` assume the repo root as the working directory and use the `workflow/` paths shown above.

## Typical run order (paper workflow)

1. **Resize images** (if needed)  
   Produces: `workflow/images/resized/`

2. **(Optional) Train model / prepare dataset**  
   Produces: `workflow/models/*.pt` and training logs (excluded from GitHub)

3. **Grid evaluation (model × postprocess params)**  
   Produces: `workflow/outputs/grid_results.xlsx`

4. **Validation vs ground-truth pixels**  
   Produces: `workflow/outputs/validation_summary.xlsx` (metrics per combination)

5. **C112 helio pipeline (marker-based ROI + PLA standardization) + Mixed model (NORM only)**  
   Produces: `workflow/outputs/helio_E_C112_NORM.xlsx` (and EMM results sheets)

6. **Derived indices** (e.g., DOA, POI)  
   Produces: `workflow/outputs/helio_E_C112_DOA_POI.xlsx`

## GitHub vs Zenodo responsibilities

- **GitHub (this repo):** code, docs, configuration files (README, CITATION, environment, scripts)  
- **Zenodo (dataset):** images, annotations, metadata, trained weights (if shared), and derived outputs
