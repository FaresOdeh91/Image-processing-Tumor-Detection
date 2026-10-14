# Project: Tumor region detection with pure image processing

## Hard constraint
NO machine learning, NO AI, NO trained models (no sklearn classifiers, no neural nets,
no k-means unless explicitly approved). Only classical image processing: filtering,
histogram/threshold methods, morphology, region growing, watershed, rule-based filtering.

## Scope
- Primary image type: brain MRI (axial slices, FLAIR preferred).
- The output is "suspicious region detection/segmentation", NOT cancer diagnosis.
- Planned extension: a second modality (e.g. mammograms) via `modalities/`.

## Architecture
load → denoise → enhance → [modality cleanup] → segment → post-process → measure
- `pipeline.py`: generic steps (modality independent). `run()` is the entry point.
- `modalities/<name>.py`: `PARAMS` dict + `cleanup(img, params) -> (img, roi_mask)`.
  Registered in `modalities/__init__.py`. Only this layer should be modality-specific.
- `evaluate.py`: Dice, IoU, image-level TP/FP/FN/TN, sensitivity/specificity.
- `visualize.py`: 6-panel step figure. `synthetic.py`: test phantoms with GT masks.
- `main.py`: CLI with `demo`, `detect`, `evaluate` subcommands.

## Commands
- `pip install -r requirements.txt`
- `python main.py demo`: sanity check on synthetic phantoms
- `python main.py detect img.png --method all [--gt mask.png] [--channel 1]`
- `python main.py evaluate <dataset_dir> --channel 1 --method all [--limit N]`

## Notes / decisions so far
- Multi-Otsu uses 4 classes (CSF, GM, WM, lesion). With 3 classes the small tumor
  merged with white matter and Otsu/watershed missed nearly all tumors.
- Default min_contrast = 1.5 (candidate mean ≥ brain mean + 1.5·std). This removed
  false positives on tumor-free slices.
- Synthetic demo scores (Dice): Otsu 0.96, region growing 0.92, watershed 0.97.
  Not yet evaluated on real data.
- Target dataset: Kaggle LGG "Brain MRI segmentation" (X.tif + X_mask.tif pairs,
  channel 1 = FLAIR).
- Uses scipy.ndimage binary morphology (skimage binary_* functions are deprecated in 0.26).

## Next steps
1. Download the LGG dataset and run `evaluate`. Analyze failure cases.
2. Improve skull stripping (erosion is crude; tumors at the brain edge get cut).
3. Possibly add a symmetry-based cue (left/right hemisphere difference), still rule-based.
4. Add a second modality as the extension.
