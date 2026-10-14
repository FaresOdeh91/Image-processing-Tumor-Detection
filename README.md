# Tumor Region Detection with Pure Image Processing

Detects and outlines tumor regions in brain MRI slices using only classical
image processing: filtering, histogram methods, morphology, region growing and
watershed. **No machine learning, no training, no learned models.**

> It finds *suspicious regions* (abnormally bright, compact areas). It does
> not diagnose cancer and is not for clinical use.

## Pipeline

```
load → denoise → enhance → [modality cleanup] → segment → post-process → measure
        median     CLAHE     skull stripping     Otsu /       morphology +   area,
        filter                (brain MRI)        region grow / rule filters   centroid,
                                                 watershed                    diameter
```

| File | Purpose |
|---|---|
| `pipeline.py` | Generic steps that work for every image type |
| `modalities/brain_mri.py` | Brain-MRI parameters and skull stripping (the only image-type-specific step) |
| `modalities/__init__.py` | Registry: where you add new image types |
| `evaluate.py` | Dice, IoU, sensitivity and specificity |
| `visualize.py` | Figure showing all six stages |
| `synthetic.py` | Phantom images with known masks, for testing |
| `main.py` | Command-line interface |

### Segmentation methods compared
- **otsu**: multi-level Otsu (4 classes: CSF, gray matter, white matter, lesion); the brightest class is the candidate.
- **region_growing**: seed at the brightest smoothed point, then flood-fill neighbors within an intensity tolerance.
- **watershed**: markers from the Otsu levels, flooded over the gradient (Sobel) image.

### Rule-based candidate filtering (post-processing)
A region is reported as a tumor only if it:
- is larger than `min_area` and smaller than `max_area_frac` of the brain,
- is compact (`solidity ≥ min_solidity`),
- is clearly brighter than the brain (`mean ≥ brain_mean + min_contrast × brain_std`).

If no region passes, the result is "no tumor detected". This lets the system handle healthy slices.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# 1. Test on synthetic phantoms (no dataset needed)
python main.py demo

# 2. Run on a single image, comparing all methods
python main.py detect slice.png --method all
python main.py detect slice.tif --channel 1 --gt slice_mask.tif   # with ground truth

# 3. Evaluate on a full dataset
python main.py evaluate path/to/kaggle_3m --channel 1 --method all
```
Outputs go to `results/`: step-by-step figures, `per_image_results.csv` and `summary.json`.

Useful flags: `--min-area`, `--min-contrast`, `--pixel-spacing` (mm per pixel, to report area in mm²), and `--limit N`.

## Dataset
Recommended: **Kaggle "Brain MRI segmentation" (LGG, `mateuszbuda/lgg-mri-segmentation`)**.
- Each slice `X.tif` has a ground-truth mask `X_mask.tif` in the same folder, which `evaluate` pairs automatically.
- The images are 3-channel (pre-contrast, FLAIR, post-contrast). Use `--channel 1` for FLAIR, where tumors are brightest.
- Many slices contain no tumor, which tests your "no tumor" rules (specificity).

Other datasets: if the masks are in a separate folder with the same file names, use `--masks path/to/masks`.

## Demo results (synthetic phantoms)
| Method | Sensitivity | Specificity | Dice | IoU |
|---|---|---|---|---|
| Otsu | 1.00 | 1.00 | 0.962 | 0.927 |
| Region growing | 1.00 | 1.00 | 0.915 | 0.844 |
| Watershed | 1.00 | 1.00 | 0.965 | 0.932 |

The phantoms are deliberately easy. Expect noticeably lower scores on real MRI. Analyzing *why* (faint tumors, tumors touching the skull, edema, intensity variation between scans) is the most valuable part of your report.

## Adding another image type (e.g. mammograms)
1. Create `modalities/mammogram.py` with a `PARAMS` dict and a `cleanup(img, params)` function that returns `(cleaned_image, roi_mask)` (for mammograms: remove the background, labels and pectoral muscle).
2. Register it in `modalities/__init__.py`.
3. Run `python main.py evaluate ... --modality mammogram`.

The generic steps in `pipeline.py` don't change.

## Known limitations
- Skull stripping by erosion is crude. It can cut off tumors at the brain's edge, or leave bits of skull on oddly shaped slices.
- It assumes tumors are brighter than the surrounding tissue (true for FLAIR/T2, not always for T1).
- It reports one main region per slice (`keep="best"`); set `keep` to `"all"` to keep every candidate.
- Parameters are tuned by hand. Document how you chose them and keep a separate set of test images you didn't tune on.
