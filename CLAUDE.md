# Project: Tumor detection and typing with pure image processing

## Hard constraint
NO machine learning, NO AI, NO trained models (no sklearn classifiers, no neural nets,
no k-means unless explicitly approved). Only classical image processing: filtering,
histogram/threshold methods, morphology, region growing, watershed, rule-based filtering.
Rule thresholds are hand-set; checking them on development data is fine, fitting
them automatically (grid search against labels) is not.

## Working style (owner's request)
After every step of work, give a short plain-language summary of what the step
does and *why it works without AI or training*: which rule, filter or measurement
it uses. The owner needs to be able to explain the system.

## Scope
- Detect the tumor region AND suggest its type using hand-written rules on
  measured features (location, size, brightness, evenness). No classifier.
- Output is a "suspicious region" plus a rule-based "suggested type", NOT a diagnosis.
- One **section** per body part, each with its own full pipeline. Built: brain.
  Next: breast (mammograms, mini-MIAS). The official mini-MIAS host
  (peipa.essex.ac.uk) was unreachable on 2026-10-14; find another source.

## Architecture
- `sections/common.py`: shared blocks (load, median/gaussian denoise, CLAHE,
  morphology helpers, measurements) and the `Step` / `Result` records. Every step
  carries a plain-language explanation that the app and CLI print.
- `sections/<name>/`: one package per body part exposing NAME, TITLE, DESCRIPTION,
  METHODS, TYPES, PARAMS, `run(img, method, overrides) -> Result`, and optionally
  `classify_region(img, mask)` (type rules on a given outline). Registered in
  `sections/__init__.py`.
- `sections/brain/detect.py`: skull strip, brain outline, segmentation methods,
  candidate filtering, hysteresis growth. `rules.py`: features + type rules.
- `app.py`: Streamlit app (pick section, upload or sample, see every step).
- `main.py`: CLI `demo` / `detect` / `evaluate` (`--section`, `--folds`).
- `evaluate.py`: Dice/IoU, outcomes TP / WRONG / FN / FP / TN, type accuracy.
- `scripts/`: dataset conversion and sample building.

## Commands
- Environment: `.venv` (Python 3.10). `.venv/Scripts/python -m pip install -r requirements.txt`
- App: `.venv/Scripts/python -m streamlit run app.py` (port 8501)
- `python main.py evaluate data/figshare --section brain --folds 3,4,5 --method thick_blob`
- `python main.py evaluate data/healthy --section brain` (false alarms on healthy slices)

## Data (all under `data/`, git-ignored)
- `data/figshare/`: Figshare brain tumor set (Cheng et al.): 3,064 contrast T1
  slices, 233 patients, masks + type labels, mixed axial/coronal/sagittal.
  Built by `scripts/convert_figshare.py`. `labels.csv` has patient-wise folds 1-5.
  **Folds 1-2 = development (look at these), folds 3-5 = test (only for final runs).**
- `data/healthy/`: 22 tumor-free axial slices from 2 OpenNeuro ds000001 volunteers
  (non-contrast T1), empty masks. `scripts/make_healthy_set.py`.
- `samples/`: 3 test-patient tumors (one per type) + 2 healthy slices for the app.

## Publishing
- GitHub (public): https://github.com/FaresOdeh91/Image-processing-Tumor-Detection,
  branch `main`. Remote URL carries the username (`https://FaresOdeh91@github.com/...`)
  so Git Credential Manager uses FaresOdeh91; the machine's default GitHub login is
  erptechno26 (TechnoCPT), which has no write access here. Don't change that login.
- Google Drive copy: `G:\My Drive\FARES ODEH TUMOR DETECTION\` (Drive for desktop),
  shared with the owner's professor. Refresh it from `git archive HEAD` after pushes.
- Slides: `slides/NN-name/` (png/pdf/html/notes) exported with
  `scripts/export_slide.py` from the Slides artifacts
  (intro https://claude.ai/artifact/2wVJmU6GuR7G8toHNMRw3k,
  how-it-works https://claude.ai/artifact/WKuD6mALJt8NXpLmWHbwib).

## Results (test patients, folds 3-5, 1,843 slices; healthy set 22 slices)
| | fluid step (2026-11-09, default) | before it (2026-10-14) | original `otsu` |
|---|---|---|---|
| tumor found in right place | 50.5% (glioma 40, menin 57, pit 61) | 46% (glioma 29) | 25% |
| Dice | 0.30 | 0.27 | 0.15 |
| type right when found | 71% | 64% | 62% |
| type rules on expert outline | 82.5% (glioma recall 92%) | 81% | |
| healthy false alarms | 2/22 | 2/22 | 4/22 |

## Notes / decisions
- Detection, not typing, is the bottleneck. The brightest Otsu class on contrast T1
  contains vessels, dura, scalp fat and face/neck tissue. What separates the tumor:
  it is THICK (max inscribed radius), never touches the search-area edge, sits
  nearer the head centre, and lies inside the brain outline. `thick_blob` scores
  brightness x thickness x centrality x share-inside-brain.
- Brain outline (`brain_outline`): drop darkest class, erode 2% of head diameter
  to snap bridges to face/neck, keep largest, grow back, widen by 6% (meningiomas
  sit outside the brain). Used as a soft score, NOT as the search area: as a hard
  search area it kept only 61% of tumor pixels.
- min_contrast 1.5 kept. 1.0 found a few % more tumors but false-alarmed on 16/22
  healthy slices.
- Tried and rejected: LoG scale-space blob seeds (seed in tumor 3-23%), tight
  brain mask as search area, hysteresis reach 0.1 (overgrows).
- Fluid (`find_fluid_lesion`, step "6c"): 86% of dev gliomas are >=30% darker than
  grey matter (meningioma 3%, pituitary 6%). Dark fluid-level regions (darkest Otsu
  cut) are tumor candidates if round (solidity >= 0.8), thick, inside the brain
  outline, off the edge, and NOT mirrored: `mirror_dark_share` flips the region
  about the head's vertical midline; > 0.5 dark = normal anatomy (ventricles).
  Without the mirror rule healthy false alarms went 2 -> 13/22; with it 2/22.
  A fluid lesion wins if thicker than the bright region; a touching bright wall is
  merged. `Result.fluid` / `Step.fluid` carry the fluid mask (cyan in the app).
- Glioma type rule "+1 if dark_share >= 0.30" (darker than grey matter = 2nd Otsu
  cut): dev expert-outline accuracy 85.5% -> 88.7%. A fluid-level-only version
  barely helped (85.8%).
- Tried and rejected for fluid: adding dark regions that touch a found bright tumor
  (border contact or convex hull). On correct detections only 1-22% of what it
  added was tumor; Dice fell. Enclosed fluid was already covered by fill-holes.
- Multi-Otsu 4 classes; scipy.ndimage binary morphology (skimage binary_* deprecated).

## Next steps
1. Breast section (mammograms): find a mini-MIAS source, `sections/breast/`.
2. Brain: plane-aware rules (axial/coronal/sagittal); the mirror check is
   meaningless on sagittal slices.
3. Brain: skull-base meningiomas are often called pituitary (meningioma recall 60%).
