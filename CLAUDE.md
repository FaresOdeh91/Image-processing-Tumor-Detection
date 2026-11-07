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

## Results (2026-10-14)
Test patients (folds 3-5, 1,843 slices), default `thick_blob`:
- tumor found in the right place 46% (843), Dice 0.27; type right when found 64%
- type rules alone on the expert outline: 81% (dev: 82%; always-glioma: 48%)
- healthy slices: 2 / 22 false alarms
Original `otsu` (brightest blob x size) on the same test patients: 25% found,
Dice 0.15, 4 / 22 healthy false alarms.

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
- Tumors darker than brain (cystic / non-enhancing gliomas) are invisible to a
  brightness detector by design.
- Multi-Otsu 4 classes; scipy.ndimage binary morphology (skimage binary_* deprecated).

## Next steps
1. Breast section (mammograms): find a mini-MIAS source, `sections/breast/`.
2. Brain: plane-aware rules (axial/coronal/sagittal) for location features.
3. Brain: a separate "dark lesion" search for non-enhancing tumors.
