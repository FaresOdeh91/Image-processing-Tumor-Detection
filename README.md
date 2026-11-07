# Tumor Detection with Pure Image Processing

Finds a tumor in a medical image and suggests its type using only classical
image processing: filters, brightness thresholds, morphology (shape operations),
measurements and hand-written rules. **No machine learning, no training, no
learned model.** Every decision is a measured number compared with a rule a
person wrote down, so every result can be traced and explained.

> It reports a *suspicious region* and a *suggested type*. It is not a
> diagnosis and not for clinical use.

## Sections: one pipeline per body part

Each body part is a **section** with its own complete pipeline. You pick the
section, then give it an image of that body part.

| Section | Images | Suggests | Status |
|---|---|---|---|
| `brain` | MRI slice (contrast T1, FLAIR or T2; any orientation) | glioma / meningioma / pituitary | built |
| `breast` | mammogram | calcification / smooth mass / spiky mass | next |

### Brain pipeline

```
load → median denoise → CLAHE → skull removal → brain outline → bright regions
     → rule-based selection → hysteresis growth → type rules
```

| Step | What it does | Why it needs no AI |
|---|---|---|
| Load | Grayscale, brightness rescaled to 0–1 | Fixed formula |
| Denoise | Median filter: each pixel becomes the middle value of its 3×3 neighbours | Fixed filter |
| Enhance | CLAHE: histogram equalization tile by tile, clip-limited | Fixed filter |
| Skull removal | Threshold finds the head, erosion shrinks it inward past the skull | Threshold + morphology |
| Brain outline | Drop the darkest class (bone, fluid), snap thin bridges to face/neck by a small erosion, keep the largest piece, widen by a margin | Threshold + morphology |
| Bright regions | Multi-level Otsu splits the brightness histogram into 4 groups; keep the brightest | Otsu computes the cut-offs from this image alone |
| Selection | Reject blobs that are too small, irregular, not clearly brighter than the brain, or touching the search edge. Keep the best brightness × thickness × closeness-to-centre × share-inside-brain. Thin bright lines are blood vessels. | Written rules |
| Growth | Hysteresis (as in Canny): extend the chosen core into connected, moderately bright pixels nearby | Fixed rule |
| Type | Points per type from radiology signs: **pituitary**: near the head's centre, small. **meningioma**: very bright, evenly bright, out near the skull. **glioma**: weak or patchy brightness, inside the brain tissue. Most points wins; the reasons are listed. | Written rules |

Other detection methods (`otsu`, `region_growing`, `watershed`) remain
selectable for comparison.

## Results (brain)

Thresholds were chosen from radiology descriptions and checked on the
**development** patients only (folds 1–2). The numbers below are from the
**test** patients (folds 3–5, 1,843 slices), which were never looked at while
writing the rules.

| | `thick_blob` (default) | `otsu` (original) |
|---|---|---|
| Tumor found in the right place | **46%** (843) | 25% (461) |
| Mean overlap with expert outline (Dice) | 0.27 | 0.15 |
| Type correct, when the tumor was found | 64% | 62% |
| False alarms on 22 healthy slices | 2 | 4 |

Type rules alone, applied to the expert's outline: **81%** correct on test
patients (82% on development; always answering "glioma" would score 48%).

| true ↓ / suggested → | glioma | meningioma | pituitary | recall |
|---|---|---|---|---|
| glioma | 722 | 50 | 71 | 86% |
| meningioma | 50 | 260 | 119 | 61% |
| pituitary | 46 | 16 | 509 | 89% |

**Detection is the weak step, not typing.** Known reasons:
- Mixed orientations: on side and front views the face and neck contain bright
  fat that competes with the tumor.
- Tumors darker than the brain (cystic or non-enhancing gliomas) are invisible
  to a brightness-based search, by design.
- Skull-base meningiomas sit near the centre and are confused with pituitary
  tumors.

## Use it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows path

# App: pick a section, upload an image or pick a dataset sample
.venv/Scripts/python -m streamlit run app.py

# Command line
python main.py detect samples/tumor_meningioma_0445.png --section brain
python main.py evaluate data/figshare --section brain --folds 3,4,5
python main.py evaluate data/healthy --section brain      # false alarms
python main.py demo                                       # synthetic phantoms
```

`detect` prints each step's explanation and saves a figure of every step.
`evaluate` writes `per_image_results.csv` and `summary.json`. Outcomes are
TP (found it), WRONG (marked somewhere else), FN (found nothing), FP (alarm on a
healthy image), TN.

`samples/` holds 3 tumor slices (one per type, test patients) and 2 healthy
slices for trying the app.

## Data

| Folder | Source | Built by |
|---|---|---|
| `data/figshare/` | [Figshare brain tumor dataset](https://doi.org/10.6084/m9.figshare.1512427) (Cheng et al.): 3,064 contrast T1 slices, 233 patients, masks + type labels, patient-wise folds | `scripts/convert_figshare.py` |
| `data/healthy/` | [OpenNeuro ds000001](https://openneuro.org/datasets/ds000001), 2 healthy volunteers, 22 axial T1 slices | `scripts/make_healthy_set.py` |

`data/` is not committed. To rebuild it, download the Figshare dataset into
`data/figshare_mat/` (the four zips, unzipped, plus `cvind.mat`) and run
`python scripts/convert_figshare.py data/figshare_mat data/figshare_raw/cvind.mat data/figshare`.

**Licences.** The images in `samples/` and `slides/` come from these datasets.
The Figshare brain tumor dataset is **CC BY 4.0**: Jun Cheng, *brain tumor
dataset*, figshare, doi:10.6084/m9.figshare.1512427, and Cheng et al., "Enhanced
Performance of Brain Tumor Classification via Tumor Region Augmentation and
Partition", PLoS ONE 10(10), 2015. OpenNeuro ds000001 is **CC0**.

## Slides

One-slide presentations, with step pictures produced by the system itself, are
in [`slides/`](slides/) as PNG, PDF and a web page, with speaker notes.

## Code layout

| Path | Purpose |
|---|---|
| `sections/common.py` | Shared building blocks and the `Step` / `Result` records |
| `sections/brain/` | Brain pipeline (`__init__.py`), detection (`detect.py`), type rules (`rules.py`) |
| `sections/__init__.py` | Registry: where a new body part is added |
| `app.py` | Streamlit app |
| `main.py` | Command line |
| `evaluate.py` | Dice, IoU, outcomes, type accuracy |
| `visualize.py` | Step-by-step figures |
| `synthetic.py` | Phantom images for a quick sanity check |

### Adding a body part
Create `sections/<name>/` exposing `NAME`, `TITLE`, `DESCRIPTION`, `METHODS`,
`TYPES`, `PARAMS` and `run(img, method, overrides) -> Result`, where each step
is a `Step(title, image, explanation, overlay)`. Register it in
`sections/__init__.py`. The app and command line pick it up automatically.
