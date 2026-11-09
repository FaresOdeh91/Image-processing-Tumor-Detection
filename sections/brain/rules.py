"""
Rule-based brain tumor type suggestion: glioma / meningioma / pituitary.

No classifier and no training. The region found by the detector is measured
(where it sits in the head, how big, how bright, how even, how regular) and
hand-written rules taken from radiology descriptions turn the measurements
into points for each type:

  * Pituitary tumor - grows from the pituitary gland in the sella turcica, at
    the centre of the skull base: small, round, close to the head's centre.
  * Meningioma - grows from the meninges (the brain's lining), outside the
    brain tissue: touches the inner skull, enhances strongly and evenly.
  * Glioma - grows inside the brain tissue: away from the skull, often large,
    irregular, unevenly enhancing (a dark necrotic core with a bright rim).

The type with the most points wins, and the reasons are reported.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

TYPES = ["glioma", "meningioma", "pituitary"]


def features(img, mask, head, enh=None, dark_level=None):
    """Measurements of the tumor region, all normalised so they do not depend
    on image size or scanner brightness.

    enh, dark_level: the contrast-enhanced image and the brightness of normal grey
    matter (the second Otsu cut-off); when given, `dark_share` is the part of the
    tumor darker than grey matter (fluid, dead or non-enhancing tissue)."""
    head_area = head.sum()
    head_r = np.sqrt(head_area / np.pi)                    # radius of an equal-area disk
    hc = np.array(ndi.center_of_mass(head))
    lab = measure.label(mask)
    r = max(measure.regionprops(lab), key=lambda r: r.area)
    tumor = lab == r.label
    tc = np.array(r.centroid)

    # distance of every head pixel to the outside of the head (i.e. to the scalp)
    depth = ndi.distance_transform_edt(head)
    vals = img[tumor]
    brain_ref = np.median(img[head & ~tumor])
    f = {
        "size": r.area / head_area,                         # share of the head
        "center_dist": float(np.linalg.norm(tc - hc) / head_r),
        "min_depth": float(depth[tumor].min() / head_r),    # 0 = touches the scalp
        "brightness": float(vals.mean() / max(brain_ref, 1e-6)),
        "unevenness": float(vals.std() / max(vals.mean(), 1e-6)),
        "solidity": float(r.solidity),
        "eccentricity": float(r.eccentricity),
    }
    if enh is not None and dark_level is not None:
        f["dark_share"] = float((enh[tumor] < dark_level).mean())
    return f


# Each rule: (type, test on the features, points, reason shown to the user).
# Thresholds are hand-set from the radiology description of each type and were
# checked on the development patients only (folds 1-2), never on the test ones.
RULES = [
    ("pituitary", lambda f: f["center_dist"] < 0.30, 2,
     "it sits near the centre of the head (where the sella / pituitary gland is): "
     "{center_dist:.2f} of the head radius from the centre, under 0.30"),
    ("pituitary", lambda f: f["size"] < 0.02, 1,
     "it is small ({size:.1%} of the head, under 2%)"),
    ("meningioma", lambda f: f["brightness"] >= 1.25, 1,
     "it lights up strongly with contrast ({brightness:.2f}× normal tissue, at least 1.25×)"),
    ("meningioma", lambda f: f["unevenness"] <= 0.22, 1,
     "its brightness is even inside ({unevenness:.2f} spread, at most 0.22)"),
    ("meningioma", lambda f: f["center_dist"] >= 0.55, 1,
     "it lies out towards the skull ({center_dist:.2f} of the head radius from the centre, "
     "at least 0.55), where the brain's lining is"),
    ("glioma", lambda f: f["brightness"] < 1.15, 2,
     "it enhances only weakly ({brightness:.2f}× normal tissue, under 1.15×)"),
    ("glioma", lambda f: f["unevenness"] > 0.25, 1,
     "its brightness is patchy inside ({unevenness:.2f} spread, over 0.25), e.g. a dark "
     "core with a bright rim"),
    ("glioma", lambda f: f.get("dark_share", 0) >= 0.30, 1,
     "{dark_share:.0%} of it is darker than normal grey matter (fluid, dead or non-enhancing "
     "tissue; at least 30%), while meningiomas and pituitary tumors light up with contrast"),
    ("glioma", lambda f: 0.30 <= f["center_dist"] < 0.55, 1,
     "it lies within the brain tissue, neither at the centre nor out at the skull "
     "({center_dist:.2f} of the head radius from the centre)"),
]

# Rules that rule a type out entirely.
VETOES = [
    ("pituitary", lambda f: f["center_dist"] > 0.40,
     "a pituitary tumor cannot be this far from the centre of the head"),
]

# If points tie, the more common type is suggested (glioma is the most frequent).
TIE_ORDER = ["glioma", "pituitary", "meningioma"]


def score(f):
    scores = {t: 0 for t in TYPES}
    reasons = {t: [] for t in TYPES}
    for t, test, pts, text in RULES:
        if test(f):
            scores[t] += pts
            reasons[t].append(f"+{pts}: " + text.format(**f))
    vetoed = []
    for t, test, text in VETOES:
        if test(f):
            scores[t] = -1
            reasons[t].append("ruled out: " + text)
            vetoed.append(t)
    best = max(TYPES, key=lambda t: (scores[t], -TIE_ORDER.index(t)))
    return best, scores, reasons


def classify(img, mask, head, enh=None, dark_level=None):
    f = features(img, mask, head, enh, dark_level)
    label, scores, reasons = score(f)
    ranked = sorted(TYPES, key=lambda t: (t != label, -scores[t]))
    margin = scores[ranked[0]] - scores[ranked[1]]
    confidence = "clear" if margin >= 2 else "weak" if margin == 1 else "tie"
    lines = [f"Suggested type: {label.upper()} ({confidence}; points "
             + ", ".join(f"{t} {max(scores[t], 0)}" for t in ranked) + ")."]
    for t in ranked:
        if reasons[t]:
            lines.append(f"{t.capitalize()}: " + "; ".join(reasons[t]) + ".")
    lines.append("Each type collects points from fixed rules about location, size, "
                 "brightness and evenness; no training or learned model is involved. "
                 "This is a suggestion from imaging signs, not a diagnosis.")
    return {"label": label, "scores": scores, "reasons": reasons,
            "confidence": confidence, "features": f, "explanation": "\n".join(lines)}
