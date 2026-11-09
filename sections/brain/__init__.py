"""
Brain section: MRI slices (FLAIR, T2 or contrast-enhanced T1).

    original -> median denoise -> CLAHE -> skull stripping -> segment
             -> rule-based candidate filter -> measure -> rule-based tumor type
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

from .. import common as c
from . import detect, rules

NAME = "brain"
TITLE = "Brain MRI"
DESCRIPTION = ("Axial, coronal or sagittal brain MRI slice. Finds the abnormally "
               "bright region and suggests glioma / meningioma / pituitary tumor "
               "from where it is and what it looks like.")
PARAMS = detect.PARAMS
METHODS = detect.METHODS
TYPES = rules.TYPES


def classify_region(img, mask, overrides=None):
    """Type rules applied to a given region (e.g. an expert's outline), to test the
    rules separately from the detector."""
    p = dict(PARAMS)
    p.update(overrides or {})
    enh = c.clahe(c.median_denoise(img, p["denoise_size"]), p["clahe_clip"])
    roi, head, _ = detect.skull_strip(enh, p["head_thresh_frac"], p["skull_erosion_frac"])
    return rules.classify(img, mask, head, enh, _grey_level(enh, roi, p))


def _grey_level(enh, roi, p):
    """Brightness of normal grey matter: the second Otsu cut-off inside the brain area."""
    if not roi.any():
        return None
    return filters.threshold_multiotsu(enh[roi], classes=p["otsu_classes"])[1]


def run(img, method=None, overrides=None):
    p = dict(PARAMS)
    p.update(overrides or {})
    if method:
        p["method"] = method
    steps = [c.Step("1. Original", img,
                    "The image is loaded as grayscale and rescaled so the darkest pixel "
                    "is 0 and the brightest is 1, making every scanner's output comparable.")]

    den = c.median_denoise(img, p["denoise_size"])
    steps.append(c.Step("2. Denoised", den,
                        f"Median filter ({p['denoise_size']}×{p['denoise_size']}): each pixel "
                        "becomes the middle value of its neighbours. Single noisy pixels are "
                        "outvoted, while real edges (tumor borders) stay sharp."))

    enh = c.clahe(den, p["clahe_clip"])
    steps.append(c.Step("3. Contrast enhanced", enh,
                        "CLAHE equalizes the brightness histogram tile by tile, so a tumor "
                        "stands out from the tissue right around it, even in darker parts of "
                        f"the scan. The clip limit ({p['clahe_clip']}) stops noise being "
                        "amplified."))

    roi, head, r = detect.skull_strip(enh, p["head_thresh_frac"], p["skull_erosion_frac"])
    brain = enh.copy()
    brain[~roi] = 0
    steps.append(c.Step("4. Skull removed", enh,
                        "The skull and scalp are bright and would look like tumor. The head is "
                        "found by thresholding (everything brighter than air), then shrunk "
                        f"inward by {r} px (morphological erosion, sized to the head) to peel "
                        "off the skull. The outline shows the brain area that is searched.",
                        overlay=roi))

    thick = p["method"] == "thick_blob"
    near = None
    if thick:
        near = detect.brain_outline(enh, head, p["brain_erosion_frac"], p["brain_margin_frac"])
        steps.append(c.Step("4b. Brain outline", enh,
                            "Bone and fluid are dark on MRI, so after removing the darkest "
                            "brightness group the brain hangs on to the face and neck only by "
                            "thin bridges. Shrinking the shape slightly snaps them, the largest "
                            "piece is kept as the brain, grown back, and widened by a margin "
                            "(tumors of the brain's lining sit just outside it). Blobs mostly "
                            "outside this outline are not trusted.", overlay=near))

    raw, why = detect.segment(brain, roi, **p)
    steps.append(c.Step("5. Bright-region segmentation", raw, why))

    mask, why = detect.filter_candidates(
        raw, brain, roi, selection="thick_central" if thick else "brightness_size",
        head=head, near_brain=near, **p)
    steps.append(c.Step("6. Rule-based filtering", img, why, overlay=mask))

    if thick and mask.any():
        low = filters.threshold_multiotsu(brain[roi], classes=p["otsu_classes"])[-2]
        mask, why = detect.grow_hysteresis(mask, brain, roi, low, head, p["grow_frac"],
                                           p["open_radius"])
        steps.append(c.Step("6b. Grow to the full tumor", img, why, overlay=mask))

    fluid = None
    if thick and p["fluid_search"] and roi.any():
        mask, fluid, why = _fluid_step(brain, roi, head, near, mask, p)
        steps.append(c.Step("6c. Fluid-filled areas", img, why, overlay=mask, fluid=fluid))

    meas = c.measure_region(mask, p.get("pixel_spacing_mm"))
    tumor_type = None
    if meas["detected"]:
        share = float(fluid.sum() / mask.sum()) if fluid is not None else 0.0
        meas["fluid_px"] = int(fluid.sum()) if fluid is not None else 0
        meas["fluid_share"] = round(share, 3)
        tumor_type = rules.classify(img, mask, head, enh, _grey_level(enh, roi, p))
        steps.append(c.Step("7. Tumor type rules", img, tumor_type["explanation"], overlay=mask,
                            fluid=fluid))
    return c.Result(NAME, steps, img, roi, mask, meas, tumor_type, fluid)


def _thickness(mask, head):
    """Radius of the widest circle that fits inside the region, relative to the head."""
    if not mask.any():
        return 0.0
    return float(ndi.distance_transform_edt(mask).max() / np.sqrt(head.sum() / np.pi))


def _fluid_step(brain, roi, head, near, mask, p):
    """Look for a fluid-filled tumor, keep it if it beats the bright region, and mark
    the fluid part of whatever is kept. Returns (tumor_mask, fluid_mask, explanation)."""
    level = filters.threshold_multiotsu(brain[roi], classes=p["otsu_classes"])[0]
    lesion, l_thick, why = detect.find_fluid_lesion(
        brain, roi, head, near, level, min_solidity=p["fluid_min_solidity"],
        max_mirror=p["fluid_mirror_max"])
    b_thick = _thickness(mask, head)
    if lesion.any() and l_thick > b_thick:
        wall = mask & ndi.binary_dilation(lesion, iterations=2)
        if wall.any():
            mask = ndi.binary_fill_holes(lesion | mask)
            why += (f" It is thicker than the bright region ({l_thick:.3f} vs {b_thick:.3f}) and "
                    f"the bright region forms its wall, so both together are the tumor.")
        else:
            mask = lesion
            why += (f" It is thicker than the best bright region ({l_thick:.3f} vs "
                    f"{b_thick:.3f}), so the fluid-filled region is taken as the tumor.")
    elif lesion.any():
        why += (f" The bright region is thicker ({b_thick:.3f} vs {l_thick:.3f}), so it stays "
                f"the tumor.")
    fluid = mask & (brain < level)
    fluid = ndi.binary_opening(fluid, structure=morphology.disk(2)) if fluid.any() else fluid
    if mask.any():
        why += (f" Fluid inside the tumor: {fluid.sum() / mask.sum():.0%} of its area "
                f"(cyan outline)." if fluid.any() else " The tumor has no fluid-filled part.")
    return mask, fluid, why
