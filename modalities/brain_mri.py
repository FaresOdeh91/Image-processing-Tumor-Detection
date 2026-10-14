"""
Brain MRI (axial slices, T2 / FLAIR / contrast T1).

Modality-specific step: skull stripping.
The skull and scalp are bright and would be picked up as "tumor" by any
brightness-based segmentation, so they are removed first using
thresholding + morphology only.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

PARAMS = {
    # preprocessing
    "denoise": "median",
    "denoise_size": 3,
    "enhance": "clahe",
    "clahe_clip": 0.01,
    # skull stripping
    "head_thresh_frac": 0.5,   # head = pixels above this fraction of Otsu threshold
    "skull_erosion_frac": 0.07,  # erosion radius as a fraction of the head's diameter
    # segmentation
    "method": "otsu",          # otsu | region_growing | watershed
    "otsu_classes": 4,
    "rg_tolerance": 0.15,
    "seed_sigma": 3,
    "ws_margin": 0.05,
    # post-processing rules
    "open_radius": 2,
    "min_area": 80,
    "max_area_frac": 0.5,
    "min_solidity": 0.6,
    "min_contrast": 1.5,
    "keep": "best",
    "pixel_spacing_mm": None,
}


def _largest_component(mask):
    lab = measure.label(mask)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def skull_strip(img, head_thresh_frac=0.5, skull_erosion_frac=0.07):
    """Return a boolean brain mask.

    1. Head mask: threshold well below Otsu (captures everything that is not air),
       fill holes, keep the largest connected component.
    2. Erode by a radius proportional to the head size to peel off scalp + skull.
    3. Keep the largest remaining component (the brain) and smooth its border.
    """
    head = img > head_thresh_frac * filters.threshold_otsu(img)
    head = ndi.binary_fill_holes(_largest_component(head))
    if not head.any():
        return head

    diameter = 2 * np.sqrt(head.sum() / np.pi)
    r = max(2, int(skull_erosion_frac * diameter))
    brain = ndi.binary_erosion(head, structure=morphology.disk(r))
    brain = _largest_component(brain)
    brain = ndi.binary_opening(brain, structure=morphology.disk(max(1, r // 3)))
    brain = ndi.binary_fill_holes(brain)
    return brain


def cleanup(img, params):
    """Modality hook called by the generic pipeline.

    Returns (image_with_non_brain_zeroed, roi_mask).
    """
    roi = skull_strip(img, params["head_thresh_frac"], params["skull_erosion_frac"])
    out = img.copy()
    out[~roi] = 0
    return out, roi
