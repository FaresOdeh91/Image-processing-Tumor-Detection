"""
Generic, modality-independent tumor region detection pipeline.

    load -> denoise -> enhance -> [modality cleanup] -> segment -> post-process -> measure

Pure image processing only: filtering, histogram methods, morphology,
region growing and watershed. No training, no learned models.
Only the `cleanup` step and the parameters come from the modality module.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import color, exposure, filters, io, measure, morphology, segmentation, util


# ----------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------
def load_image(path, channel=None):
    """Load an image as a 2-D float array in [0, 1].

    channel: for multi-channel images, pick one channel (e.g. 1 = FLAIR in the
    Kaggle LGG dataset) instead of converting to grayscale.
    """
    img = io.imread(path)
    if img.ndim == 3:
        if img.shape[-1] == 4:                       # drop alpha
            img = img[..., :3]
        img = img[..., channel] if channel is not None else color.rgb2gray(img)
    img = util.img_as_float(img).astype(np.float64)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


# ----------------------------------------------------------------------------
# 2. Denoise
# ----------------------------------------------------------------------------
def denoise(img, method="median", size=3):
    if method == "median":
        return ndi.median_filter(img, size=size)
    if method == "gaussian":
        return filters.gaussian(img, sigma=size / 3)
    return img


# ----------------------------------------------------------------------------
# 3. Enhance
# ----------------------------------------------------------------------------
def enhance(img, method="clahe", clip_limit=0.01):
    if method == "clahe":
        return exposure.equalize_adapthist(img, clip_limit=clip_limit)
    if method == "hist_eq":
        return exposure.equalize_hist(img)
    if method == "stretch":
        p2, p98 = np.percentile(img, (2, 98))
        return exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1))
    return img


# ----------------------------------------------------------------------------
# 5. Segment (inside the region of interest returned by cleanup)
# ----------------------------------------------------------------------------
def segment(img, roi, method="otsu", **p):
    p.pop("method", None)
    if not roi.any():
        return np.zeros_like(roi)
    vals = img[roi]

    if method == "otsu":
        # Multi-level Otsu: brain tissue has several intensity classes
        # (CSF / gray / white matter / lesion); the tumor is the brightest class.
        classes = p.get("otsu_classes", 4)
        thr = filters.threshold_multiotsu(vals, classes=classes)[-1]
        return roi & (img > thr)

    if method == "region_growing":
        # Seed = brightest point after heavy smoothing (avoids single noisy pixels).
        smooth = filters.gaussian(img, sigma=p.get("seed_sigma", 3))
        smooth[~roi] = 0
        seed = np.unravel_index(np.argmax(smooth), img.shape)
        tol = p.get("rg_tolerance", 0.15)
        grown = segmentation.flood(img, seed, tolerance=tol)
        return grown & roi

    if method == "watershed":
        # Markers from intensity, flooding on the gradient magnitude.
        thrs = filters.threshold_multiotsu(vals, classes=p.get("otsu_classes", 4))
        markers = np.zeros(img.shape, dtype=np.int32)
        markers[roi & (img < thrs[0])] = 1                       # background tissue
        markers[roi & (img > thrs[-1] + p.get("ws_margin", 0.05))] = 2   # sure tumor
        markers[~roi] = 1
        if not (markers == 2).any():
            return np.zeros_like(roi)
        labels = segmentation.watershed(filters.sobel(img), markers)
        return (labels == 2) & roi

    raise ValueError(f"unknown segmentation method: {method}")


# ----------------------------------------------------------------------------
# 6. Post-process: clean the mask and keep only plausible tumor candidates
# ----------------------------------------------------------------------------
def postprocess(mask, img, roi, open_radius=2, min_area=80, max_area_frac=0.5,
                min_solidity=0.6, min_contrast=1.5, keep="best"):
    """Rule-based candidate filtering.

    A component is kept only if it is:
      - big enough (min_area pixels) and not most of the ROI (max_area_frac),
      - compact (solidity >= min_solidity),
      - clearly brighter than the surrounding tissue:
            mean(candidate) >= mean(roi) + min_contrast * std(roi)
    If nothing passes, the image is reported as "no tumor found".
    """
    se = morphology.disk(open_radius)
    m = ndi.binary_opening(mask, structure=se)
    m = ndi.binary_closing(m, structure=se, border_value=0)
    m = ndi.binary_fill_holes(m)
    labels = measure.label(m)
    sizes = np.bincount(labels.ravel())
    m = (sizes[labels] >= min_area) & m              # drop small objects

    roi_area = roi.sum()
    roi_mean, roi_std = img[roi].mean(), img[roi].std()
    labels = measure.label(m)
    candidates = []
    for r in measure.regionprops(labels, intensity_image=img):
        if r.area > max_area_frac * roi_area:
            continue
        if r.solidity < min_solidity:
            continue
        if r.intensity_mean < roi_mean + min_contrast * roi_std:
            continue
        candidates.append(r)

    out = np.zeros_like(mask, dtype=bool)
    if not candidates:
        return out
    if keep == "best":
        # score = brightness x size: favours the dominant bright lesion
        best = max(candidates, key=lambda r: r.intensity_mean * np.sqrt(r.area))
        candidates = [best]
    for r in candidates:
        out[labels == r.label] = True
    return out


# ----------------------------------------------------------------------------
# 7. Measure
# ----------------------------------------------------------------------------
def measure_tumor(mask, pixel_spacing_mm=None):
    if not mask.any():
        return {"detected": False}
    r = max(measure.regionprops(measure.label(mask)), key=lambda r: r.area)
    out = {
        "detected": True,
        "area_px": int(mask.sum()),
        "centroid_rc": tuple(round(float(c), 1) for c in r.centroid),
        "bbox_rc": tuple(int(b) for b in r.bbox),
        "equiv_diameter_px": round(float(r.equivalent_diameter_area), 1),
        "solidity": round(float(r.solidity), 3),
    }
    if pixel_spacing_mm:
        out["area_mm2"] = round(out["area_px"] * pixel_spacing_mm ** 2, 1)
        out["equiv_diameter_mm"] = round(out["equiv_diameter_px"] * pixel_spacing_mm, 1)
    return out


# ----------------------------------------------------------------------------
# Pipeline runner
# ----------------------------------------------------------------------------
@dataclass
class Result:
    original: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    roi: np.ndarray
    raw_mask: np.ndarray
    mask: np.ndarray
    measurements: dict = field(default_factory=dict)


def run(img, modality, method=None, overrides=None):
    """Run the full pipeline on a [0,1] grayscale image.

    modality : module from `modalities/` exposing PARAMS (dict) and cleanup(img, params)
    method   : segmentation method, overrides PARAMS['method']
    overrides: dict of parameter overrides (e.g. from the command line)
    """
    p = dict(modality.PARAMS)
    p.update(overrides or {})
    if method:
        p["method"] = method

    den = denoise(img, p["denoise"], p["denoise_size"])
    enh = enhance(den, p["enhance"], p["clahe_clip"])
    enh_clean, roi = modality.cleanup(enh, p)
    raw = segment(enh_clean, roi, **p)
    mask = postprocess(raw, enh_clean, roi,
                       open_radius=p["open_radius"], min_area=p["min_area"],
                       max_area_frac=p["max_area_frac"], min_solidity=p["min_solidity"],
                       min_contrast=p["min_contrast"], keep=p["keep"])
    return Result(img, den, enh, roi, raw, mask,
                  measure_tumor(mask, p.get("pixel_spacing_mm")))
