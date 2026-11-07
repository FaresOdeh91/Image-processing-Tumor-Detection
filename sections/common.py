"""
Shared building blocks that every body-part section can pick from.

Everything here is classical image processing: filters, histogram thresholds,
morphology and measurements. Nothing is learned from data. Each section
(brain, breast, ...) chooses which blocks to use, in which order and with
which settings, and records every stage as a `Step` with a plain-language
explanation so the result can be explained without any "black box".
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import color, exposure, filters, io, measure, morphology, util


# ----------------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------------
@dataclass
class Step:
    """One stage of a section's pipeline, as shown to the user."""
    title: str
    image: np.ndarray                 # what to display (grayscale [0,1] or bool mask)
    explanation: str                  # plain language: what happened and why
    overlay: np.ndarray | None = None  # optional mask outlined on top of `image` (tumor)
    fluid: np.ndarray | None = None    # optional second mask: fluid inside the tumor


@dataclass
class Result:
    section: str
    steps: list[Step]
    original: np.ndarray
    roi: np.ndarray
    mask: np.ndarray
    measurements: dict = field(default_factory=dict)
    tumor_type: dict | None = None    # {"label", "scores", "reasons", "features"}
    fluid: np.ndarray | None = None   # part of `mask` that is fluid (cyst / dead core)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_image(path, channel=None):
    """Load an image as a 2-D float array in [0, 1].

    channel: for multi-channel images, pick one channel (e.g. 1 = FLAIR in the
    Kaggle LGG dataset) instead of converting to grayscale.
    """
    return to_gray01(io.imread(path), channel)


def to_gray01(img, channel=None):
    img = np.asarray(img)
    if img.ndim == 3:
        if img.shape[-1] == 4:                       # drop alpha
            img = img[..., :3]
        if channel is not None:
            img = img[..., channel]
        elif np.allclose(img[..., 0], img[..., 1]) and np.allclose(img[..., 0], img[..., 2]):
            img = img[..., 0]                        # gray image stored as RGB
        else:
            img = color.rgb2gray(img)
    img = util.img_as_float(img).astype(np.float64)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


# ----------------------------------------------------------------------------
# Denoising
# ----------------------------------------------------------------------------
def median_denoise(img, size=3):
    """Each pixel becomes the middle value of its size x size neighbourhood.
    Isolated noisy pixels are outvoted by their neighbours; edges survive."""
    return ndi.median_filter(img, size=size)


def gaussian_denoise(img, sigma=1.0):
    """Weighted average of neighbours: smooths fine grain (film/sensor noise)."""
    return filters.gaussian(img, sigma=sigma)


# ----------------------------------------------------------------------------
# Contrast enhancement
# ----------------------------------------------------------------------------
def clahe(img, clip_limit=0.01):
    """Contrast-Limited Adaptive Histogram Equalization: stretches contrast
    tile by tile, so a faint region becomes visible against its own
    neighbourhood. The clip limit stops noise from being amplified."""
    return exposure.equalize_adapthist(img, clip_limit=clip_limit)


def percentile_stretch(img, lo=2, hi=98):
    p_lo, p_hi = np.percentile(img, (lo, hi))
    return exposure.rescale_intensity(img, in_range=(p_lo, p_hi), out_range=(0, 1))


# ----------------------------------------------------------------------------
# Masks and morphology
# ----------------------------------------------------------------------------
def largest_component(mask):
    lab = measure.label(mask)
    if lab.max() == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def clean_mask(mask, open_radius=2):
    """Opening removes thin specks and bridges, closing seals small gaps,
    hole filling makes each region solid."""
    se = morphology.disk(open_radius)
    m = ndi.binary_opening(mask, structure=se)
    m = ndi.binary_closing(m, structure=se, border_value=0)
    return ndi.binary_fill_holes(m)


def remove_small(mask, min_area):
    lab = measure.label(mask)
    sizes = np.bincount(lab.ravel())
    return (sizes[lab] >= min_area) & mask.astype(bool)


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
def measure_region(mask, pixel_spacing_mm=None):
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


def fmt(x, nd=2):
    return f"{x:.{nd}f}"
