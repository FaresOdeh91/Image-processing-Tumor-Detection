"""Synthetic brain-MRI-like phantom with a known tumor mask.

Useful for testing the code before you download a real dataset.
It is NOT a substitute for real data in your evaluation.
"""
import numpy as np
from skimage import draw, filters


def make_phantom(size=256, tumor=True, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size))
    c = size / 2

    # scalp + skull (bright ring)
    rr, cc = draw.ellipse(c, c, 0.44 * size, 0.36 * size, shape=img.shape)
    img[rr, cc] = 0.85
    # CSF gap (dark)
    rr, cc = draw.ellipse(c, c, 0.395 * size, 0.315 * size, shape=img.shape)
    img[rr, cc] = 0.15
    # brain (gray matter)
    rr, cc = draw.ellipse(c, c, 0.38 * size, 0.30 * size, shape=img.shape)
    img[rr, cc] = 0.40
    # white matter
    rr, cc = draw.ellipse(c, c, 0.26 * size, 0.19 * size, shape=img.shape)
    img[rr, cc] = 0.55
    # ventricles
    for dx in (-0.05, 0.05):
        rr, cc = draw.ellipse(c - 0.02 * size, c + dx * size, 0.08 * size, 0.025 * size,
                              shape=img.shape)
        img[rr, cc] = 0.15

    mask = np.zeros_like(img, dtype=bool)
    if tumor:
        tr = c + rng.uniform(-0.15, 0.15) * size
        tc = c + rng.uniform(-0.12, 0.12) * size
        rr, cc = draw.ellipse(tr, tc, rng.uniform(0.05, 0.09) * size,
                              rng.uniform(0.05, 0.08) * size,
                              rotation=rng.uniform(0, np.pi), shape=img.shape)
        img[rr, cc] = 0.92
        mask[rr, cc] = True

    img = filters.gaussian(img, sigma=1.5)
    img += rng.normal(0, 0.04, img.shape)          # scanner noise
    return np.clip(img, 0, 1), mask
