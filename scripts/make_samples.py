"""
Build the samples/ folder: 3 tumor slices (one per type, from the Figshare
*test* patients, folds 3-5) and 2 tumor-free slices from healthy volunteers
(OpenNeuro ds000001, T1-weighted).

    python scripts/make_samples.py
"""
import csv
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage import io, transform

OUT = Path("samples")
FIG = Path("data/figshare")
NII = Path("data/openneuro_raw")


def tumor_samples(seed=7):
    rows = list(csv.DictReader(open(FIG / "labels.csv")))
    rng = np.random.default_rng(seed)
    for t in ("glioma", "meningioma", "pituitary"):
        pool = [r for r in rows if r["type"] == t and r["fold"] in ("3", "4", "5")]
        r = pool[rng.integers(len(pool))]
        shutil.copy(FIG / "images" / f"{r['id']}.png", OUT / f"tumor_{t}_{r['id']}.png")
        shutil.copy(FIG / "images" / f"{r['id']}_mask.png", OUT / f"tumor_{t}_{r['id']}_mask.png")


def healthy_slice(subject, z, size=512):
    io.imsave(OUT / f"healthy_sub{subject}_slice{z}.png", healthy_slice_array(subject, z, size),
              check_contrast=False)


def healthy_slice_array(subject, z, size=512):
    """One axial slice as an 8-bit square image, voxels resampled to square pixels."""
    img = nib.as_closest_canonical(nib.load(NII / f"sub-{subject}_T1w.nii.gz"))
    zx, zy, _ = img.header.get_zooms()
    sl = np.rot90(img.get_fdata()[:, :, z])                  # rows = y (front at top)
    # voxels are not square (1.0 x 1.33 mm): resample to square pixels
    sl = transform.resize(sl, (round(sl.shape[0] * zy), round(sl.shape[1] * zx)),
                          anti_aliasing=True)
    h, w = sl.shape                                          # pad to a square, then 512x512
    n = max(h, w)
    sq = np.zeros((n, n))
    sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = sl
    sq = transform.resize(sq, (size, size), anti_aliasing=True)
    sq = (sq - sq.min()) / (sq.max() - sq.min())
    return np.uint8(np.round(255 * sq))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tumor_samples()
    healthy_slice("01", 105)
    healthy_slice("02", 100)
    print(sorted(p.name for p in OUT.glob("*.png")))
