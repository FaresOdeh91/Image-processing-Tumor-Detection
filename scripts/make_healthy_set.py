"""
Tumor-free test slices for measuring false alarms (specificity): axial slices
from two healthy volunteers (OpenNeuro ds000001, T1-weighted), each with an
empty mask so `main.py evaluate` can score them.

    python scripts/make_healthy_set.py
"""
from pathlib import Path

import numpy as np
from skimage import io

from make_samples import NII, healthy_slice_array

OUT = Path("data/healthy")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for subject in ("01", "02"):
        for z in range(80, 141, 6):
            img = healthy_slice_array(subject, z)
            name = f"healthy_sub{subject}_z{z:03d}"
            io.imsave(OUT / f"{name}.png", img, check_contrast=False)
            io.imsave(OUT / f"{name}_mask.png", np.zeros_like(img), check_contrast=False)
            n += 1
    print(f"wrote {n} healthy slices to {OUT}")
