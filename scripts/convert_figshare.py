"""
Convert the Figshare brain tumor dataset (Cheng et al., 3064 T1-contrast slices)
from MATLAB .mat files to PNG images + masks + a labels.csv.

    python scripts/convert_figshare.py data/figshare_mat data/figshare_raw/cvind.mat data/figshare

Output:
    <out>/images/0001.png, 0001_mask.png, ...   (8-bit, min-max scaled like the dataset README)
    <out>/labels.csv   id, label, type, patient, fold
"""
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from skimage import io

TYPES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def main(mat_dir, cvind_path, out_dir):
    mat_dir, out_dir = Path(mat_dir), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    with h5py.File(cvind_path, "r") as f:
        folds = np.array(f["cvind"]).ravel().astype(int)

    rows = []
    files = sorted(mat_dir.glob("*.mat"), key=lambda p: int(p.stem))
    for p in files:
        i = int(p.stem)
        with h5py.File(p, "r") as f:
            cj = f["cjdata"]
            img = np.array(cj["image"]).T.astype(np.float64)      # MATLAB is column-major
            mask = np.array(cj["tumorMask"]).T > 0
            label = int(np.array(cj["label"]).ravel()[0])
            pid = "".join(chr(c) for c in np.array(cj["PID"]).ravel())
        lo, hi = img.min(), img.max()
        img8 = np.uint8(np.round(255 * (img - lo) / (hi - lo))) if hi > lo else np.zeros(img.shape, np.uint8)
        io.imsave(out_dir / "images" / f"{i:04d}.png", img8, check_contrast=False)
        io.imsave(out_dir / "images" / f"{i:04d}_mask.png", np.uint8(mask) * 255,
                  check_contrast=False)
        rows.append({"id": f"{i:04d}", "label": label, "type": TYPES[label],
                     "patient": pid, "fold": int(folds[i - 1])})

    with open(out_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} slices to {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
