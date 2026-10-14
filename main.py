"""
Tumor region detection with pure image processing (no AI / no training).

Usage
-----
  # Try it on synthetic images (no dataset needed)
  python main.py demo

  # One image (optionally with a ground-truth mask)
  python main.py detect path/to/slice.png --method otsu
  python main.py detect slice.tif --channel 1 --gt slice_mask.tif

  # A whole dataset, comparing all three segmentation methods
  python main.py evaluate path/to/kaggle_3m --channel 1 --method all
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from skimage import io

import evaluate as ev
import modalities
import pipeline
from synthetic import make_phantom
from visualize import save_steps_figure

METHODS = ["otsu", "region_growing", "watershed"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_mask(path):
    m = io.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0


def find_pairs(data_dir, masks_dir=None):
    """Pair images with masks.

    - Kaggle LGG convention: X.tif + X_mask.tif in the same folder (searched recursively)
    - Or: --masks DIR containing masks with the same file names as the images
    """
    data_dir = Path(data_dir)
    pairs = []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or p.stem.endswith("_mask"):
            continue
        if masks_dir:
            m = Path(masks_dir) / p.name
        else:
            m = p.with_name(p.stem + "_mask" + p.suffix)
        pairs.append((p, m if m.exists() else None))
    return pairs


def overrides_from(args):
    o = {}
    if args.min_area is not None:
        o["min_area"] = args.min_area
    if args.min_contrast is not None:
        o["min_contrast"] = args.min_contrast
    if args.pixel_spacing is not None:
        o["pixel_spacing_mm"] = args.pixel_spacing
    return o


def cmd_detect(args):
    mod = modalities.get(args.modality)
    img = pipeline.load_image(args.image, args.channel)
    gt = load_mask(args.gt) if args.gt else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for method in (METHODS if args.method == "all" else [args.method]):
        res = pipeline.run(img, mod, method, overrides_from(args))
        line = {"method": method, **res.measurements}
        if gt is not None:
            line["dice"] = round(ev.dice(res.mask, gt), 3)
            line["iou"] = round(ev.iou(res.mask, gt), 3)
        print(json.dumps(line))
        fig = out / f"{Path(args.image).stem}_{method}.png"
        save_steps_figure(res, fig, f"{Path(args.image).name} — {method}", gt)
        print(f"  figure -> {fig}")


def evaluate_set(items, mod, methods, overrides, out, n_figures=5):
    """items: list of (name, img, gt_mask)."""
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for method in methods:
        method_rows = []
        for i, (name, img, gt) in enumerate(items):
            res = pipeline.run(img, mod, method, overrides)
            r = {
                "image": name, "method": method, "has_gt": bool(gt.any()),
                "detected": res.measurements["detected"],
                "dice": round(ev.dice(res.mask, gt), 4),
                "iou": round(ev.iou(res.mask, gt), 4),
                "outcome": ev.detection_outcome(res.mask, gt),
            }
            method_rows.append(r)
            if i < n_figures:
                save_steps_figure(res, out / f"{Path(name).stem}_{method}.png",
                                  f"{name} — {method}", gt)
        rows += method_rows
        summary[method] = ev.summarize(method_rows)

    with open(out / "per_image_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'method':<16}{'sens':>7}{'spec':>7}{'Dice':>8}{'IoU':>8}   TP/FP/FN/TN")
    for m, s in summary.items():
        print(f"{m:<16}{s['sensitivity']:>7}{s['specificity']:>7}"
              f"{str(s['mean_dice_tumor_images']):>8}{str(s['mean_iou_tumor_images']):>8}"
              f"   {s['TP']}/{s['FP']}/{s['FN']}/{s['TN']}")
    print(f"\nResults written to {out}/")


def cmd_evaluate(args):
    mod = modalities.get(args.modality)
    pairs = [(p, m) for p, m in find_pairs(args.data_dir, args.masks) if m is not None]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No image/mask pairs found.")
    print(f"Evaluating {len(pairs)} images...")
    items = [(str(p.relative_to(args.data_dir)), pipeline.load_image(p, args.channel),
              load_mask(m)) for p, m in pairs]
    methods = METHODS if args.method == "all" else [args.method]
    evaluate_set(items, mod, methods, overrides_from(args), Path(args.out))


def cmd_demo(args):
    mod = modalities.get("brain_mri")
    items = []
    for i in range(args.n):
        img, gt = make_phantom(tumor=(i % 4 != 3), seed=i)   # every 4th has no tumor
        items.append((f"phantom_{i:02d}.png", img, gt))
    evaluate_set(items, mod, METHODS, overrides_from(args), Path(args.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--modality", default="brain_mri", choices=list(modalities.MODALITIES))
        p.add_argument("--method", default="otsu", choices=METHODS + ["all"])
        p.add_argument("--channel", type=int, default=None,
                       help="channel of a multi-channel image to use (LGG FLAIR = 1)")
        p.add_argument("--min-area", type=int, default=None)
        p.add_argument("--min-contrast", type=float, default=None)
        p.add_argument("--pixel-spacing", type=float, default=None, help="mm per pixel")
        p.add_argument("--out", default="results")

    d = sub.add_parser("detect", help="run on a single image")
    d.add_argument("image")
    d.add_argument("--gt", help="ground-truth mask for this image")
    common(d)

    e = sub.add_parser("evaluate", help="run on a dataset with ground-truth masks")
    e.add_argument("data_dir")
    e.add_argument("--masks", help="folder of masks with the same file names")
    e.add_argument("--limit", type=int, default=None, help="only the first N images")
    common(e)

    m = sub.add_parser("demo", help="run on synthetic phantoms")
    m.add_argument("--n", type=int, default=12)
    common(m)

    args = ap.parse_args()
    {"detect": cmd_detect, "evaluate": cmd_evaluate, "demo": cmd_demo}[args.cmd](args)


if __name__ == "__main__":
    main()
