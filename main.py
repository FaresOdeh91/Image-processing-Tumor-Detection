"""
Tumor detection and rule-based typing with pure image processing (no AI / no training).

Every body part is its own *section* with its own pipeline (see sections/).

Usage
-----
  # Interactive app: pick a section, drop in an image
  streamlit run app.py

  # Synthetic sanity check (no dataset needed)
  python main.py demo

  # One image (optionally with a ground-truth mask)
  python main.py detect scan.png --section brain
  python main.py detect slice.tif --section brain --channel 1 --gt slice_mask.tif

  # A whole dataset. If the folder has a labels.csv (id,type[,fold]) the
  # suggested tumor types are scored too; --folds picks the test patients.
  python main.py evaluate data/figshare --section brain --folds 3,4,5
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from skimage import io

import evaluate as ev
import sections
from sections.common import load_image
from synthetic import make_phantom
from visualize import save_steps_figure

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_mask(path):
    m = io.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0


def find_pairs(data_dir, masks_dir=None):
    """Pair images with masks: X.ext + X_mask.ext in the same folder (searched
    recursively), or --masks DIR holding masks with the same file names."""
    data_dir = Path(data_dir)
    pairs = []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or p.stem.endswith("_mask"):
            continue
        m = Path(masks_dir) / p.name if masks_dir else p.with_name(p.stem + "_mask" + p.suffix)
        pairs.append((p, m if m.exists() else None))
    return pairs


def read_labels(data_dir):
    """labels.csv next to the data: {image id: row}. Optional."""
    path = Path(data_dir) / "labels.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def overrides_from(args):
    o = {}
    if args.min_area is not None:
        o["min_area"] = args.min_area
    if args.min_contrast is not None:
        o["min_contrast"] = args.min_contrast
    if args.pixel_spacing is not None:
        o["pixel_spacing_mm"] = args.pixel_spacing
    return o


def methods_for(sec, choice):
    return sec.METHODS if choice == "all" else [choice or sec.PARAMS["method"]]


def cmd_detect(args):
    sec = sections.get(args.section)
    img = load_image(args.image, args.channel)
    gt = load_mask(args.gt) if args.gt else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for method in methods_for(sec, args.method):
        res = sec.run(img, method, overrides_from(args))
        print(f"\n=== {sec.TITLE} — {method} ===")
        for step in res.steps:
            print(f"\n[{step.title}]\n{step.explanation}")
        line = {"method": method, **res.measurements}
        if res.tumor_type:
            line["suggested_type"] = res.tumor_type["label"]
        if gt is not None:
            line["dice"] = round(ev.dice(res.mask, gt), 3)
            line["iou"] = round(ev.iou(res.mask, gt), 3)
        print("\n" + json.dumps(line))
        fig = out / f"{Path(args.image).stem}_{method}.png"
        save_steps_figure(res, fig, f"{Path(args.image).name} — {method}", gt)
        print(f"figure -> {fig}")


def evaluate_set(items, sec, methods, overrides, out, n_figures=5):
    """items: list of (name, load, true_type or None), where load() -> (img, gt_mask).

    Images are loaded one at a time, so a large dataset does not fill memory.
    """
    out.mkdir(parents=True, exist_ok=True)
    has_types = any(t for *_, t in items) and hasattr(sec, "classify_region")
    rows = {m: [] for m in methods}
    type_pairs = {m: [] for m in methods}
    outline_pairs = []
    for i, (name, load, true_type) in enumerate(items):
        img, gt = load()
        if has_types and true_type and gt.any():
            # Type rules alone, on the expert's outline: is the typing logic right?
            outline_pairs.append((true_type, sec.classify_region(img, gt, overrides)["label"]))
        for method in methods:
            res = sec.run(img, method, overrides)
            r = {
                "image": name, "method": method, "has_gt": bool(gt.any()),
                "detected": res.measurements["detected"],
                "dice": round(ev.dice(res.mask, gt), 4),
                "iou": round(ev.iou(res.mask, gt), 4),
                "outcome": ev.detection_outcome(res.mask, gt),
                "true_type": true_type or "",
                "suggested_type": res.tumor_type["label"] if res.tumor_type else "",
            }
            rows[method].append(r)
            if true_type and r["outcome"] == "TP":
                type_pairs[method].append((true_type, r["suggested_type"]))
            if i < n_figures:
                save_steps_figure(res, out / f"{Path(name).stem}_{method}.png",
                                  f"{name} — {method}", gt)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)

    summary = {}
    if has_types:
        summary["type_rules_on_expert_outline"] = ev.type_summary(outline_pairs, sec.TYPES)
    for method in methods:
        summary[method] = ev.summarize(rows[method])
        if has_types:
            summary[method]["type_on_detected_region"] = ev.type_summary(type_pairs[method],
                                                                         sec.TYPES)
    rows = [r for m in methods for r in rows[m]]

    with open(out / "per_image_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'method':<16}{'sens':>7}{'spec':>7}{'Dice':>8}{'IoU':>8}   TP/WRONG/FN/FP/TN"
          + ("   type acc." if has_types else ""))
    for m in methods:
        s = summary[m]
        line = (f"{m:<16}{s['sensitivity']:>7}{s['specificity']:>7}"
                f"{str(s['mean_dice_tumor_images']):>8}{str(s['mean_iou_tumor_images']):>8}"
                f"   {s['TP']}/{s['WRONG']}/{s['FN']}/{s['FP']}/{s['TN']}")
        if has_types:
            line += f"   {s['type_on_detected_region']['accuracy']}"
        print(line)
    if has_types:
        t = summary["type_rules_on_expert_outline"]
        print(f"\nType rules on the expert outline: accuracy {t['accuracy']} "
              f"over {t['n']} images")
        ev.print_confusion(t, sec.TYPES)
    print(f"\nResults written to {out}/")


def cmd_evaluate(args):
    sec = sections.get(args.section)
    labels = read_labels(args.data_dir)
    folds = set(args.folds.split(",")) if args.folds else None
    pairs = []
    for p, m in find_pairs(args.data_dir, args.masks):
        if m is None:
            continue
        lab = labels.get(p.stem)
        if folds and (lab is None or lab.get("fold") not in folds):
            continue
        pairs.append((p, m, lab["type"] if lab else None))
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No image/mask pairs found.")
    print(f"Evaluating {len(pairs)} images...")
    items = [(str(p.relative_to(args.data_dir)),
              lambda p=p, m=m: (load_image(p, args.channel), load_mask(m)), t)
             for p, m, t in pairs]
    evaluate_set(items, sec, methods_for(sec, args.method), overrides_from(args), Path(args.out))


def cmd_demo(args):
    sec = sections.get("brain")
    items = []
    for i in range(args.n):
        # every 4th phantom has no tumor
        items.append((f"phantom_{i:02d}.png",
                      lambda i=i: make_phantom(tumor=(i % 4 != 3), seed=i), None))
    evaluate_set(items, sec, methods_for(sec, "all"), overrides_from(args), Path(args.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--section", default="brain", choices=list(sections.SECTIONS))
        p.add_argument("--method", default=None,
                       help="segmentation method of the section, or 'all'")
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
    e.add_argument("--folds", help="only these folds from labels.csv, e.g. 3,4,5")
    e.add_argument("--limit", type=int, default=None, help="only the first N images")
    common(e)

    m = sub.add_parser("demo", help="run on synthetic phantoms")
    m.add_argument("--n", type=int, default=12)
    common(m)

    args = ap.parse_args()
    {"detect": cmd_detect, "evaluate": cmd_evaluate, "demo": cmd_demo}[args.cmd](args)


if __name__ == "__main__":
    main()
