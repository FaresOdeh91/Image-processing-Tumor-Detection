"""Evaluation metrics against ground-truth masks."""
import numpy as np


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0                      # correctly found nothing
    return 2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum())


def iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return np.logical_and(pred, gt).sum() / union


def detection_outcome(pred, gt):
    """Image-level outcome: TP / FP / FN / TN (did we find a tumor when one exists?)."""
    p, g = pred.any(), gt.any()
    if p and g:
        return "TP" if np.logical_and(pred, gt).any() else "FP"   # found the wrong spot
    if p and not g:
        return "FP"
    if g and not p:
        return "FN"
    return "TN"


def summarize(rows):
    """rows: list of dicts with keys dice, iou, outcome."""
    n = len(rows)
    if n == 0:
        return {}
    c = {k: sum(r["outcome"] == k for r in rows) for k in ("TP", "FP", "FN", "TN")}
    with_tumor = [r for r in rows if r["has_gt"]]
    sens = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else float("nan")
    spec = c["TN"] / (c["TN"] + c["FP"]) if (c["TN"] + c["FP"]) else float("nan")
    return {
        "images": n,
        **c,
        "sensitivity": round(sens, 3),
        "specificity": round(spec, 3),
        "mean_dice_tumor_images": round(float(np.mean([r["dice"] for r in with_tumor])), 3)
        if with_tumor else None,
        "mean_iou_tumor_images": round(float(np.mean([r["iou"] for r in with_tumor])), 3)
        if with_tumor else None,
    }
