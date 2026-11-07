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
    """Image-level outcome.

    TP    tumor present and the reported region overlaps it
    WRONG tumor present but the reported region is somewhere else (a miss AND a false alarm)
    FN    tumor present, nothing reported
    FP    no tumor, something reported
    TN    no tumor, nothing reported
    """
    p, g = pred.any(), gt.any()
    if p and g:
        return "TP" if np.logical_and(pred, gt).any() else "WRONG"
    if p and not g:
        return "FP"
    if g and not p:
        return "FN"
    return "TN"


def type_summary(pairs, types):
    """pairs: list of (true_type, suggested_type). Accuracy, per-type recall, confusion."""
    n = len(pairs)
    conf = {t: {p: 0 for p in types} for t in types}
    for t, p in pairs:
        if t in conf and p in conf[t]:
            conf[t][p] += 1
    recall = {t: round(conf[t][t] / sum(conf[t].values()), 3) if sum(conf[t].values()) else None
              for t in types}
    acc = round(sum(conf[t][t] for t in types) / n, 3) if n else None
    return {"n": n, "accuracy": acc, "recall": recall, "confusion": conf}


def print_confusion(s, types):
    w = max(len(t) for t in types) + 2
    print(" " * w + "suggested →".ljust(4) + "".join(f"{t:>{w}}" for t in types) + "   recall")
    for t in types:
        print(f"{t:<{w}}" + " " * 11 + "".join(f"{s['confusion'][t][p]:>{w}}" for p in types)
              + f"   {s['recall'][t]}")


def summarize(rows):
    """rows: list of dicts with keys dice, iou, outcome."""
    n = len(rows)
    if n == 0:
        return {}
    c = {k: sum(r["outcome"] == k for r in rows) for k in ("TP", "WRONG", "FN", "FP", "TN")}
    with_tumor = [r for r in rows if r["has_gt"]]
    # sensitivity: share of tumor images where the tumor itself was found
    # (a region in the wrong place counts as a miss)
    sens = c["TP"] / len(with_tumor) if with_tumor else float("nan")
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
