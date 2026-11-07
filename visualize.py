"""Figures showing every stage of a section's pipeline."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_step(ax, step, gt=None):
    ax.imshow(step.image, cmap="gray")
    if step.overlay is not None and step.overlay.any():
        ax.contour(step.overlay, levels=[0.5], colors="red", linewidths=1.5)
    if step.fluid is not None and step.fluid.any():
        ax.contour(step.fluid, levels=[0.5], colors="cyan", linewidths=1.5)
    if gt is not None and gt.any() and step.overlay is not None:
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1, linestyles="--")
    ax.set_title(step.title)
    ax.axis("off")


def summary_line(res):
    m = res.measurements
    if not m.get("detected"):
        return "No tumor region detected"
    s = f"Tumor region: {m['area_px']} px"
    if res.tumor_type:
        s += f" — suggested type: {res.tumor_type['label']} ({res.tumor_type['confidence']})"
    return s


def save_steps_figure(res, path, title="", gt=None):
    n = len(res.steps)
    cols = 4 if n > 6 else 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, step in zip(axes.ravel(), res.steps):
        draw_step(ax, step, gt)
    info = summary_line(res)
    if gt is not None:
        info += "   (red = detected, green dashed = ground truth)"
    fig.suptitle(f"{title}\n{info}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
