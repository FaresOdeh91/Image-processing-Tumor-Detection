"""Figures showing every pipeline stage."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import segmentation


def save_steps_figure(res, path, title="", gt=None):
    panels = [
        ("1. Original", res.original, None),
        ("2. Denoised", res.denoised, None),
        ("3. Enhanced (CLAHE)", res.enhanced, None),
        ("4. Region of interest", res.roi, None),
        ("5. Raw segmentation", res.raw_mask, None),
        ("6. Result", res.original, res.mask),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (name, im, mask) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap="gray")
        if mask is not None:
            if mask.any():
                ax.contour(mask, levels=[0.5], colors="red", linewidths=1.5)
            if gt is not None and gt.any():
                ax.contour(gt, levels=[0.5], colors="lime", linewidths=1, linestyles="--")
        ax.set_title(name)
        ax.axis("off")

    m = res.measurements
    info = (f"Tumor region: area {m['area_px']} px, centroid {m['centroid_rc']}"
            if m.get("detected") else "No tumor region detected")
    if gt is not None:
        info += "   (red = detected, green dashed = ground truth)"
    fig.suptitle(f"{title}\n{info}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_overlay(res, path):
    """Single image with the detected boundary drawn on the original."""
    out = segmentation.mark_boundaries(res.original, res.mask, color=(1, 0, 0), mode="thick")
    plt.imsave(path, out)
