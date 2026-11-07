"""
Brain MRI: skull stripping, tumor segmentation and rule-based candidate filtering.

All steps are classical image processing (thresholds, morphology, flood fill,
watershed). Parameters are set by hand in `PARAMS` and never fitted to data.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from ..common import largest_component

PARAMS = {
    # preprocessing
    "denoise_size": 3,
    "clahe_clip": 0.01,
    # skull stripping
    "head_thresh_frac": 0.5,     # head = pixels above this fraction of the Otsu threshold
    "skull_erosion_frac": 0.07,  # erosion radius as a fraction of the head's diameter
    # segmentation
    "method": "thick_blob",      # thick_blob | otsu | region_growing | watershed
    "otsu_classes": 4,
    # thick_blob selection (see filter_candidates)
    "brain_erosion_frac": 0.02,  # how far the brain outline is shrunk to cut it off the face/neck
    "brain_margin_frac": 0.06,   # how far outside that outline a tumor may still reach
    "max_edge_touch": 0.05,      # share of a blob allowed to touch the search area's edge
    "grow_frac": 0.05,           # hysteresis growth reach, as a fraction of the head radius
    "rg_tolerance": 0.15,
    "seed_sigma": 3,
    "ws_margin": 0.05,
    # post-processing rules
    "open_radius": 2,
    "min_area": 80,
    "max_area_frac": 0.5,
    "min_solidity": 0.6,
    "min_contrast": 1.5,
    "keep": "best",
    "pixel_spacing_mm": None,
}

METHODS = ["thick_blob", "otsu", "region_growing", "watershed"]


def head_mask(img, head_thresh_frac=0.5):
    """Everything that is not air: threshold well below Otsu, keep the largest
    component and fill its holes."""
    head = img > head_thresh_frac * filters.threshold_otsu(img)
    return ndi.binary_fill_holes(largest_component(head))


def skull_strip(img, head_thresh_frac=0.5, skull_erosion_frac=0.07):
    """Return (brain_mask, head_mask, erosion_radius).

    1. Head mask (see `head_mask`).
    2. Erode by a radius proportional to the head size to peel off scalp + skull.
    3. Keep the largest remaining component (the brain) and smooth its border.
    """
    head = head_mask(img, head_thresh_frac)
    if not head.any():
        return head, head, 0
    diameter = 2 * np.sqrt(head.sum() / np.pi)
    r = max(2, int(skull_erosion_frac * diameter))
    brain = ndi.binary_erosion(head, structure=morphology.disk(r))
    brain = largest_component(brain)
    brain = ndi.binary_opening(brain, structure=morphology.disk(max(1, r // 3)))
    brain = ndi.binary_fill_holes(brain)
    return brain, head, r


def brain_outline(img, head, erosion_frac=0.02, margin_frac=0.06):
    """Where the brain itself is, as opposed to the face, neck and scalp.

    Bone and fluid are dark on MRI, so the darkest brightness class is removed;
    what remains is a few thin bridges away from being separate pieces (brain,
    scalp, face, neck). Shrinking by a small radius snaps those bridges, the
    largest piece is the brain, and it is grown back by the same radius. The
    result is widened by a margin because tumors growing from the brain's
    lining (meningiomas) sit just outside it.
    """
    if not head.any():
        return head
    d = 2 * np.sqrt(head.sum() / np.pi)
    dark = filters.threshold_multiotsu(img[head], classes=4)[0]
    tissue = ndi.binary_erosion(head, structure=morphology.disk(2)) & (img > dark)
    r = max(2, int(erosion_frac * d))
    core = largest_component(ndi.binary_erosion(tissue, structure=morphology.disk(r)))
    brain = ndi.binary_dilation(core, structure=morphology.disk(r)) & head
    brain = ndi.binary_fill_holes(ndi.binary_closing(brain, structure=morphology.disk(r)))
    return ndi.binary_dilation(brain, structure=morphology.disk(max(1, int(margin_frac * d)))) & head


def segment(img, roi, method="otsu", otsu_classes=4, rg_tolerance=0.15, seed_sigma=3,
            ws_margin=0.05, **_):
    """Return (raw_mask, explanation)."""
    if not roi.any():
        return np.zeros_like(roi), "No brain area was found, so nothing was segmented."
    vals = img[roi]

    if method in ("otsu", "thick_blob"):
        thrs = filters.threshold_multiotsu(vals, classes=otsu_classes)
        cut = ", ".join(f"{t:.2f}" for t in thrs)
        return roi & (img > thrs[-1]), (
            f"Multi-level Otsu split the brain's brightness histogram into {otsu_classes} "
            f"groups (cut-offs {cut}) so that the groups differ from each other as much as "
            f"possible (roughly fluid / gray matter / white matter / lesion). The brightest "
            f"group, above {thrs[-1]:.2f}, is the tumor candidate.")

    if method == "region_growing":
        smooth = filters.gaussian(img, sigma=seed_sigma)
        smooth[~roi] = 0
        seed = np.unravel_index(np.argmax(smooth), img.shape)
        grown = segmentation.flood(img, seed, tolerance=rg_tolerance) & roi
        return grown, (
            f"The seed is the brightest point after heavy smoothing, at row {seed[0]}, "
            f"column {seed[1]}. The region then grows to every connected neighbour whose "
            f"brightness is within ±{rg_tolerance} of the seed, like a paint-bucket fill.")

    if method == "watershed":
        thrs = filters.threshold_multiotsu(vals, classes=otsu_classes)
        markers = np.zeros(img.shape, dtype=np.int32)
        markers[roi & (img < thrs[0])] = 1
        markers[roi & (img > thrs[-1] + ws_margin)] = 2
        markers[~roi] = 1
        if not (markers == 2).any():
            return np.zeros_like(roi), "No pixel was bright enough to be a 'sure tumor' marker."
        labels = segmentation.watershed(filters.sobel(img), markers)
        return (labels == 2) & roi, (
            "Pixels that are surely tumor (very bright) and surely background (dark) are "
            "marked. The edge-strength image (Sobel filter) is treated as a landscape and "
            "flooded from both sets of markers; where the floods meet, on the strongest "
            "edges, becomes the tumor border.")

    raise ValueError(f"unknown segmentation method: {method}")


def filter_candidates(mask, img, roi, open_radius=2, min_area=80, max_area_frac=0.5,
                      min_solidity=0.6, min_contrast=1.5, keep="best", selection="brightness_size",
                      head=None, near_brain=None, max_edge_touch=0.05, **_):
    """Rule-based candidate filtering. Returns (mask, explanation).

    A component is kept only if it is big enough but not most of the brain,
    compact (solidity), and clearly brighter than the brain on average.

    selection="brightness_size": among those, keep the brightest x largest.
    selection="thick_central":   also reject blobs touching the search area's
        edge (leftover skull / scalp), then keep the best
        brightness x thickness x closeness-to-centre x share-inside-the-brain.
        Thickness (the widest circle that fits inside the blob) is what tells a
        tumor from the blood vessels and folds that also light up with contrast.
    """
    se = morphology.disk(open_radius)
    m = ndi.binary_opening(mask, structure=se)
    m = ndi.binary_closing(m, structure=se, border_value=0)
    m = ndi.binary_fill_holes(m)
    labels = measure.label(m)
    sizes = np.bincount(labels.ravel())
    m = (sizes[labels] >= min_area) & m

    roi_area = roi.sum()
    roi_mean, roi_std = img[roi].mean(), img[roi].std()
    bright_needed = roi_mean + min_contrast * roi_std
    labels = measure.label(m)
    regions = measure.regionprops(labels, intensity_image=img)
    thick_rules = selection == "thick_central" and head is not None
    if thick_rules:
        depth = ndi.distance_transform_edt(roi)
        head_r = np.sqrt(head.sum() / np.pi)
        head_c = np.array(ndi.center_of_mass(head))
    candidates, rejected = [], {"too large": 0, "irregular": 0, "not bright enough": 0,
                                "touching the edge": 0}
    for r in regions:
        reg = labels == r.label
        if r.area > max_area_frac * roi_area:
            rejected["too large"] += 1
        elif r.solidity < min_solidity:
            rejected["irregular"] += 1
        elif r.intensity_mean < bright_needed:
            rejected["not bright enough"] += 1
        elif thick_rules and (depth[reg] < 4).mean() > max_edge_touch:
            rejected["touching the edge"] += 1
        else:
            if thick_rules:
                r.thick = ndi.distance_transform_edt(reg).max() / head_r
                r.centrality = 1.2 - min(np.linalg.norm(np.array(r.centroid) - head_c) / head_r, 1)
                r.inside = near_brain[reg].mean() if near_brain is not None else 1.0
                r.score = r.intensity_mean * r.thick * r.centrality * r.inside
            else:
                r.score = r.intensity_mean * np.sqrt(r.area)
            candidates.append(r)

    rules = (f"Morphological opening/closing removed specks and filled gaps. Each "
             f"remaining blob had to be at least {min_area} px, under "
             f"{int(max_area_frac * 100)}% of the brain, compact (solidity ≥ {min_solidity}) "
             f"and brighter than {bright_needed:.2f} (brain mean {roi_mean:.2f} + "
             f"{min_contrast} × spread {roi_std:.2f})")
    rules += (f", and touch the edge of the search area along at most "
              f"{max_edge_touch:.0%} of itself (edge contact means leftover skull or scalp)."
              if thick_rules else ".")
    rej = ", ".join(f"{v} {k}" for k, v in rejected.items() if v)
    out = np.zeros_like(mask, dtype=bool)
    if not candidates:
        return out, (f"{rules} {len(regions)} blob(s) checked, none passed"
                     f"{' (' + rej + ')' if rej else ''}: no tumor reported.")
    if keep == "best":
        best = max(candidates, key=lambda r: r.score)
        if len(candidates) == 1:
            note = " One blob passed."
        elif thick_rules:
            note = (f" {len(candidates)} passed; kept the one with the best score of "
                    f"brightness × thickness × closeness to the centre × share inside the brain "
                    f"(thickness {best.thick:.3f}, inside the brain {best.inside:.0%}). Thin "
                    f"bright lines are blood vessels, not tumor.")
        else:
            note = (f" {len(candidates)} passed; the one with the best brightness × size "
                    f"score was kept.")
        candidates = [best]
    else:
        note = f" {len(candidates)} passed and all were kept."
    for r in candidates:
        out[labels == r.label] = True
    return out, rules + note + (f" Rejected: {rej}." if rej else "")


def grow_hysteresis(region, img, roi, low, head, grow_frac=0.05, open_radius=2):
    """Hysteresis: add moderately bright pixels that connect to the chosen core,
    within a limited reach. Returns (mask, explanation)."""
    if not region.any() or grow_frac <= 0:
        return region, "No growth step."
    head_r = np.sqrt(head.sum() / np.pi)
    reach = max(1, int(grow_frac * head_r))
    zone = ndi.binary_dilation(region, structure=morphology.disk(reach))
    lab = measure.label(roi & (img > low) & zone)
    keep = np.unique(lab[region])
    keep = keep[keep > 0]
    grown = ndi.binary_opening(np.isin(lab, keep), structure=morphology.disk(open_radius))
    grown = ndi.binary_fill_holes(grown | region)
    return grown, (f"Hysteresis (as in Canny edge detection): the chosen bright core "
                   f"({int(region.sum())} px) is extended to connected pixels brighter than the "
                   f"lower cut-off {low:.2f}, at most {reach} px away. Tumors are often only "
                   f"partly very bright, so this recovers the rest of them "
                   f"({int(grown.sum())} px in total).")


def find_fluid(tumor, img, roi, fluid_level, contact_min=0.5, max_ratio=2.0, snap_radius=2,
               min_area=30):
    """Fluid inside or attached to the tumor (cyst, dead/necrotic core).

    Fluid is dark on contrast MRI, as dark as the fluid around the brain, so a
    brightness-based search treats it as background. Here a dark region is
    counted as part of the tumor when the tumor's bright wall surrounds it:

      * dark = darker than `fluid_level` (the darkest Otsu cut-off: fluid level),
      * thin links are snapped first (opening with a small disk), so a cyst that
        touches a ventricle through a narrow channel is not merged with it,
      * at least `contact_min` of the dark region's border touches the tumor,
      * it does not touch the edge of the brain area (fluid around the brain),
      * it is not more than `max_ratio` times the size of the tumor's bright part.

    Healthy fluid spaces (ventricles, grooves between folds) are bordered mostly
    by normal brain, so they fail the contact rule. Returns (fluid_mask, explanation).
    """
    empty = np.zeros_like(tumor, dtype=bool)
    if not tumor.any():
        return empty, "No tumor, so no fluid search."
    dark = roi & (img < fluid_level)
    dark = ndi.binary_opening(dark, structure=morphology.disk(snap_radius))
    lab = measure.label(dark)
    near_tumor = ndi.binary_dilation(tumor, structure=morphology.disk(2))
    edge = roi & ~ndi.binary_erosion(roi, structure=morphology.disk(3))
    t_area = tumor.sum()
    fluid = empty.copy()
    checked = kept = 0
    for rp in measure.regionprops(lab):
        if rp.area < min_area:
            continue
        sl = rp.slice
        comp = lab[sl] == rp.label
        if not near_tumor[sl][comp].any():
            continue                                   # not next to the tumor at all
        checked += 1
        pad = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in sl)
        comp_p = lab[pad] == rp.label
        ring = ndi.binary_dilation(comp_p, structure=morphology.disk(1)) & ~comp_p
        contact = (ring & near_tumor[pad]).sum() / max(ring.sum(), 1)
        if contact < contact_min or (comp & edge[sl]).any() or rp.area > max_ratio * t_area:
            continue
        fluid[sl] |= comp
        kept += 1
    fluid &= ~tumor
    why = (f"Fluid is as dark as the fluid around the brain (below {fluid_level:.2f}), so a "
           f"search for bright tissue skips it. {checked} dark region(s) touch the tumor; a "
           f"region counts as tumor fluid only if the tumor's bright wall covers at least "
           f"{contact_min:.0%} of its border, it does not reach the edge of the brain and it is "
           f"at most {max_ratio:g}× the bright part's size. Healthy fluid spaces (ventricles, "
           f"grooves between folds) are bordered mostly by normal brain and fail this rule. ")
    if kept:
        why += (f"{kept} region(s) passed: {int(fluid.sum())} px of fluid, "
                f"{fluid.sum() / (fluid.sum() + t_area):.0%} of the tumor.")
    else:
        why += "None passed: the tumor looks solid."
    return fluid, why
