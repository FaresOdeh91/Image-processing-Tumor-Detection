"""
Interactive app: pick a body-part section, give it an image, see every step.

    streamlit run app.py

Runs locally; images never leave the computer. No AI is involved anywhere:
each step is a filter, threshold, morphological operation or hand-written rule,
and its explanation is shown under its picture.
"""
import csv
import random
from pathlib import Path

import numpy as np
import streamlit as st
from scipy import ndimage as ndi
from skimage import io, segmentation

import sections
from sections.common import to_gray01

SAMPLES = Path("data/figshare")

st.set_page_config(page_title="Tumor detection (no AI)", layout="wide")


def to_rgb(img, overlay=None, gt=None, fluid=None):
    """Grayscale image -> RGB: red outline for `overlay` (tumor), cyan for `fluid`
    inside the tumor, green for `gt` (expert's outline)."""
    img = img.astype(float)
    if img.max() > 1:
        img = img / img.max()
    rgb = np.dstack([img] * 3)
    for m, colour in ((gt, (0.1, 0.9, 0.1)), (overlay, (1.0, 0.15, 0.15)),
                      (fluid, (0.1, 0.85, 1.0))):
        if m is not None and m.any():
            edge = ndi.binary_dilation(segmentation.find_boundaries(m, mode="inner"),
                                       iterations=max(1, img.shape[0] // 256))
            rgb[edge] = colour
    return np.clip(rgb, 0, 1)


@st.cache_data
def sample_index():
    path = SAMPLES / "labels.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Sidebar: section, image, method
# ---------------------------------------------------------------------------
st.sidebar.title("Tumor detection")
st.sidebar.caption("Classical image processing only. No AI, no training.")

names = list(sections.SECTIONS)
sec_name = st.sidebar.radio("Section", names, format_func=lambda n: sections.get(n).TITLE)
st.sidebar.caption("Breast (mammogram): coming next.")
sec = sections.get(sec_name)

source = st.sidebar.radio("Image", ["Upload", "Sample from dataset"]
                          if sec_name == "brain" and sample_index() else ["Upload"])
img, gt, true_type, caption = None, None, None, ""
if source == "Upload":
    up = st.sidebar.file_uploader("Image file", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if up is not None:
        raw = io.imread(up)
        channel = None
        if raw.ndim == 3 and raw.shape[-1] >= 3 and not (
                np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 0], raw[..., 2])):
            pick = st.sidebar.selectbox("Colour image: which channel?",
                                        ["Convert to gray", "Channel 0", "Channel 1 (LGG FLAIR)",
                                         "Channel 2"])
            channel = None if pick.startswith("Convert") else int(pick.split()[1])
        img = to_gray01(raw, channel)
        caption = up.name
else:
    rows = sample_index()
    want = st.sidebar.selectbox("Tumor type", ["any"] + sec.TYPES)
    test_only = st.sidebar.checkbox("Only test patients (rules never saw them)", value=True)
    pool = [r for r in rows if (want == "any" or r["type"] == want)
            and (not test_only or r["fold"] in ("3", "4", "5"))]
    if "sample_id" not in st.session_state or st.sidebar.button("Another sample"):
        st.session_state.sample_id = random.choice(pool)["id"]
    if st.session_state.sample_id not in {r["id"] for r in pool}:
        st.session_state.sample_id = random.choice(pool)["id"]
    sid = st.session_state.sample_id
    row = next(r for r in pool if r["id"] == sid)
    img = to_gray01(io.imread(SAMPLES / "images" / f"{sid}.png"))
    if st.sidebar.checkbox("Show the expert's outline (green)", value=True):
        gt = io.imread(SAMPLES / "images" / f"{sid}_mask.png") > 0
    true_type = row["type"]
    caption = f"Dataset slice {sid}"

method = st.sidebar.selectbox("Detection method", sec.METHODS,
                              index=sec.METHODS.index(sec.PARAMS["method"]))

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title(f"{sec.TITLE}")
st.write(sec.DESCRIPTION)

with st.expander("How does this work without AI?"):
    st.markdown(
        "An AI system learns its own decision rules from thousands of labelled "
        "examples, and those rules are hidden inside the model. This system has "
        "**no learning step**: every decision is a measurement compared with a "
        "threshold that a person wrote down, and you can read each one below.\n\n"
        "1. **Clean up the image** with fixed filters (median filter, contrast equalization).\n"
        "2. **Keep only the organ** using thresholds and morphology (shrinking / growing shapes).\n"
        "3. **Find the abnormal region**: a tumor differs in brightness from healthy tissue, "
        "so brightness thresholds (Otsu), flood filling or watershed separate it.\n"
        "4. **Check it with rules**: big enough, compact, clearly brighter than normal.\n"
        "5. **Suggest a type** from radiology rules about location, size, brightness and "
        "evenness. Each rule that applies adds points; the reasons are listed.\n\n"
        "It finds *suspicious regions* and *suggests* a type. It is not a diagnosis.")

if img is None:
    st.info("Upload an image in the sidebar, or pick a sample from the dataset.")
    st.stop()

with st.spinner("Processing…"):
    res = sec.run(img, method)

m = res.measurements
left, right = st.columns([1, 1])
with left:
    st.image(to_rgb(img, res.mask, gt, res.fluid), caption=caption, use_container_width=True)
with right:
    if not m.get("detected"):
        st.subheader("No tumor region found")
        st.write("No region passed the rules (size, compactness, brightness). "
                 "See step 6 below for why.")
    else:
        tt = res.tumor_type
        st.subheader(f"Suggested type: {tt['label'].capitalize()}")
        st.caption(f"Rule confidence: {tt['confidence']}")
        if true_type:
            ok = "matches" if true_type == tt["label"] else "does not match"
            st.write(f"Dataset label: **{true_type}** ({ok})")
        st.write("**Points per type**")
        for t in sorted(sec.TYPES, key=lambda t: (t != tt["label"], -tt["scores"][t])):
            st.write(f"- **{t}**: {max(tt['scores'][t], 0)}")
            for r in tt["reasons"][t]:
                st.caption(f"  {r}")
        if m.get("fluid_share"):
            st.write(f"**Fluid inside the tumor:** {m['fluid_share']:.0%} of its area "
                     "(outlined in cyan)")
        st.write("**Measurements**")
        st.json({"area (px)": m["area_px"], "equivalent diameter (px)": m["equiv_diameter_px"],
                 "solidity": m["solidity"],
                 **{k: round(v, 3) for k, v in tt["features"].items()}}, expanded=False)
    if gt is not None:
        from evaluate import dice
        st.write(f"Overlap with expert outline (Dice): **{dice(res.mask, gt):.2f}**  "
                 "(1 = perfect, 0 = none)")
    st.caption("Red = detected tumor. Cyan = fluid inside it. "
               "Green = expert's outline (dataset samples).")

st.header("Step by step")
for i, step in enumerate(res.steps):
    if i % 2 == 0:
        cols = st.columns(2)
    with cols[i % 2]:
        st.subheader(step.title)
        if step.image.dtype == bool:
            st.image(step.image.astype(float), use_container_width=True)
        else:
            st.image(to_rgb(step.image, step.overlay, fluid=step.fluid),
                     use_container_width=True)
        for para in step.explanation.split("\n"):
            st.write(para)
