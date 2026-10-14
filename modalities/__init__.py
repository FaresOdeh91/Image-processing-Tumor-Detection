"""Modality registry.

To add a new image type (e.g. mammograms), create modalities/<name>.py with:
    PARAMS  : dict of parameters (same keys as brain_mri.PARAMS)
    cleanup : function(img, params) -> (cleaned_img, roi_mask)
and register it below. Nothing else in the pipeline has to change.
"""
from . import brain_mri

MODALITIES = {
    "brain_mri": brain_mri,
    # "mammogram": mammogram,   # future extension
}


def get(name):
    try:
        return MODALITIES[name]
    except KeyError:
        raise ValueError(f"unknown modality '{name}', choose from {list(MODALITIES)}")
