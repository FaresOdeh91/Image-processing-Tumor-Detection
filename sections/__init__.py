"""
Body-part sections.

Each section is a sub-package (sections/<name>/) with its own complete pipeline:
its own denoising, enhancement, organ isolation, detection and type rules,
assembled from the shared blocks in `common.py`. A section exposes:

    NAME, TITLE, DESCRIPTION : strings shown in the app
    METHODS                  : segmentation methods it supports
    TYPES                    : tumor types its rules can suggest
    PARAMS                   : hand-set parameters
    run(img, method=None, overrides=None) -> common.Result

To add a body part, create sections/<name>/ and register it below.
"""
from . import brain

SECTIONS = {
    "brain": brain,
    # "breast": breast,   # next
}


def get(name):
    try:
        return SECTIONS[name]
    except KeyError:
        raise ValueError(f"unknown section '{name}', choose from {list(SECTIONS)}")
