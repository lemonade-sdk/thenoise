"""Tensor post-processing for decoded pixel tensors.

A filter takes the fp32 GPU pixel tensor ``[C, H, W]`` in ``[-1, 1]`` and returns the
same shape, leaving any channel beyond the first three (an alpha matte) alone.

The layout is the one a knob needs: :mod:`.options` declares every knob once — its
bounds, its words, the filter it belongs to and the stage it runs in — and
:mod:`.chain` runs a stage of that declaration. The filters themselves are
submodules (``.rcas``, ``.nyquist``, ``.film_grain``, ``.color``); importing this
package deliberately pulls no torch, so the CLI can build its flags from the
declaration.
"""
from __future__ import annotations

from .chain import FILTERS, apply_stage
from .options import (
    CHAIN,
    DEFAULTS,
    DECODE,
    FINAL,
    POSTPROCESS_NAMES,
    FilterSpec,
    PostprocessOptions,
    filter_specs,
    flat_options,
    options_from,
)

__all__ = [
    "CHAIN",
    "DEFAULTS",
    "DECODE",
    "FINAL",
    "FILTERS",
    "POSTPROCESS_NAMES",
    "FilterSpec",
    "PostprocessOptions",
    "apply_stage",
    "filter_specs",
    "flat_options",
    "options_from",
]
