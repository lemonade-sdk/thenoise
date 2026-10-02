"""Tensor post-processing filters for decoded pixel tensors.

Filters take the fp32 GPU pixel tensor ``[C, H, W]`` in ``[-1, 1]`` and return the
same shape.
"""
from __future__ import annotations

from .film_grain import film_grain
from .nyquist import nyquist_notch
from .rcas import rcas

__all__ = ["film_grain", "nyquist_notch", "rcas"]
