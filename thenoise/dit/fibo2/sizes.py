"""The canvas sizes Fibo 2.0 was trained at.

The model takes any ``16``-aligned grid, but it was *shown* about a megapixel: Bria's
pipeline turns an aspect ratio into one of those sizes rather than scaling a request
freely, so an edit whose result is named by its reference lands where the network was.
"""
from __future__ import annotations

import math
from typing import Tuple

#: Area the trained grids cluster around, and the fraction of it they may miss by.
TRAINED_AREA = 1024 * 1024
TRAINED_BAND = 0.1

# The grid is the VAE's own: whole 16-pixel latent cells, never narrower than 4 cells.
_STEP = 16
_MIN_SIDE = 64


def trained_size(aspect_ratio: float) -> Tuple[int, int]:
    """``(width, height)`` the model was trained at for a ``width / height`` ratio.

    Of the grids whose sides are whole latent cells and whose area is within
    :data:`TRAINED_BAND` of :data:`TRAINED_AREA`: the aspect that costs the least
    cropping, then the area closest to a megapixel.
    """
    if aspect_ratio <= 0:
        raise ValueError(f"aspect ratio must be positive, got {aspect_ratio}")
    candidates = (
        (w, h)
        for w in range(_MIN_SIDE, 2 * TRAINED_AREA // _MIN_SIDE, _STEP)
        for h in range(
            max(_MIN_SIDE,
                _STEP * math.ceil((1 - TRAINED_BAND) * TRAINED_AREA / w / _STEP)),
            _STEP * math.floor((1 + TRAINED_BAND) * TRAINED_AREA / w / _STEP) + 1,
            _STEP,
        )
    )
    return min(
        candidates,
        key=lambda s: (
            round(1 - min(s[0] / s[1] / aspect_ratio, aspect_ratio * s[1] / s[0]), 9),
            abs(s[0] * s[1] - TRAINED_AREA),
            -s[0],
        ),
    )


__all__ = ["TRAINED_AREA", "TRAINED_BAND", "trained_size"]
