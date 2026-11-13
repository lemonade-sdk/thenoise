"""Wan 2.2 TI2V-5B flow schedule (fixed model shift)."""
from __future__ import annotations

import math

from thenoise.utils.math import generalized_time_shift

# Wan 2.2 TI2V-5B checkpoint's training/inference shift, not an API control.
FLOW_SHIFT = 5.0


def get_sigmas(steps: int, *, shift: float = FLOW_SHIFT) -> list[float]:
    return [generalized_time_shift(1 - i / steps, math.log(shift), 1.0)
            for i in range(steps)] + [0.0]


def wan_frame_grid(wanted: float, *, upward: bool) -> int:
    if not math.isfinite(wanted) or wanted <= 0:
        raise ValueError("video duration / frame count must be positive and finite")
    # Wan 2.2 packs (F-1)/4 temporal cells. Explicit frames never shrink;
    # an approximate duration chooses the nearest valid frame count.
    cells = (wanted - 1) / 4
    # Duration uses ties-away-from-zero frame rounding.
    n = math.ceil(cells) if upward else math.floor(cells + 0.5)
    return 4 * max(1, n) + 1
