"""The H3-Image denoising schedule: the checkpoint's own sigma grid, walked uniformly.

The output head is one head per interval of that grid, so the grid is the shape of the
trained model rather than a tuning knob and the ladder runs over its *index*
(``samplers.sigma_ladder``). The flow shift never applies: it would move the ladder off
the points the heads were distilled for.
"""
from __future__ import annotations

from typing import Sequence

from thenoise.samplers import sigma_at


def percent_to_sigma(grid: Sequence[float], percent: float) -> float:
    """Denoising progress (0 = start, 1 = clean) as a sigma on the grid."""
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    return sigma_at(grid, 1.0 - percent)


__all__ = ["percent_to_sigma"]
