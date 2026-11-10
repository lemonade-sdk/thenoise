"""The H3-Image denoising schedule: the checkpoint's own sigma grid, walked uniformly.

The output head is one head per interval of that grid, so the grid is the shape of the
trained model rather than a tuning knob and the ladder runs over its *index*. The flow
shift never applies: it would move the ladder off the points the heads were distilled
for.
"""
from __future__ import annotations

from typing import List, Sequence


def sigma_at(grid: Sequence[float], t: float) -> float:
    """The sigma at timestep ``t`` (``0`` clean, ``1`` noise), linear in the grid index.

    ``grid`` decreases from 1 to 0, so it is traversed from its clean end.
    """
    ascending = tuple(grid)[::-1]
    n = len(ascending) - 1
    pos = min(max(float(t), 0.0), 1.0) * n
    i = min(int(pos), n - 1)
    return ascending[i] + (ascending[i + 1] - ascending[i]) * (pos - i)


def sigma_ladder(grid: Sequence[float], steps: int) -> List[float]:
    """The run's sigmas, descending, with the terminal ``0.0`` appended."""
    ladder = [sigma_at(grid, 1.0 - k / steps) for k in range(steps)]
    ladder.append(0.0)
    return ladder


def percent_to_sigma(grid: Sequence[float], percent: float) -> float:
    """Denoising progress (0 = start, 1 = clean) as a sigma on the grid."""
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    return sigma_at(grid, 1.0 - percent)


__all__ = ["percent_to_sigma", "sigma_at", "sigma_ladder"]
