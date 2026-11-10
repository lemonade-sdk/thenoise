"""The H3-Image denoising schedule: the checkpoint's own sigma grid, walked uniformly.

The DiT's output head is a bank of ``pdd_steps`` heads, one per interval of the sigma
grid baked into the checkpoint's metadata, so the grid is not a tuning knob — it is
the shape of the trained model. The step ladder follows from it: percents run uniformly
over the *grid index*, and the sigmas come out wherever that lands on the distilled
curve. At the release's own step count the ladder is the grid exactly; at any other
count each step spans a whole number of distilled intervals and the head bank averages
them (see :class:`thenoise.dit.h3_image.models.FinalLayer`).

The flow shift therefore never applies: a shift would move the ladder off the points
the heads were distilled for, which the bank papers over rather than fixes.
"""
from __future__ import annotations

from typing import List, Sequence


def sigma_at(grid: Sequence[float], t: float) -> float:
    """The sigma at timestep ``t`` (``0`` clean, ``1`` noise), linear in the grid index.

    ``grid`` decreases from 1 to 0, so it is traversed from its clean end: ``t`` is a
    fraction of the way along it, interpolated between the two neighbouring points.
    """
    ascending = tuple(grid)[::-1]
    n = len(ascending) - 1
    pos = min(max(float(t), 0.0), 1.0) * n
    i = min(int(pos), n - 1)
    return ascending[i] + (ascending[i + 1] - ascending[i]) * (pos - i)


def sigma_ladder(grid: Sequence[float], steps: int) -> List[float]:
    """The run's sigmas, descending, with the terminal ``0.0`` appended.

    ``steps`` intervals of the grid's index space, so the first entry is ``1.0`` and
    the last is the clean end.
    """
    ladder = [sigma_at(grid, 1.0 - k / steps) for k in range(steps)]
    ladder.append(0.0)
    return ladder


def percent_to_sigma(grid: Sequence[float], percent: float) -> float:
    """Denoising progress (0 = start, 1 = clean) as a sigma on the distilled grid."""
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    return sigma_at(grid, 1.0 - percent)


__all__ = ["percent_to_sigma", "sigma_at", "sigma_ladder"]
