"""Wan 2.2's flow schedule: sigmas shifted by whatever the run asked for."""
from __future__ import annotations

from thenoise.utils.math import time_shift


def get_sigmas(steps: int, *, shift: float) -> list[float]:
    return [time_shift(1 - i / steps, shift) for i in range(steps)] + [0.0]


def percent_to_sigma(percent: float, *, shift: float) -> float:
    """The sigma a ``percent`` of the way down the axis ``get_sigmas`` walks."""
    return time_shift(1 - percent, shift)

