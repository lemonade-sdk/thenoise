"""Wan 2.2's flow schedule: sigmas shifted by whatever the run asked for."""
from __future__ import annotations

import math

from thenoise.utils.math import generalized_time_shift


def get_sigmas(steps: int, *, shift: float) -> list[float]:
    return [generalized_time_shift(1 - i / steps, math.log(shift), 1.0)
            for i in range(steps)] + [0.0]


def percent_to_sigma(percent: float, *, shift: float) -> float:
    """The sigma a ``percent`` of the way down the axis ``get_sigmas`` walks.

    The shift divides by ``t``, so the far end of the axis is answered here.
    """
    if percent >= 1.0:
        return 0.0
    return generalized_time_shift(1 - percent, math.log(shift), 1.0)

