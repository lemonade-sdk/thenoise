"""Mage-Flow flow-matching schedule: a *static* shift of a uniform sigma grid.

Upstream is ``FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0,
use_dynamic_shifting=False)`` fed ``linspace(1, 1/steps, steps)``, whose step is

    sigma' = shift * sigma / (1 + (shift - 1) * sigma)

— ``generalized_time_shift(sigma, log(shift), 1.0)``, the shared helper. With dynamic
shifting off the grid is resolution-INDEPENDENT, unlike the mu-steered ones, and the
model is fed the shifted sigma itself (the x1000 happens in its timestep embedder),
which is exactly the ``Step.t`` convention of this repo's Euler loop.
"""
from __future__ import annotations

import math
from typing import List

import torch

from thenoise.utils.math import generalized_time_shift

#: The reference's sampling shift (ComfyUI's ``sampling_settings["shift"]``).
SHIFT = 6.0


def get_sigmas(steps: int, shift: float = SHIFT) -> torch.Tensor:
    """The ``steps`` shifted sigmas, descending from 1.0 (terminal 0 NOT included)."""
    sigmas = torch.linspace(1.0, 1.0 / steps, steps)
    return generalized_time_shift(sigmas, math.log(shift), 1.0)


def get_schedule(steps: int, shift: float = SHIFT) -> List[torch.Tensor]:
    """Return ``steps+1`` timesteps in ``[0, 1]`` (the last is the terminal 0)."""
    sigmas = get_sigmas(steps, shift)
    return [sigmas[i] for i in range(steps)] + [torch.zeros((), dtype=sigmas.dtype)]


__all__ = ["SHIFT", "get_schedule", "get_sigmas"]
