"""Mage-Flow flow-matching schedule: a *static* shift of a uniform sigma grid.

Upstream builds a ``FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,
shift=6.0, use_dynamic_shifting=False)`` and feeds it ``sigmas =
linspace(1, 1/steps, steps)`` (no terminal zero: the scheduler appends it). With
dynamic shifting off, its step is the plain static shift

    sigma' = shift * sigma / (1 + (shift - 1) * sigma)

which is exactly ``generalized_time_shift(sigma, log(shift), 1.0)`` — the shared
helper — so the grid is resolution-INDEPENDENT (unlike Qwen-Image's or Flux.2's
mu-steered shift). The model is fed the shifted sigma itself as its timestep, the
x1000 happening inside the timestep embedder, which is precisely the ``Step.t``
convention of this repo's Euler loop.
"""
from __future__ import annotations

import math
from typing import List

import torch

from thenoise.utils.math import generalized_time_shift

#: The reference's sampling shift (also ComfyUI's ``sampling_settings["shift"]``).
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
