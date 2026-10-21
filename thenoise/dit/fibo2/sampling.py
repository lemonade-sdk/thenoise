"""Fibo 2.0 sampling helpers.

One static flow shift over a uniform timestep grid: ``linspace(1, 0, steps + 1)``
through :data:`SHIFT`. Every step is an Euler step of the flow ODE, so the DiT's raw
output is the velocity the shared sampler integrates.
"""
from __future__ import annotations

from typing import Union

import torch

#: Static flow shift of the released sampler. Above 1 it holds every grid point
#: closer to noise than the uniform grid, which the few-step variants want.
SHIFT = 1.15


def shift(sigma: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """The model's own sigma for a uniform-grid ``sigma``."""
    return SHIFT * sigma / (1 + (SHIFT - 1) * sigma)


def get_sigmas(steps: int, device: torch.device) -> torch.Tensor:
    """``steps + 1`` timesteps from 1 -> 0, uniformly spaced then shifted."""
    return shift(torch.linspace(1.0, 0.0, steps + 1, device=device))


__all__ = ["SHIFT", "get_sigmas", "shift"]
