"""MOSS-Image flow-matching schedule: a *static* shift of a uniform sigma grid.

``linspace(1, 0, steps + 1)`` pushed through ``shift * s / (1 + (shift - 1) * s)``,
which maps its own endpoints exactly and does not move with the resolution.

Every step is an Euler step of the flow ODE, so the adapter's ``denoise_step`` returns
the model's raw output as the velocity.
"""
from __future__ import annotations

import torch

#: The timestep shift the model was released with.
FLOW_SHIFT = 5.0

__all__ = ["FLOW_SHIFT", "get_schedule"]


def get_schedule(num_steps: int, shift: float = FLOW_SHIFT) -> list[float]:
    """``num_steps + 1`` timesteps from 1 -> 0, statically shifted.

    The ``i``-th Euler step integrates ``x += (t[i+1] - t[i]) * velocity``.
    """
    # Float64 on the way through: float32 does not map the shift's own endpoints
    # exactly, and a first ``t`` above 1 starts the run outside the range the model saw.
    timesteps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
    return timesteps.tolist()
