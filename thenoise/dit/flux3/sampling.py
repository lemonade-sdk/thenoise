"""Flux.3 sampling helpers.

Flux.3 is a flow-matching model on the Flux.2 grid — ``linspace(1, 0, steps + 1)``
pushed through a time shift — but the shift is **static**: the released checkpoints
are trained with one reference shift (ComfyUI's Flux.3 ``sampling_settings``, the
``ln(6.93)`` timestep shift), not the resolution-dependent ``mu`` Flux.2 Klein fits.
Every step is an Euler step of the flow ODE.

A *flow map* (``flow_map=True``) is step-conditioned as well as time-conditioned: it
embeds the timestep each step jumps to, which is :func:`target_times` of the grid
being run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from thenoise.samplers import Step

#: The reference timestep shift (ComfyUI's ``Flux3.sampling_settings["shift"]``).
FLOW_SHIFT = 1.93586

__all__ = ["FLOW_SHIFT", "get_schedule", "target_times"]


def get_schedule(num_steps: int, shift: float = FLOW_SHIFT) -> list[float]:
    """Flux.3 timestep grid: ``(num_steps + 1)`` floats from 1 -> 0, statically shifted.

    The ``i``-th Euler step integrates ``x += (t[i+1] - t[i]) * velocity``, so the
    adapter's ``denoise_step`` returns the model velocity directly.
    """
    # Float64 on the way through: float32 does not map the shift's own endpoints
    # exactly, and ``t = 1`` overshooting to 1.0000001 would put the first step just
    # outside the range the model saw.
    timesteps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
    return timesteps.tolist()


def target_times(schedule: list["Step"]) -> list[float]:
    """The timestep each step of ``schedule`` jumps to, one entry per step.

    ``Step.delta`` is ``t - t_next``, so the target is simply ``t - delta``: the next
    grid point, and 0 on the last step. Deriving it from the schedule the solver is
    actually running keeps a flow-map DiT correct for a user-supplied sigma grid and
    for the refine's short one too.
    """
    return [float(step.t) - float(step.delta) for step in schedule]
