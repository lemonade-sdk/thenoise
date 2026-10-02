"""Z-Image flow-matching sampler helpers.

Z-Image-Turbo is a distilled flow model: its schedule is ``linspace(1, 1/steps,
steps)`` pushed through a static flow shift, plus a trailing 0 sigma. The schedule
carries *sigmas*; the DiT's model timestep ``t = 1 - sigma`` is derived in the
adapter's ``denoise_step``.
"""
from __future__ import annotations

import torch


#: Static flow shift of the Z-Image-Turbo scheduler config.
SHIFT = 3.0


def get_sigmas(steps: int, device: torch.device) -> torch.Tensor:
    """Z-Image-Turbo sigma grid (1 -> ~1/steps) plus a trailing 0."""
    sigmas = torch.linspace(1.0, 1.0 / steps, steps)
    sigmas = SHIFT * sigmas / (1.0 + (SHIFT - 1.0) * sigmas)
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas.to(torch.float32).to(device)
