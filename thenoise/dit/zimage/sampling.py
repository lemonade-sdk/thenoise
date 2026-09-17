"""Z-Image flow-matching sampler helpers.

Z-Image-Turbo is a distilled flow model. Its default schedule is ``linspace(1,
1/steps, steps)`` pushed through the scheduler's static flow shift (shift = 3.0,
``use_dynamic_shifting`` is False), plus a trailing 0 sigma.

The schedule's ``Step.t`` carries this *sigma* grid (1 -> ~1/steps -> 0); the shared
``DiffusionModel`` Euler loop integrates ``x -= delta * velocity`` with
``delta = sigma - sigma_next``, reproducing the FlowMatch Euler update (where the
velocity is the negated DiT output). The DiT's model timestep ``t = 1 - sigma``
(in [0, 1], scaled by the shared embedding's 1000 time factor) is derived in
the adapter's ``denoise_step`` from the sigma it receives.
"""
from __future__ import annotations

import torch


#: Z-Image-Turbo scheduler config (scheduler/scheduler_config.json): static flow shift.
SHIFT = 3.0


def get_sigmas(steps: int, device: torch.device) -> torch.Tensor:
    """Z-Image-Turbo sigma grid (1 -> ~1/steps) plus a trailing 0.

    The pipeline feeds ``linspace(1, 1/steps, steps)`` through the scheduler's
    ``set_timesteps``, which always applies the static shift (use_dynamic_shifting
    is False): ``sigmas = shift * s / (1 + (shift-1) * s)`` with shift = 3.0.
    """
    sigmas = torch.linspace(1.0, 1.0 / steps, steps)
    sigmas = SHIFT * sigmas / (1.0 + (SHIFT - 1.0) * sigmas)
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas.to(torch.float32).to(device)
