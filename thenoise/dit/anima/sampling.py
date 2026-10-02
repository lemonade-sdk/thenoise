"""Anima flow-matching sampler helpers.

Ported from the HunyuanImage-2.1-derived module.
"""
from __future__ import annotations

from typing import Tuple

import torch


def get_timesteps_sigmas(sampling_steps: int, shift: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Timesteps and sigmas of a shifted flow-matching schedule."""
    sigmas = torch.linspace(1, 0, sampling_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32, device=device)
    return timesteps, sigmas
