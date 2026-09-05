"""Qwen-Image flow-matching schedule (FlowMatchEulerDiscreteScheduler, dynamic shift).

Ported from kohya-ss/musubi-tuner's ``qwen_image/qwen_image_utils.py``. Only the
sigma schedule is needed here; the adapter's ``schedule()`` builds the project
``Step`` list from it. The model timestep is the *flow* timestep in ``[0, 1]``
(sigmas), and the shared Euler loop integrates ``x -= delta * v``.
"""
from __future__ import annotations

import math
from typing import List

import torch

from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _time_shift_exponential(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_sigmas(steps: int, image_seq_len: int, mu: float) -> torch.Tensor:
    """Return the flow timesteps (sigmas in ``[0, 1]``) for ``steps`` denoise steps."""
    sigmas = torch.linspace(1.0, 1.0 / steps, steps)
    sigmas = _time_shift_exponential(mu, 1.0, sigmas)
    # Stretch to terminate at the configured shift_terminal (0.02). With a single
    # step the last sigma is already 1.0 (``one_minus_z[-1] == 0``), so the
    # terminal stretch would divide by zero; leave the grid as-is instead.
    one_minus_z = 1 - sigmas
    if one_minus_z[-1] > 0:
        scale_factor = one_minus_z[-1] / (1 - 0.02)
        sigmas = 1 - one_minus_z / scale_factor
    return sigmas


def compute_mu(image_seq_len: int) -> float:
    from .utils import calculate_shift

    return calculate_shift(image_seq_len)


def get_schedule(steps: int, image_seq_len: int) -> List[torch.Tensor]:
    """Return ``steps+1`` flow timesteps in ``[0, 1]`` (the last is the terminal 0)."""
    mu = compute_mu(image_seq_len)
    sigmas = get_sigmas(steps, image_seq_len, mu)
    return [sigmas[i] for i in range(steps)] + [torch.zeros((), dtype=sigmas.dtype)]


__all__ = ["get_schedule", "compute_mu", "get_sigmas"]
