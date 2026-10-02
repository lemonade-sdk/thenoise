"""Qwen-Image 2.1 flow-matching schedule.

A dynamic-shift flow grid (:func:`calculate_shift`) over the *16x* latent token
count, since this model patchifies nothing. At the 1024x1024 default that is 4096
tokens, landing on the ``mu = 0.69`` the reference ships as a fixed shift.
"""
from __future__ import annotations

from typing import List

import torch

from thenoise.utils.math import calculate_shift, generalized_time_shift

#: The scheduler's terminal sigma (the grid is stretched to end on it before the 0).
SHIFT_TERMINAL = 0.02


def get_sigmas(steps: int, image_seq_len: int, mu: float) -> torch.Tensor:
    """Return the flow timesteps (sigmas in ``(0, 1]``) for ``steps`` denoise steps."""
    sigmas = torch.linspace(1.0, 1.0 / steps, steps)
    sigmas = generalized_time_shift(sigmas, mu, 1.0)
    # Stretch to terminate at ``SHIFT_TERMINAL``; with a single step the last sigma
    # is already 1.0 and the stretch would divide by zero.
    one_minus_z = 1 - sigmas
    if one_minus_z[-1] > 0:
        scale_factor = one_minus_z[-1] / (1 - SHIFT_TERMINAL)
        sigmas = 1 - one_minus_z / scale_factor
    return sigmas


def compute_mu(image_seq_len: int) -> float:
    return calculate_shift(image_seq_len)


def get_schedule(steps: int, image_seq_len: int) -> List[torch.Tensor]:
    """Return ``steps+1`` flow timesteps in ``[0, 1]`` (the last is the terminal 0)."""
    sigmas = get_sigmas(steps, image_seq_len, compute_mu(image_seq_len))
    return [sigmas[i] for i in range(steps)] + [torch.zeros((), dtype=sigmas.dtype)]


__all__ = ["get_schedule", "compute_mu", "get_sigmas", "SHIFT_TERMINAL"]
