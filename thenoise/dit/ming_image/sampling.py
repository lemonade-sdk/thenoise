"""Ming-Image's flow-matching schedule — the DYNAMIC shift the reference actually runs.

``generator.py`` overrides the shipped static ``shift: 6.0``, so ``mu`` ramps with
the image token count (see :func:`dynamic_mu`) and the shift is ``exp(mu)``.
"""
from __future__ import annotations

import math
from typing import Union

import torch

from thenoise.utils.math import calculate_shift, generalized_time_shift

#: Pixels per DiT token: the VAE's 8x compression and the DiT's 2x2 patch.
TOKEN_PIXELS = 16

# The shift curve's endpoints (the vendor's ``calculate_shift`` arguments). Its knee
# is the 4096-token (1024x1024) training bucket, 1.15 below it and 1.35 above.
BASE_SEQ_LEN = 256
BASE_SHIFT = 0.5
REFERENCE_SEQ_LEN = 4096
MAX_SHIFT_BELOW = 1.15
MAX_SHIFT_ABOVE = 1.35


def image_seq_len(height: int, width: int) -> int:
    """The DiT's image token count for a pixel size (16 pixels per token)."""
    return (height // TOKEN_PIXELS) * (width // TOKEN_PIXELS)


def dynamic_mu(height: int, width: int) -> float:
    """The scheduler's ``mu`` for this pixel size (the shift is ``exp(mu)``)."""
    seq = image_seq_len(height, width)
    max_seq = max(REFERENCE_SEQ_LEN, seq)
    # ``>`` where the vendor writes ``>=``: at exactly 1024² that takes the 1.35 branch
    # (3.857), contradicting the 3.16 the same reference documents for this bucket.
    max_shift = MAX_SHIFT_ABOVE if seq > REFERENCE_SEQ_LEN else MAX_SHIFT_BELOW
    return calculate_shift(seq, BASE_SEQ_LEN, max_seq, BASE_SHIFT, max_shift)


def dynamic_shift(height: int, width: int) -> float:
    """The flow shift ``exp(mu)`` this pixel size samples at."""
    return math.exp(dynamic_mu(height, width))


def get_sigmas(
    steps: int,
    height: int,
    width: int,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Shifted grid from 1 down to ``1/steps``, plus the trailing 0 the loop ends on."""
    sigmas = generalized_time_shift(
        torch.linspace(1.0, 1.0 / steps, steps), dynamic_mu(height, width), 1.0
    )
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas.to(device)


__all__ = [
    "TOKEN_PIXELS",
    "dynamic_mu",
    "dynamic_shift",
    "get_sigmas",
    "image_seq_len",
]
