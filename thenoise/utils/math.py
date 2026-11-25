"""Small, dependency-free math helpers shared across the codebase."""

from __future__ import annotations

import math


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the nearest multiple of ``multiple``."""
    return ((value + multiple - 1) // multiple) * multiple


def generalized_time_shift(t, mu: float, sigma: float) -> float:
    """Generalized time/SNR shift: ``exp(mu) / (exp(mu) + (1/t - 1)^sigma)``."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def time_shift(t: float, shift: float) -> float:
    """A flow-matching time under a shift: ``shift·t / (1 + t·(shift − 1))``.

    ``generalized_time_shift(t, log(shift), 1.0)`` in closed form: the same curve — to the
    last bit or within one — without the ``1/t`` at the clean end, which a schedule that
    runs to it cannot afford, or an ``exp``/``log`` round trip.
    """
    return shift * t / (1.0 + t * (shift - 1.0))



def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    """Linearly interpolate the flow-matching shift for a sequence length.

    Short sequences get ``base_shift``, long sequences ``max_shift``.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b
