"""Small, dependency-free math helpers shared across the codebase."""

from __future__ import annotations

import math


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the nearest multiple of ``multiple``."""
    return ((value + multiple - 1) // multiple) * multiple


def generalized_time_shift(t, mu: float, sigma: float) -> float:
    """Generalized time/SNR shift: ``exp(mu) / (exp(mu) + (1/t - 1)^sigma)``.

    Shared by the flow-matching samplers (Flux.2, Qwen-Image, Krea 2): all three
    apply the same shift to a uniform ``1 -> 0`` grid, only the ``mu``/``sigma``
    source differs per model.
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)



def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    """Linearly interpolate the flow-matching shift for a sequence length.

    Used by the dynamic-shift samplers (e.g. Qwen-Image): short sequences get
    ``base_shift``, long sequences get ``max_shift``, linearly in between.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b
