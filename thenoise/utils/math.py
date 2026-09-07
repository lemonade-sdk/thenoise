"""Small, dependency-free math helpers shared across the codebase."""

from __future__ import annotations


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the nearest multiple of ``multiple``."""
    return ((value + multiple - 1) // multiple) * multiple


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
