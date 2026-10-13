"""Paint measured control fields: the render half of detect/render.

A depth field is normalised per image in fp32 (min-max, not in the model's bf16),
then mapped to 8-bit grey; ``invert`` flips near/far, which is the ControlNet depth
convention either way.
"""
from __future__ import annotations

import torch
from PIL import Image

from .base import scale_field, uint8_to_pil


def normalize(field: torch.Tensor) -> torch.Tensor:
    """A ``[1, H, W]`` fp32 field -> ``[0, 1]``. A flat field stays flat rather than
    amplifying its noise into a full-contrast image."""
    lo, hi = field.min(), field.max()
    if hi > lo:
        return (field - lo) / (hi - lo)
    return field.clamp(0.0, 1.0)


def depth_to_pil(
    field: torch.Tensor, invert: bool = False, colormap: str = "gray"
) -> Image.Image:
    """A ``[1, H, W]`` fp32 relative-depth field -> an 8-bit image at the same size."""
    x = normalize(field)
    if invert:
        x = 1.0 - x
    if colormap == "gray":
        return uint8_to_pil(scale_field(x))
    raise ValueError(f"colormap {colormap!r} is not available yet")


__all__ = ["depth_to_pil", "normalize"]
