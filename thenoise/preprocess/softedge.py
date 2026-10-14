"""Soft-edge maps in pure torch: continuous, anti-aliased dark edges on a white canvas.
The ControlNet ``softedge`` (HED-style) conditioning input, approximated without a
model: the continuous Sobel magnitude, passed through a smooth threshold so a soft
edge keeps its grey gradient instead of being binarised, then optionally blurred.
"""
from __future__ import annotations

from typing import ClassVar

import torch
from PIL import Image

from thenoise.utils.image_tensor import pil_to_pixels
from .base import Preprocessor
from .canny import _GRAY_SCALE, _sobel, gaussian_blur, luminance
from .options import SoftedgeOptions

#: The width of the smooth threshold transition, in normalised magnitude units.
_SOFTNESS = 0.25


class SoftedgePreprocessor(Preprocessor):
    """White canvas, soft dark edges — the ControlNet ``softedge`` conditioning input."""

    name: ClassVar[str] = "softedge"
    kind: ClassVar[str] = "softedge"
    DTYPE: ClassVar[torch.dtype] = torch.float32

    def detect(self, image: Image.Image, options: SoftedgeOptions) -> torch.Tensor:
        """The soft edge field at the input size as a ``[1, H, W]`` float tensor in ``[0, 1]``."""
        rgb = pil_to_pixels(image, 3).to(self.device, self.DTYPE)
        gray = luminance(rgb)
        gx, gy = _sobel(gray)  # no pre-blur: a soft edge keeps its gradient
        magnitude = (_GRAY_SCALE * torch.sqrt(gx * gx + gy * gy)).squeeze(1)  # [1, H, W]
        hi = magnitude.max()
        normalized = magnitude / hi if hi > 0 else magnitude
        soft = _smoothstep(normalized, options.threshold, options.threshold + _SOFTNESS)
        if options.blur > 0:
            size = max(3, int(round(options.blur * 3)) | 1)
            soft = gaussian_blur(soft.unsqueeze(1), size, options.blur).squeeze(1)
        # Inverted palette: white background, soft dark edges.
        return 1.0 - soft


def _smoothstep(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    """A smooth 0->1 ramp between ``edge0`` and ``edge1``, so edges fade rather than snap."""
    width = max(edge1 - edge0, 1e-6)
    t = ((x - edge0) / width).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


__all__ = ["SoftedgeOptions", "SoftedgePreprocessor"]
