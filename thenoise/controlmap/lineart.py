"""Line-art edge maps in pure torch: thin, clean dark lines on a white canvas.
The ControlNet ``lineart`` conditioning input, approximated without a model: the same
Sobel gradients and non-maximum suppression as Canny, but a single threshold (no
hysteresis) and the conventional inverted palette (dark lines, white background).
"""
from __future__ import annotations

from typing import ClassVar

import torch
from PIL import Image

from thenoise.utils.image_tensor import pil_to_pixels
from .base import Preprocessor
from .canny import _GRAY_SCALE, _blur, _sobel, _suppress, luminance, thicken
from .options import LineartOptions


class LineartPreprocessor(Preprocessor):
    """White canvas, dark line-art — the ControlNet ``lineart`` conditioning input."""

    name: ClassVar[str] = "lineart"
    kind: ClassVar[str] = "lineart"
    DTYPE: ClassVar[torch.dtype] = torch.float32

    def detect(self, image: Image.Image, options: LineartOptions) -> torch.Tensor:
        """The line-art mask at the input size as a ``[1, H, W]`` float tensor in ``{0, 1}``."""
        rgb = pil_to_pixels(image, 3).to(self.device, self.DTYPE)
        gray = luminance(rgb)
        gx, gy = _sobel(_blur(gray))
        magnitude = _GRAY_SCALE * torch.sqrt(gx * gx + gy * gy)
        thin = _suppress(magnitude, gx, gy)[0]  # [1, H, W]
        hi = thin.max()
        normalized = thin / hi if hi > 0 else thin
        lines = thicken((normalized >= options.threshold).to(gray.dtype), options.thickness)
        # Inverted palette: white background, dark lines.
        return 1.0 - lines


__all__ = ["LineartOptions", "LineartPreprocessor"]
