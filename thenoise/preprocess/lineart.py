"""Line-art edge maps in pure torch: thin, clean dark lines on a white canvas.
The ControlNet ``lineart`` conditioning input, approximated without a model: the same
Sobel gradients and non-maximum suppression as Canny, but a single threshold (no
hysteresis) and the conventional inverted palette (dark lines, white background).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from PIL import Image

from thenoise.utils.image_tensor import pil_to_pixels
from .base import PreprocessOptions, Preprocessor
from .canny import _GRAY_SCALE, _blur, _sobel, _suppress, luminance, thicken


@dataclass
class LineartOptions(PreprocessOptions):
    """Line intensity threshold (0-1) and rendered line thickness."""

    threshold: float = 0.5
    thickness: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.thickness < 1:
            raise ValueError(f"thickness must be >= 1, got {self.thickness}")


class LineartPreprocessor(Preprocessor):
    """White canvas, dark line-art — the ControlNet ``lineart`` conditioning input."""

    name: ClassVar[str] = "lineart"
    kind: ClassVar[str] = "lineart"
    OPTIONS: ClassVar[type] = LineartOptions
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
