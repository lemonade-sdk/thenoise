"""Depth-Anything-V2 depth maps: relative depth (larger = closer), measured at the
model's resolution and interpolated back, then painted per-image min-max.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import ClassVar

import torch
import torch.nn.functional as F
from PIL import Image

from thenoise.utils.image_tensor import flatten_alpha
from .base import Preprocessor, load_processor, require_checkpoint
from .options import DepthOptions
from .render import depth_to_pil

class DepthPreprocessor(Preprocessor):
    """The default depth extractor: Depth-Anything-V2 Small, Apache-2.0."""

    name: ClassVar[str] = "depth-anything-v2-small"
    kind: ClassVar[str] = "depth"
    DTYPE: ClassVar[torch.dtype] = torch.bfloat16

    def load(self, device, path: str | None = None) -> None:
        """Load the checkpoint directory ``path``; its name is the pool's name for it."""
        self.device = device
        self.name = os.path.basename(path) if path else self.name
        require_checkpoint(self, path)
        self._processor = load_processor(path)
        self._model = load_model(path, self.DTYPE).to(device).eval()

    def unload(self) -> None:
        del self._model
        del self._processor

    def detect(self, image: Image.Image, options: DepthOptions) -> torch.Tensor:
        """The raw fp32 relative-depth field at the input's size, ``[1, H, W]``."""
        # Control models are RGB: composite any alpha onto white before the processor.
        rgb = flatten_alpha(image)
        inputs = self._processor(images=rgb, return_tensors="pt").to(self.device, self.DTYPE)
        with torch.no_grad():
            depth = self._model(**inputs).predicted_depth  # [B, h, w]
        return F.interpolate(
            depth.unsqueeze(1), size=image.size[::-1], mode="bilinear", align_corners=False
        )[0].float()

    def render(self, field: torch.Tensor, options: DepthOptions | None = None) -> Image.Image:
        return depth_to_pil(field, **asdict(options or DepthOptions()))


def load_model(path: str, dtype) -> torch.nn.Module:
    """The Depth-Anything model of a checkpoint directory."""
    from transformers import AutoModelForDepthEstimation

    return AutoModelForDepthEstimation.from_pretrained(path, dtype=dtype)


__all__ = [
    "DepthOptions",
    "DepthPreprocessor",
    "load_model",
]
