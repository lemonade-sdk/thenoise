"""Depth-Anything-V2 depth maps: relative depth (larger = closer), measured at the
model's own resolution and interpolated back to the input's, then painted by
:mod:`render` (per-image min-max normalisation, not in the model's bf16).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from PIL import Image

from .base import PreprocessOptions, Preprocessor
from .render import depth_to_pil
from thenoise.utils.image_tensor import flatten_alpha

#: The colormaps a depth map can be painted with. ``turbo`` needs a vendored LUT and
#: lands later (M4); keeping the option typed now avoids widening the surface later.
_COLORMAPS = ("gray",)


@dataclass
class DepthOptions(PreprocessOptions):
    """How the relative-depth field is painted."""

    invert: bool = False
    colormap: str = "gray"

    def __post_init__(self) -> None:
        if self.colormap not in _COLORMAPS:
            raise ValueError(
                f"unknown depth colormap {self.colormap!r}; expected one of {_COLORMAPS}"
            )


class DepthPreprocessor(Preprocessor):
    """The default depth extractor: Depth-Anything-V2 Small, Apache-2.0."""

    name: ClassVar[str] = "depth-anything-v2-small"
    kind: ClassVar[str] = "depth"
    OPTIONS: ClassVar[type] = DepthOptions
    DTYPE: ClassVar[torch.dtype] = torch.bfloat16

    def load(self, device, path: str | None = None) -> None:
        """Load the checkpoint directory ``path``; its directory name is the name the
        pool reports it under."""
        self.device = device
        self.name = os.path.basename(path) if path else self.name
        from ._hf import load_depth_model, load_depth_processor, require_depth_files

        require_depth_files(path)
        self._processor = load_depth_processor(path)
        self._model = load_depth_model(path, self.DTYPE).to(device).eval()

    def unload(self) -> None:
        del self._model
        del self._processor

    def detect(self, image: Image.Image, options: DepthOptions) -> torch.Tensor:
        """The raw fp32 relative-depth field at the input's size, ``[1, H, W]``."""
        # Control models are RGB: composite any alpha onto white, so the DPT
        # processor never sees a 4-channel image against a 3-channel mean.
        rgb = flatten_alpha(image)
        inputs = self._processor(images=rgb, return_tensors="pt").to(self.device, self.DTYPE)
        with torch.no_grad():
            depth = self._model(**inputs).predicted_depth  # [B, h, w]
        return F.interpolate(
            depth.unsqueeze(1), size=image.size[::-1], mode="bilinear", align_corners=False
        )[0].float()

    def render(self, field: torch.Tensor, options: DepthOptions | None = None) -> Image.Image:
        return depth_to_pil(field, **asdict(options or DepthOptions()))


__all__ = ["DepthOptions", "DepthPreprocessor"]
