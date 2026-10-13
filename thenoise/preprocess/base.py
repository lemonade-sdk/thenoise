"""The preprocessor contract and its request options: *detect* a control signal, then
*render* it, so a vector signal (a skeleton) is drawn rather than stretched.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from PIL import Image


@dataclass
class PreprocessOptions:
    """What a request asks of a kind. A kind with nothing to ask uses this as-is."""


def scale_field(field: torch.Tensor) -> torch.Tensor:
    """A ``[1, H, W]`` float field in ``[0, 1]`` -> ``[1, H, W]`` uint8 grey levels."""
    return (field[:1].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def uint8_to_pil(x: torch.Tensor) -> Image.Image:
    """A ``[1, H, W]`` uint8 field -> 8-bit greyscale image."""
    return Image.fromarray(x[0].cpu().numpy(), mode="L")


def field_to_pil(field: torch.Tensor) -> Image.Image:
    """``[1, H, W]`` float field in ``[0, 1]`` -> 8-bit greyscale image."""
    return uint8_to_pil(scale_field(field))


class Preprocessor(ABC):
    """One control-map extractor: :meth:`detect` measures, :meth:`render` paints, both
    at the size of the image it was given."""

    name: ClassVar[str] = ""
    kind: ClassVar[str] = ""
    # The options dataclass requests for this kind carry.
    OPTIONS: ClassVar[type] = PreprocessOptions
    # The dtype its weights run in; not every auxiliary network is happy in BF16.
    DTYPE: ClassVar[torch.dtype]

    def __init__(self) -> None:
        self.device: Any = "cpu"

    def load(self, device: Any, path: str | None = None) -> None:
        """Put the weights on ``device``. Called once, under the inference lock.
        ``path`` is the checkpoint directory a discoverable preprocessor loads from."""
        self.device = device

    def unload(self) -> None:
        """Drop anything resident on the device."""

    @abstractmethod
    def detect(self, image: Image.Image, options: PreprocessOptions) -> Any:
        """Measure the control signal on the input image."""

    def render(self, found: Any, options: PreprocessOptions | None = None) -> Image.Image:
        """Paint what :meth:`detect` measured. Override to paint it differently."""
        return field_to_pil(found)

    def run(self, image: Image.Image, options: PreprocessOptions) -> Image.Image:
        return self.render(self.detect(image, options), options)


__all__ = ["PreprocessOptions", "Preprocessor", "field_to_pil", "scale_field", "uint8_to_pil"]
