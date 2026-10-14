"""The preprocessor contract: *detect* a control signal, then *render* it, so a vector
signal (a skeleton) is drawn rather than stretched.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from PIL import Image

from .options import PreprocessOptions


def scale_field(field: torch.Tensor) -> torch.Tensor:
    """A ``[1, H, W]`` float field in ``[0, 1]`` -> ``[1, H, W]`` uint8 grey levels."""
    return (field[:1].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def uint8_to_pil(x: torch.Tensor) -> Image.Image:
    """A ``[1, H, W]`` uint8 field -> 8-bit greyscale image."""
    return Image.fromarray(x[0].cpu().numpy(), mode="L")


#: The files a checkpoint directory must hold for ``from_pretrained`` to load it.
CHECKPOINT_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")


def field_to_pil(field: torch.Tensor) -> Image.Image:
    """``[1, H, W]`` float field in ``[0, 1]`` -> 8-bit greyscale image."""
    return uint8_to_pil(scale_field(field))


def load_processor(path: str):
    """A checkpoint's image processor, PIL-backed (the fast backend needs torchvision)."""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(path, backend="pil")


class Preprocessor(ABC):
    """One control-map extractor: :meth:`detect` measures, :meth:`render` paints."""

    name: ClassVar[str] = ""
    kind: ClassVar[str] = ""
    # The dtype its weights run in; not every auxiliary network is happy in BF16.
    DTYPE: ClassVar[torch.dtype]
    #: The command that fetches its checkpoints, quoted when they are missing.
    DOWNLOAD: ClassVar[str] = "scripts/download.py --model preprocess"

    def __init__(self) -> None:
        self.device: Any = "cpu"

    def load(self, device: Any, path: str | None = None, **resources: Any) -> None:
        """Put the weights on ``device``; ``path`` is its checkpoint directory, ``resources``
        the extra checkpoint pools a kind needs beside its own."""
        self.device = device

    def unload(self) -> None:
        """Drop anything resident on the device."""

    @abstractmethod
    def detect(self, image: Image.Image, options: PreprocessOptions) -> Any:
        """Measure the control signal on the input image."""

    def render(self, found: Any, options: PreprocessOptions | None = None) -> Image.Image:
        """Paint what :meth:`detect` measured."""
        return field_to_pil(found)

    def run(self, image: Image.Image, options: PreprocessOptions) -> Image.Image:
        return self.render(self.detect(image, options), options)


def require_checkpoint(preprocessor: Preprocessor, path: str | None) -> None:
    """Raise with the remedy when a checkpoint directory is incomplete."""
    missing = [
        name
        for name in CHECKPOINT_FILES
        if not os.path.isfile(os.path.join(path or "", name))
    ]
    if missing:
        raise ValueError(
            f"{preprocessor.kind} preprocessor at {path} is missing {', '.join(missing)}; "
            f"run '{preprocessor.DOWNLOAD}' to fetch them"
        )


__all__ = [
    "PreprocessOptions",
    "Preprocessor",
    "field_to_pil",
    "load_processor",
    "require_checkpoint",
    "scale_field",
    "uint8_to_pil",
]
