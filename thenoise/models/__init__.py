"""Checkpoint-detected model catalog for image, video and mesh models."""
from __future__ import annotations

from typing import Union

from thenoise.utils.safetensors import Header, as_header

from .base import DiffusionModel
from .video_base import VideoModel
from .anima import AnimaModel
from .fibo2 import Fibo2Model
from .flux_klein import FluxKleinModel
from .flux3 import Flux3Model
from .h3_image import H3ImageModel
from .hunyuan3d import Hunyuan3DShapeModel
from .krea2 import Krea2Model
from .mage_flow import MageFlowModel
from .ming_image import MingImageModel
from .moss_image import MossImageModel
from .zimage import ZImageModel
from .qwen_image import QwenImageModel
from .qwen_image21 import QwenImage21Model
from .wan_video import Wan22Model
from .ltx_video import LTX23Video, LTX25Video

MODEL_CATALOG: list[type] = [
    Krea2Model,
    AnimaModel,
    ZImageModel,
    MingImageModel,
    FluxKleinModel,
    Flux3Model,
    H3ImageModel,
    MageFlowModel,
    QwenImageModel,
    QwenImage21Model,
    MossImageModel,
    Fibo2Model,
    Wan22Model,
    LTX23Video,
    LTX25Video,
    Hunyuan3DShapeModel,
]


def resolve(checkpoint: Union[str, Header]) -> Header:
    """The header of a DiT checkpoint, with the catalog entry that claimed it attached.

    The header is what every later reader wants — geometry, markers, the weight load —
    so the lookup that had to parse it hands it on instead of making them parse it again.
    """
    header = as_header(checkpoint)
    for cls in MODEL_CATALOG:
        if cls.detect(header):
            header.model_cls = cls
            return header
    names = ", ".join(cls.name for cls in MODEL_CATALOG)
    raise ValueError(f"could not determine model type from {header.path} (known: {names})")


__all__ = ["MODEL_CATALOG", "DiffusionModel", "VideoModel", "resolve"]
