"""Checkpoint-detected model catalog for image and video models."""
from __future__ import annotations

from .base import DiffusionModel
from .video_base import VideoModel
from .anima import AnimaModel
from .fibo2 import Fibo2Model
from .flux_klein import FluxKleinModel
from .flux3 import Flux3Model
from .h3_image import H3ImageModel
from .krea2 import Krea2Model
from .mage_flow import MageFlowModel
from .ming_image import MingImageModel
from .moss_image import MossImageModel
from .zimage import ZImageModel
from .qwen_image import QwenImageModel
from .qwen_image21 import QwenImage21Model
from .wan_video import Wan22Model

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
]


def resolve(dit_path: str) -> type:
    """Determine the model class from a DiT checkpoint."""
    from safetensors import safe_open

    with safe_open(dit_path, framework="pt") as f:
        for cls in MODEL_CATALOG:
            if cls.detect(f):
                return cls
    names = ", ".join(cls.name for cls in MODEL_CATALOG)
    raise ValueError(f"could not determine model type from {dit_path} (known: {names})")


__all__ = ["MODEL_CATALOG", "DiffusionModel", "VideoModel", "resolve"]
