"""Model catalog: the set of registered DiffusionModel classes.

To add a new model, write a ``DiffusionModel`` subclass (with ``name`` and a
``detect(f)`` routine) and append it to ``MODEL_CATALOG``.
"""
from __future__ import annotations

from typing import List, Type

from .base import DiffusionModel
from .anima import AnimaModel
from .flux_klein import FluxKleinModel
from .sdxl import SdxlModel
from .krea2 import Krea2Model
from .zimage import ZImageModel

MODEL_CATALOG: List[Type[DiffusionModel]] = [
    Krea2Model,
    AnimaModel,
    ZImageModel,
    FluxKleinModel,
    SdxlModel,
]


def resolve(dit_path: str) -> Type[DiffusionModel]:
    """Determine the model class from a DiT checkpoint.

    Opens the safetensors header ONCE and passes the handle to each registered
    class's ``detect()`` until one matches.
    """
    from safetensors import safe_open

    with safe_open(dit_path, framework="pt") as f:
        for cls in MODEL_CATALOG:
            if cls.detect(f):
                return cls
    names = ", ".join(cls.name for cls in MODEL_CATALOG)
    raise ValueError(f"could not determine model type from {dit_path} (known: {names})")


__all__ = ["MODEL_CATALOG", "DiffusionModel", "resolve"]
