"""Ming-Image DiT loading — one module tree, both released checkpoint namings.

``lumina_state_map`` covers the split (bf16) and fused (int8-convrot) exports: its
qkv fold is a no-op when the file is already fused.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from thenoise.dit.lumina.keys import lumina_key_map, lumina_state_map
from thenoise.dit.ming_image.models import MING_IMAGE_DIT_CONFIG, MingImageTransformer2DModel
from thenoise.utils.loader import build_dit
from thenoise.utils.safetensors import Header

logger = logging.getLogger(__name__)


def load_ming_dit(
    checkpoint: Union[str, Header],
    device: Union[str, torch.device],
    dtype: torch.dtype,
    config: Optional[dict] = None,
) -> MingImageTransformer2DModel:
    """Build the Ming-Image S3-DiT on meta and load weights (bf16 or int8-convrot).

    A file this loader cannot fill is an error.
    """
    cfg = dict(MING_IMAGE_DIT_CONFIG)
    if config:
        cfg.update(config)

    return build_dit(
        lambda: MingImageTransformer2DModel(**cfg),
        checkpoint,
        device=device,
        dtype=dtype,
        key_map=lumina_key_map,
        state_map=lumina_state_map,
    )


__all__ = ["load_ming_dit"]
