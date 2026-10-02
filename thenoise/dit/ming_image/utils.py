"""Ming-Image DiT loading — one module tree, both released checkpoint namings.

``lumina_state_map`` covers the split (bf16) and fused (int8-convrot) exports: its
qkv fold is a no-op when the file is already fused.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from accelerate import init_empty_weights

from thenoise.dit.lumina.keys import lumina_key_map, lumina_state_map
from thenoise.dit.ming_image.models import MING_IMAGE_DIT_CONFIG, MingImageTransformer2DModel
from thenoise.utils.loader import load_dit

logger = logging.getLogger(__name__)


def load_ming_dit(
    dit_path: str,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    config: Optional[dict] = None,
) -> MingImageTransformer2DModel:
    """Build the Ming-Image S3-DiT on meta and load weights (bf16 or int8-convrot).

    A file this loader cannot fill is an error.
    """
    device = torch.device(device)
    cfg = dict(MING_IMAGE_DIT_CONFIG)
    if config:
        cfg.update(config)

    logger.info("Loading Ming-Image DiT weights from %s", dit_path)
    with init_empty_weights():
        dit = MingImageTransformer2DModel(**cfg)

    return load_dit(
        dit,
        dit_path,
        device=device,
        dtype=dtype,
        key_map=lumina_key_map,
        state_map=lumina_state_map,
    )


__all__ = ["load_ming_dit"]
