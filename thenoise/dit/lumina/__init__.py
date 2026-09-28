"""Lumina / S3-DiT — the shared single-stream DiT family.

The core in ``models.py`` backs Z-Image (``thenoise.dit.zimage``) and Ming-Image
(``thenoise.dit.ming_image``): the same noise/context refiner + unified-block
transformer, the same 3-axis (t, h, w) matrix RoPE, the same
``[image, caption]`` sequence layout and the same pad-to-a-multiple-of-32 stream
geometry. ``keys.py`` holds the checkpoint-name helpers both loaders need.
"""
from .models import (
    ADALN_EMBED_DIM,
    PAD_MODES,
    SEQ_MULTI_OF,
    LuminaTransformer2DModel,
    LuminaTransformerBlock,
    TokenStream,
)

__all__ = [
    "ADALN_EMBED_DIM",
    "PAD_MODES",
    "SEQ_MULTI_OF",
    "LuminaTransformer2DModel",
    "LuminaTransformerBlock",
    "TokenStream",
]
