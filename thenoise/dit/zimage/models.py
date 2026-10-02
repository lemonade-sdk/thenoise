"""Z-Image (S3-DiT) transformer — the Lumina core with learned alignment padding."""
from __future__ import annotations

from thenoise.dit.lumina.models import (
    ADALN_EMBED_DIM,
    SEQ_MULTI_OF,
    LuminaTransformer2DModel,
)

__all__ = ["SEQ_MULTI_OF", "ADALN_EMBED_DIM", "ZImageTransformer2DModel"]


class ZImageTransformer2DModel(LuminaTransformer2DModel):
    """The S3-DiT shipped by Z-Image / Z-Image-Turbo."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("pad_mode", "learned")
        super().__init__(*args, **kwargs)
