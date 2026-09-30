"""Z-Image (S3-DiT) transformer — the Lumina core with learned alignment padding.

Z-Image pads both token streams up to a multiple of ``SEQ_MULTI_OF`` and fills the
pad slots with the learned ``x_pad_token``/``cap_pad_token`` embeddings, which
attention then reads. That is the core's ``pad_mode="learned"`` mode; everything
else — the stream layout, the refiners, the RoPE geometry — is shared with the rest
of the Lumina/S3-DiT family in ``thenoise.dit.lumina.models`` (Ming-Image is the
other member, and pads with masked-out zeros instead).
"""
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
