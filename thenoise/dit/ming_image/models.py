"""Ming-Image's S3-DiT — the Lumina core with zero-masked padding and two cond tensors."""
from __future__ import annotations

from thenoise.dit.lumina.models import LuminaTransformer2DModel

__all__ = ["MING_IMAGE_DIT_CONFIG", "MingImageTransformer2DModel"]


MING_IMAGE_DIT_CONFIG = dict(
    patch_size=2,
    f_patch_size=1,
    in_channels=16,
    dim=3840,
    n_layers=30,
    n_refiner_layers=2,
    n_heads=30,
    n_kv_heads=30,
    norm_eps=1e-5,
    cap_feat_dim=2560,
    rope_theta=256.0,
    axes_dims=(32, 48, 48),
)


class MingImageTransformer2DModel(LuminaTransformer2DModel):
    """The S3-DiT shipped by Ming-Image 0.1 (``image_model == "ming_image"``)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("pad_mode", "zero_masked")
        super().__init__(*args, **kwargs)
