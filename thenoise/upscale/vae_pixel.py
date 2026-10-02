"""The VAE round-trip latent upscaler strategy.

``VAEPixelUpscaler`` needs no extra weights: it decodes the canonical latent with
the model's *own* VAE, upscales the pixels with bicubic interpolation, and encodes
them back into a canonical latent — at the cost of one extra decode + encode per
upscale request and the lossiness of the round trip.

The upscale itself happens in fp32 regardless of the VAE's dtype: bicubic has
negative lobes and a bf16 mantissa is 8 bits wide, coarse enough to band the smooth
gradients this path exists to enlarge. Its overshoot is clamped back to the VAE's
own ``[-1, 1]`` pixel range before encoding.

Usage:
    upscaler = VAEPixelUpscaler(model.vae, scale=2)
    z_up = upscaler(z)   # canonical [B,C,H,W] -> canonical [B,C,2H,2W]
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from .base import LatentUpscaler

logger = logging.getLogger(__name__)


class VAEPixelUpscaler(LatentUpscaler):
    """VAE round-trip upscale: canonical latent in, canonical latent out.

    Wraps the adapter's *own* VAE instance, which is already registered with the
    model's ``MemoryManager`` and kept resident. Device and dtype are read off the
    VAE on every call, so the round trip follows it wherever it is moved.
    """

    def __init__(self, vae: Any, *, scale: int):
        """Wrap ``vae`` for a ``scale``x latent round trip.

        ``scale`` has no architectural default, so the adapter states it. The VAE
        must be able to go both ways.
        """
        if scale < 1:
            raise ValueError(f"upscale scale must be >= 1, got {scale}")
        for method in ("decode_to_pixels", "encode_pixels_to_latents"):
            if not callable(getattr(vae, method, None)):
                raise ValueError(
                    f"this VAE ({type(vae).__name__}) does not support a {method}() "
                    "round trip, so it cannot be used as a latent upscaler"
                )

        self.vae = vae
        self.scale = scale
        logger.info(
            "Using VAE round-trip latent upscaler (%sx, %s)", scale, type(vae).__name__
        )

    # ------------------------------------------------------------- placement
    @property
    def device(self) -> torch.device:
        """Compute device of the wrapped VAE."""
        return self.vae.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype the wrapped VAE runs in."""
        return self.vae.dtype

    # ---------------------------------------------------------------- transform
    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Upscale the canonical latent ``scale``x by round-tripping through pixels."""
        z = latents.to(device=self.device, dtype=self.dtype)
        pixels = self.vae.decode_to_pixels(z)
        if pixels.ndim == 5:  # [B, C, 1, H, W] -> [B, C, H, W] (video layout)
            pixels = pixels.squeeze(2)

        h, w = pixels.shape[-2:]
        pixels = F.interpolate(
            pixels.float(),
            size=(h * self.scale, w * self.scale),
            mode="bicubic",
            align_corners=False,
        ).clamp(-1.0, 1.0)

        return self.vae.encode_pixels_to_latents(pixels.to(self.dtype))


__all__ = ["VAEPixelUpscaler"]
