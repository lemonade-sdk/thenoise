"""The VAE round-trip latent upscaler strategy.

``VAEPixelUpscaler`` is the ``LatentUpscaler`` implementation that needs no extra
weights at all: it decodes the canonical latent with the model's *own* VAE,
upscales the pixels with bicubic interpolation, and encodes them back into a
canonical latent. What it trades for being weight-free and VAE-agnostic is one
extra decode + encode per upscale request, and the lossiness of a VAE round trip —
the re-encoded latent is a reconstruction of an upsampled reconstruction, so any
residual the VAE cannot represent is gone before the refine ever sees it.

That makes it the fallback in this lineup: a trained latent-domain network
(``sesqui.SesquiLSRUpscaler``) is both cheaper and better where its weights exist.
It is the option for a VAE no upscaler was trained against, and it is genuinely
generic — it speaks nothing but the engine's VAE interface
(``decode_to_pixels`` / ``encode_pixels_to_latents``), so it carries an arbitrary
latent format, channel count, spatial compression and pixel width (including
alpha: an RGBA VAE round-trips its alpha through the resize like any other
channel, which no RGB-only pixel-domain upscaler can do).

The upscale itself happens in fp32 regardless of the VAE's dtype: bicubic has
negative lobes and a bf16 mantissa is 8 bits wide, which is coarse enough to band
the smooth gradients this path exists to enlarge. The overshoot those lobes
produce is clamped back to the VAE's own ``[-1, 1]`` pixel range before encoding.

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

    Holds the adapter's *own* VAE instance rather than loading a second one, so
    there are no extra weights to place: the VAE is already registered with the
    model's ``MemoryManager`` and kept resident (the pipeline never offloads it),
    which is what makes it usable from the upscale-and-refine stage, where the
    DiT is the component being managed.

    Device and dtype are read off the VAE on every call rather than cached, so the
    round trip follows the VAE wherever it is moved.
    """

    def __init__(self, vae: Any, *, scale: int):
        """Wrap ``vae`` for a ``scale``x latent round trip.

        ``scale`` is required rather than defaulted: unlike a trained network,
        nothing in the architecture fixes the factor here, so the adapter that
        picks this strategy has to state the one it advertises through
        ``DiffusionModel.UPSCALE_SCALE``.

        The VAE must be able to go both ways — the Flux VAE is decoder-only, so
        wiring this up to one fails here rather than on the first upscale request.
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
        """Upscale the canonical latent ``scale``x by round-tripping through pixels.

        Both VAE calls use the canonical (normalised) latent the pipeline already
        carries, so there is no format conversion and the result is directly the
        latent the refine denoise and the final decode expect.
        """
        z = latents.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
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
