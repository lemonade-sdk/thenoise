"""Latent- and pixel-domain upscalers.

*latent-domain* — ``base.LatentUpscaler`` is the interface the pipeline drives: a
canonical latent goes in, the canonical latent at ``scale`` times the resolution
comes out. A model adapter picks the strategy that fits its VAE in
``_create_upscaler()``.

*pixel-domain* — ``pixel.PixelUpscalerManager`` loads and runs postprocessing
upscalers on decoded pixels. It needs no diffusion model and is configured
server-wide via ``--upscaler-dir``.
"""
from __future__ import annotations

from .inference_adaptors import (
    LatentFormatAdaptor,
    make_flux,
    make_flux2,
    make_ideogram4,
    make_sdxl,
    make_wan21,
)
from .base import LatentUpscaler
from .sesqui import SesquiLSRUpscaler, _UPSCALER_FORMATS
from .vae_pixel import VAEPixelUpscaler
from .qwen21_transcode import Qwen21TranscodeUpscaler
from .sesqui_net import SesquiLSRNet
from .transcode_net import LatentTranscodeNet

from .esrgan import load_esrgan, detect_esrgan_scale, detect_esrgan_scheme


def load_pixel_upscaler(path: str, device: str) -> tuple:
    """Load a pixel-domain upscaler from a safetensors file; returns ``(model, scale)``."""
    return load_esrgan(path, device=device)


def detect_pixel_upscaler_scale(path: str) -> int:
    """Detect a pixel upscaler's upscale scale (2 or 4) from its header."""
    return detect_esrgan_scale(path)


__all__ = [
    "LatentUpscaler",
    "LatentFormatAdaptor",
    "SesquiLSRUpscaler",
    "VAEPixelUpscaler",
    "Qwen21TranscodeUpscaler",
    "SesquiLSRNet",
    "LatentTranscodeNet",
    "make_flux",
    "make_flux2",
    "make_ideogram4",
    "make_sdxl",
    "make_wan21",
    "load_esrgan",
    "detect_esrgan_scale",
    "load_pixel_upscaler",
    "detect_pixel_upscaler_scale",
]
