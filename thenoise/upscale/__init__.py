"""Latent- and pixel-domain upscalers.

Two independent families live here:

*latent-domain* — ``base.LatentUpscaler`` is the interface the pipeline drives:
a canonical latent goes in, the canonical latent at ``scale`` times the resolution
comes out, ready for the refine denoise and the VAE decode. A model adapter picks
the strategy that fits its VAE by returning one from ``_create_upscaler()``.
``sesqui.SesquiLSRUpscaler`` is the current implementation, built on the vendored
SesquiLSR network (``sesqui_net``, ~6MB bf16 weights per latent format, committed
in ``weights/``). The latent format name it is built from selects the adaptor and
weight file in ``sesqui._UPSCALER_FORMATS``; see ``inference_adaptors.make_*`` for
the formats that have adaptors.

*pixel-domain* — ``pixel.PixelUpscalerManager`` loads and runs postprocessing
upscalers (Real-ESRGAN today) on decoded pixels. It needs no diffusion model and
is configured server-wide via ``--upscaler-dir``.

Usage:
    upscaler = SesquiLSRUpscaler(
        "wan21", device="cuda", dtype=torch.bfloat16, scale=model.UPSCALE_SCALE
    )
    z_up = upscaler(z)   # canonical [B,C,H,W] -> canonical [B,C,2H,2W]
"""
from __future__ import annotations

from .inference_adaptors import (
    LatentFormatAdaptor,
    make_flux,
    make_flux2,
    make_ideogram4,
    make_qwen21,
    make_sdxl,
    make_wan21,
)
from .base import LatentUpscaler
from .sesqui import SesquiLSRUpscaler, upscale_weight_path, _UPSCALER_FORMATS
from .sesqui_net import SesquiLSRNet

from .esrgan import load_esrgan, detect_esrgan_scale, detect_esrgan_scheme


def load_pixel_upscaler(path: str, device: str) -> tuple:
    """Load a pixel-domain upscaler from a safetensors file.

    Generic entry point so the model-facing code never names a specific pixel
    upscaler architecture. Today the only pixel-space upscaler is Real-ESRGAN,
    so this dispatches to ``load_esrgan``; future pixel upscalers plug in here.
    Returns ``(model, scale)``.
    """
    return load_esrgan(path, device=device)


def detect_pixel_upscaler_scale(path: str) -> int:
    """Detect a pixel upscaler's upscale scale (2 or 4) from its header.

    Generic wrapper around the ESRGAN scale detection; see
    ``load_pixel_upscaler`` for the rationale.
    """
    return detect_esrgan_scale(path)


__all__ = [
    "LatentUpscaler",
    "LatentFormatAdaptor",
    "SesquiLSRUpscaler",
    "SesquiLSRNet",
    "upscale_weight_path",
    "make_flux",
    "make_flux2",
    "make_ideogram4",
    "make_qwen21",
    "make_sdxl",
    "make_wan21",
    "load_esrgan",
    "detect_esrgan_scale",
    "load_pixel_upscaler",
    "detect_pixel_upscaler_scale",
]
