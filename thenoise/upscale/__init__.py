"""SesquiLSR latent upscaler, vendored for thenoise.

The Qwen-Image/Anima/Krea2 models share the Wan21 upscaler, Z-Image the Flux one,
and SDXL the 4-channel SDXL one (``weights/upscaler_SDXL.safetensors``, ~6MB bf16).
``load_latent_upscaler`` takes a latent-format name and selects the adaptor
factory + weight file from ``_UPSCALER_FORMATS``; formats without committed
weights raise. See ``inference_adaptors.make_*`` for the available formats.

Usage:
    model, adaptor = load_latent_upscaler("wan21", device="cuda", dtype=torch.bfloat16)
    raw = adaptor.to_vae_latent(latent)          # normalized -> raw VAE latent
    up  = model(raw, (2*h, 2*w))                 # 2x latent upscale
    out = adaptor.from_vae_latent(up)            # raw -> pipeline latent
"""
from __future__ import annotations
from typing import Union

import logging
from pathlib import Path

import torch

from .inference_adaptors import (
    _AffineAdaptor,
    _Flux2BNAdaptor,
    _ShiftScalePatchAdaptor,
    _ZScoreAdaptor,
    LatentFormatAdaptor,
    make_flux,
    make_flux2,
    make_identity,
    make_ideogram4,
    make_sdxl,
    make_wan21,
)
from .upscaler import LatentUpscaler

logger = logging.getLogger(__name__)

from .esrgan import load_esrgan, detect_esrgan_scale, detect_esrgan_scheme

_WEIGHT_DIR = Path(__file__).resolve().parent / "weights"

# Latent format name -> (adaptor factory, weight filename, raw-VAE channel count).
# A format must be added here together with its upscaler weights before it can
# be selected. ``wan21`` (Qwen-Image VAE: Krea2/Anima), ``flux`` (Flux VAE:
# Z-Image) and ``sdxl`` weights are committed.
_UPSCALER_FORMATS = {
    "wan21": (make_wan21, "upscaler_Wan21.safetensors", 16),
    "flux":  (make_flux,  "upscaler_flux.safetensors", 16),
    "flux2": (make_flux2, "upscaler_flux2.safetensors", 32),
    # SDXL pipeline latents are already scaled (raw * 0.13025) = Sesqui's trained
    # space, so SDXL needs an identity adaptor (NOT make_sdxl's affine, which is
    # for ComfyUI's raw node latents).
    "sdxl":  (lambda: make_identity(4), "upscaler_SDXL.safetensors", 4),
    # "ideogram4":(make_ideogram4, "upscaler_ideogram4.safetensors", 32),  # not yet committed
}


def upscale_weight_path(filename: str) -> Path:
    """Path to a committed upscaler weight file."""
    path = _WEIGHT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"upscaler weights not found at {path}; "
            "the package was not installed with its package-data"
        )
    return path


def load_latent_upscaler(
    format_name: str,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> tuple[LatentUpscaler, LatentFormatAdaptor]:
    """Load the latent upscaler for the given ``format_name``.

    The adaptor and weight file are selected from ``_UPSCALER_FORMATS`` by name;
    the corresponding ``make_*`` factory is called internally. Formats without
    committed weights raise ``ValueError`` (groundwork for future VAE support).

    The state dict is shipped as bf16 to match the engine's bf16-only convention
    (the upstream README notes half-precision has no quality effect).
    """
    from safetensors.torch import load_file

    entry = _UPSCALER_FORMATS.get(format_name)
    if entry is None:
        raise ValueError(
            f"unknown latent format '{format_name}'; no upscaler weights "
            f"available. Known formats: {sorted(_UPSCALER_FORMATS)}"
        )
    make_adaptor, filename, channels = entry
    adaptor = make_adaptor()

    path = upscale_weight_path(filename)
    logger.info("Loading Sesqui latent upscaler from %s", path)
    state_dict = load_file(str(path), device=str(device))

    model = LatentUpscaler(in_channels=channels)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    logger.info("Latent upscaler ready on %s (%s)", device, dtype)
    return model, adaptor


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
    "load_latent_upscaler",
    "load_esrgan",
    "detect_esrgan_scale",
    "load_pixel_upscaler",
    "detect_pixel_upscaler_scale",
]
