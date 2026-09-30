"""The SesquiLSR latent upscaler strategy.

``SesquiLSRUpscaler`` is the ``LatentUpscaler`` implementation built on the
vendored SesquiLSR network (``sesqui_net.SesquiLSRNet``). It is constructed from
a *latent format name* and owns the whole latent-domain transform: it converts
the canonical latent into the raw VAE space Sesqui was trained on, upscales it,
and converts the result back, so callers never see the raw space or the adaptor.

The format name selects the adaptor factory, the committed weight file and the
raw channel count from ``_UPSCALER_FORMATS``. A format must be added there
together with its weights before it can be used; unknown formats raise.

Usage:
    upscaler = SesquiLSRUpscaler("wan21", device="cuda", dtype=torch.bfloat16)
    z_up = upscaler(z)   # canonical [B,C,H,W] -> canonical [B,C,2H,2W]
"""
from __future__ import annotations

import logging
from typing import Union

import torch

from .base import LatentUpscaler
from .inference_adaptors import (
    LatentFormatAdaptor,
    make_flux,
    make_flux2,
    make_wan21,
)
from .sesqui_net import SesquiLSRNet
from thenoise.inference import freeze
from thenoise.utils.safetensors import load_safetensors, upscale_weight_path

logger = logging.getLogger(__name__)

# Latent format name -> (adaptor factory, weight filename, raw-VAE channel count).
# A format must be added here together with its upscaler weights before it can
# be selected. ``wan21`` (Qwen-Image VAE: Krea2/Anima/Qwen-Image) and ``flux``
# (Flux VAE: Z-Image) weights are committed.
_UPSCALER_FORMATS = {
    "wan21": (make_wan21, "upscaler_Wan21.safetensors", 16),
    "flux":  (make_flux,  "upscaler_flux.safetensors", 16),
    "flux2": (make_flux2, "upscaler_flux2.safetensors", 32),
    # "sdxl":     (make_sdxl,     "upscaler_sdxl.safetensors", 4),  # not yet committed
    # "ideogram4":(make_ideogram4, "upscaler_ideogram4.safetensors", 32),  # not yet committed
}


def _load_net(
    format_name: str,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> tuple[SesquiLSRNet, LatentFormatAdaptor]:
    """Load the Sesqui network + adaptor pair for ``format_name``.

    The adaptor and weight file are selected from ``_UPSCALER_FORMATS`` by name;
    the corresponding ``make_*`` factory is called internally. Formats without
    committed weights raise ``ValueError`` (groundwork for future VAE support).

    The state dict is shipped as bf16 to match the engine's bf16-only convention
    (the upstream README notes half-precision has no quality effect).
    """
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
    state_dict = load_safetensors(path, device=device)

    net = SesquiLSRNet(in_channels=channels)
    net.load_state_dict(state_dict)
    net.to(device=device, dtype=dtype)
    freeze(net)

    logger.info("Latent upscaler ready on %s (%s)", device, dtype)
    return net, adaptor


class SesquiLSRUpscaler(LatentUpscaler):
    """SesquiLSR latent upscale: canonical latent in, canonical latent out.

    Loads the network and builds the format adaptor at construction time, so an
    adapter that returns one of these has already paid the (small, ~6MB) weight
    cost. The engine loads upscalers lazily — ``DiffusionModel.get_upscaler``
    builds one on the first request that actually upscales — so a generation
    that never upscales never touches them.

    The state dict and the network run in the model's dtype; the adaptor's
    mean/std/scale arithmetic runs in fp32 (the vendored adaptors upcast), which
    is what keeps a bf16 latent from drifting through the normalization.
    """

    # Fixed by the architecture: the reassembly head pixel-shuffles 2x. Not a knob
    # — Sesqui upscales 2x or not at all.
    scale = 2

    def __init__(
        self,
        format_name: str,
        *,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self.format = format_name
        self.net, self.adaptor = _load_net(format_name, device=device, dtype=dtype)

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Upscale the canonical latent ``scale``x, staying in canonical space.

        Sesqui operates on *raw* VAE latents, so the canonical latent goes
        through ``to_vae_latent`` and the result back through
        ``from_vae_latent``. The target is computed in canonical (external)
        coordinates and converted by the adaptor, which is what handles formats
        whose raw latent is a different spatial size than the pipeline's (the
        patchified Flux.2 one).
        """
        z = latents.to(device=self.device, dtype=self.dtype)
        scale = self.scale

        # Adaptor math in fp32; the network runs in the model dtype.
        raw = self.adaptor.to_vae_latent(z).to(self.dtype)
        h, w = z.shape[-2:]
        target = self.adaptor.vae_target_size((scale * h, scale * w))
        raw_up = self.net(raw, target)
        z_up = self.adaptor.from_vae_latent(raw_up.float()).to(self.dtype)

        return z_up


__all__ = ["SesquiLSRUpscaler"]
