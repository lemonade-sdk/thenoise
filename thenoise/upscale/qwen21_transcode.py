"""The Qwen-Image 2.1 latent-transcode upscaler strategy.

``Qwen21TranscodeUpscaler`` is the ``LatentUpscaler`` built on the vendored
transcode bridge (``transcode_net.LatentTranscodeNet``): a small network that turns
the model's canonical latent into the same latent at 2x the resolution, conditioned
on a feature map from that model's *own* VAE decoder (``decode_features``).

The bridge speaks the *canonical* latent on both sides — it was trained against the
normalized Qwen-Image 2.1 latent — so there is no format conversion here.

Usage:
    upscaler = Qwen21TranscodeUpscaler(model.vae, device="cuda", dtype=torch.bfloat16)
    z_up = upscaler(z)   # canonical [B,64,H,W] -> canonical [B,64,2H,2W]
"""
from __future__ import annotations

import logging
from typing import Any, Union

import torch

from .base import LatentUpscaler
from .transcode_net import LatentTranscodeNet
from thenoise.inference import freeze
from thenoise.utils.safetensors import load_safetensors, upscale_weight_path

logger = logging.getLogger(__name__)

#: The committed bridge weights.
WEIGHTS = "upscaler_qwen21_transcode_2x.safetensors"


def _load_net(
    device: Union[str, torch.device], dtype: torch.dtype
) -> LatentTranscodeNet:
    """Load the transcode bridge from its committed weights (strictly)."""
    path = upscale_weight_path(WEIGHTS)
    logger.info("Loading Qwen-Image 2.1 latent transcoder from %s", path)
    state_dict = load_safetensors(path, device=device)

    net = LatentTranscodeNet()
    net.load_state_dict(state_dict, strict=True)
    net.to(device=device, dtype=dtype)
    freeze(net)

    logger.info("Latent transcoder ready on %s (%s)", device, dtype)
    return net


class Qwen21TranscodeUpscaler(LatentUpscaler):
    """Qwen-Image 2.1 latent upscale: canonical latent in, canonical latent out.

    Holds the adapter's *own* VAE instance — already resident, and the one thing
    guaranteed to be where the upscale runs. The bridge (~6 M parameters) is placed
    once at construction and stays put.
    """

    # Fixed by the network: both interpolate calls are 2x.
    scale = 2

    def __init__(
        self,
        vae: Any,
        *,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        """Wrap ``vae`` — the model's own — and load the bridge onto ``device``.

        The VAE must expose ``decode_features`` and have the bridge's latent
        width; both are checked here so a mis-wiring fails at construction.
        """
        if not callable(getattr(vae, "decode_features", None)):
            raise ValueError(
                f"this VAE ({type(vae).__name__}) does not expose decode_features(), "
                "so it cannot condition the latent transcoder"
            )

        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype
        self.net = _load_net(self.device, dtype)

        if vae.z_dim != self.net.latent_channels:
            raise ValueError(
                f"the VAE's {vae.z_dim}ch latent does not fit this bridge's "
                f"{self.net.latent_channels}ch input"
            )

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Upscale the canonical latent 2x, staying in canonical space."""
        z = latents.to(device=self.device, dtype=self.dtype)
        feature = self.vae.decode_features(z).to(device=self.device, dtype=self.dtype)
        return self.net(feature, z)


__all__ = ["Qwen21TranscodeUpscaler"]
