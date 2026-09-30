"""The Qwen-Image 2.1 latent-transcode upscaler strategy.

``Qwen21TranscodeUpscaler`` is the ``LatentUpscaler`` implementation built on the
vendored transcode bridge (``transcode_net.LatentTranscodeNet``): a small network
trained to turn the model's canonical latent into the same latent at 2x the
resolution, conditioned on a feature map from that model's *own* VAE decoder.

That dependency on the decoder is why this is its own strategy rather than another
entry in :data:`sesqui._UPSCALER_FORMATS`. A Sesqui format is a pure latent-space
transform: canonical latent in, raw VAE latent out through an adaptor, and the same
committed net for every model that shares the format. The bridge instead needs the
VAE instance itself — one decoder upsample stage of it, via
``AutoencoderKLWan22.decode_features`` — so it is built per model, from the model's
own ``self.vae``, and there is no adaptor layer to register.

The bridge speaks the *canonical* latent on both sides — it was trained against
theNoise's normalized Qwen-Image 2.1 latent — so there is no format conversion
here at all, only the decoder feature, which the VAE denormalises for us.

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

#: The committed bridge weights, named after the upstream release they came from.
WEIGHTS = "upscaler_qwen21_transcode_2x.safetensors"


def _load_net(
    device: Union[str, torch.device], dtype: torch.dtype
) -> LatentTranscodeNet:
    """Load the transcode bridge from its committed weights.

    The state dict is shipped as bf16 to match the engine's bf16-only convention.
    The load is ``strict``: the class in ``transcode_net`` is authoritative about
    the architecture, so a checkpoint that does not match it key for key is the
    wrong file, not a configuration to accommodate.
    """
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

    Holds the adapter's *own* VAE instance rather than a second one, for the reason
    ``VAEPixelUpscaler`` gives: the VAE is already registered with the model's
    ``MemoryManager`` and kept resident, while the DiT is the component being
    swapped, so it is the one thing guaranteed to be where the upscale runs. The
    bridge itself (~6 M parameters) is placed once at construction and stays put.

    The VAE feature extraction runs in the VAE's dtype and the bridge in this
    object's; in practice those are the model's dtype for both, and the handoff is
    made explicit so a mismatched pairing cannot silently upcast a whole feature map.
    """

    # Fixed by the network: both interpolate calls are 2x. Not a knob — the bridge
    # upscales 2x or not at all, and ``DiffusionModel.UPSCALE_SCALE`` mirrors it.
    scale = 2

    def __init__(
        self,
        vae: Any,
        *,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        """Wrap ``vae`` — the model's own — and load the bridge onto ``device``.

        The VAE has to be the Wan 2.2 family one: the bridge's latent width is the
        Qwen-Image 2.1 ``z_dim``, and the feature it conditions on comes from
        ``decode_features``, which only that VAE exposes. Both are checked here so a
        mis-wiring fails where the adapter named the VAE, not on first upscale.
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
        """Upscale the canonical latent 2x, staying in canonical space.

        Both halves want the same normalized latent the DiT consumes: the VAE half
        denormalises it into raw space to build the feature, the bridge takes it as
        the crude upsample to correct. The result is the latent the refine pass and
        the final decode already expect, so neither needs to know this ran.
        """
        z = latents.to(device=self.device, dtype=self.dtype)
        feature = self.vae.decode_features(z).to(device=self.device, dtype=self.dtype)
        return self.net(feature, z)


__all__ = ["Qwen21TranscodeUpscaler"]
