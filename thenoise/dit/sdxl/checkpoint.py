# Single-file SDXL/Illustrious checkpoint loading (no pre-splitting needed).
#
# A single-file Civitai SDXL/Illustrious mix ships the UNet, VAE, and both CLIP
# text encoders concatenated under the classic Stability-AI prefixes, plus
# prediction-type marker tensors (``v_pred``, ``edm_mean``/``edm_std``, ...) and
# a ``__metadata__`` dict. This reads such a file once, partitions it in memory,
# and builds the three thenoise components directly, so ``--checkpoint`` can
# point at a single download without running the splitter.

import logging
from typing import Optional, Union

import torch

from thenoise.dit.sdxl.utils import (
    PREDICTION_MARKERS,
    build_sdxl_dit,
    build_sdxl_text_encoders,
)
from thenoise.dit.sdxl.vae import build_sdxl_vae
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, load_safetensors

logger = logging.getLogger(__name__)

#: Component key prefixes within a combined SDXL checkpoint (Stability-AI layout).
UNET_PREFIX = "model.diffusion_model."
VAE_PREFIX = "first_stage_model."
CLIP_L_PREFIX = "conditioner.embedders.0.transformer."
CLIP_G_PREFIX = "conditioner.embedders.1.model."


class SDXLCheckpoint:
    """Read a combined SDXL checkpoint and load its three components in memory.

    Reads the safetensors header once (keys + ``__metadata__``) for prediction
    detection, then loads the full tensor data once and partitions it into the
    UNet, VAE, and CLIP-L + CLIP-G text encoders.
    """

    def __init__(
        self,
        path: str,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.path = path
        self.device = torch.device(device)
        self.dtype = dtype
        with MemoryEfficientSafeOpen(path) as f:
            self.keys = list(f.keys())
            self.metadata: dict = f.metadata()

    def load_components(self):
        """Load ``(dit, clip_l, clip_g, vae)`` from the combined checkpoint."""
        logger.info("Loading combined SDXL checkpoint from %s", self.path)
        sd = load_safetensors(
            self.path, device=str(self.device), disable_mmap=True, dtype=self.dtype
        )
        unet_sd, vae_sd, clip_l_sd, clip_g_sd = self._partition(sd)
        del sd

        dit = build_sdxl_dit(unet_sd, self.device, self.dtype)
        clip_l, clip_g = build_sdxl_text_encoders(
            clip_l_sd, clip_g_sd, self.device, self.dtype
        )
        vae = build_sdxl_vae(vae_sd, self.device)
        return dit, clip_l, clip_g, vae

    def _partition(self, sd: dict) -> tuple[dict, dict, dict, dict]:
        """Split the combined state dict into (unet, vae, clip_l, clip_g)."""
        unet = {k[len(UNET_PREFIX):]: v for k, v in sd.items() if k.startswith(UNET_PREFIX)}
        unet.update({k: v for k, v in sd.items() if k in PREDICTION_MARKERS})
        vae = {
            k[len(VAE_PREFIX):]: v
            for k, v in sd.items()
            if k.startswith(VAE_PREFIX)
            and (
                k.startswith(VAE_PREFIX + "decoder.")
                or "post_quant_conv" in k
            )
        }
        clip_l = {k[len(CLIP_L_PREFIX):]: v for k, v in sd.items() if k.startswith(CLIP_L_PREFIX)}
        clip_g = {k[len(CLIP_G_PREFIX):]: v for k, v in sd.items() if k.startswith(CLIP_G_PREFIX)}
        for name, part in [("unet", unet), ("vae", vae), ("clip_l", clip_l), ("clip_g", clip_g)]:
            if not part:
                raise ValueError(
                    f"partition {name!r} is empty: {self.path} may not be an "
                    "SDXL/Illustrious combined checkpoint"
                )
        return unet, vae, clip_l, clip_g


__all__ = ["SDXLCheckpoint"]
