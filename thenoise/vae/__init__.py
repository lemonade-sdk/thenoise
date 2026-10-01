"""Model-specific VAE components.

Some VAEs are shared by more than one adapter and some are configured per adapter
(RGBA pixels, a latent scale), so each type is its own module, exported here for the
models to pick. Loading one is always ``load_<name>_vae(path, device, dtype)``; where a
model accepts more than one codec, ``load_<name>_family_vae`` reads the checkpoint header
and dispatches to the right one."""
from .qwen_image import (
    AutoencoderKLQwenImage,
    load_ming_vae,
    load_qwen_family_vae,
    load_qwen_vae,
)
from .flux import AutoencoderKLFlux, load_flux_vae
from .flux2 import AutoencoderKLFlux2, load_flux2_vae
from .mage_flow import AutoencoderKLMageFlow, load_mage_family_vae, load_mage_vae
from .wan22 import AutoencoderKLWan22, load_wan22_vae

__all__ = [
    "AutoencoderKLQwenImage",
    "load_ming_vae",
    "load_qwen_family_vae",
    "load_qwen_vae",
    "AutoencoderKLFlux",
    "load_flux_vae",
    "AutoencoderKLFlux2",
    "load_flux2_vae",
    "AutoencoderKLMageFlow",
    "load_mage_family_vae",
    "load_mage_vae",
    "AutoencoderKLWan22",
    "load_wan22_vae",
]
