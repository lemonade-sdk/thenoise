"""Model-specific VAE components.

The Qwen-Image VAE is shared across Anima/Krea2; the Flux VAE (decoder-only) is
used by Z-Image. Future VAE types should be added here as their own modules and
exported here so models can pick whichever they need.
"""
from .qwen_image import AutoencoderKLQwenImage, load_qwen_vae
from .flux import AutoencoderKLFlux, load_flux_vae
from .flux2 import AutoencoderKLFlux2, load_flux2_vae

__all__ = [
    "AutoencoderKLQwenImage",
    "load_qwen_vae",
    "AutoencoderKLFlux",
    "load_flux_vae",
    "AutoencoderKLFlux2",
    "load_flux2_vae",
]
