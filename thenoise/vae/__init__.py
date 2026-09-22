"""Model-specific VAE components.

The Qwen-Image VAE is shared across Anima/Krea2; the Flux VAE (decoder-only) is
used by Z-Image; the Flux.2 VAE by Flux Klein; and the Wan 2.2 family VAE
(2D still-image port) covers both the Wan 2.2 weights and the Qwen-Image 2.1
ones, which share its layout. Future VAE types should be added here as their own
modules and exported here so models can pick whichever they need.
"""
from .qwen_image import AutoencoderKLQwenImage, load_qwen_vae
from .flux import AutoencoderKLFlux, load_flux_vae
from .flux2 import AutoencoderKLFlux2, load_flux2_vae
from .wan22 import AutoencoderKLWan22, load_wan22_vae

__all__ = [
    "AutoencoderKLQwenImage",
    "load_qwen_vae",
    "AutoencoderKLFlux",
    "load_flux_vae",
    "AutoencoderKLFlux2",
    "load_flux2_vae",
    "AutoencoderKLWan22",
    "load_wan22_vae",
]
