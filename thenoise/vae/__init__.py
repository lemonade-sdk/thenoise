"""Model-specific VAE components.

The Qwen-Image VAE is shared across the current models; future VAE types should
be added here as their own modules and exported here so models can pick whichever 
they need.
"""
from .qwen_image import AutoencoderKLQwenImage, load_qwen_vae

__all__ = ["AutoencoderKLQwenImage", "load_qwen_vae"]
