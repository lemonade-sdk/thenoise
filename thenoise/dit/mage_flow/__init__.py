"""Mage-Flow — the 12-layer member of the Qwen-Image dual-stream family.

The DiT (which reuses the Qwen-Image transformer block), its Qwen3-VL-4B
conditioner, its resolution-independent static-shift schedule, and the checkpoint
name helpers that tell it apart from its 60-layer sibling. The codec it denoises —
a one-step diffusion VAE, not a KL-VAE — lives in :mod:`thenoise.vae.mage_flow`.
"""
from .models import (
    MageFlowParams,
    MageFlowTransformer2DModel,
    MageTimestepProjEmbeddings,
    latent_to_tokens,
    load_mage_flow_dit,
    tokens_to_latent,
)

__all__ = [
    "MageFlowParams",
    "MageFlowTransformer2DModel",
    "MageTimestepProjEmbeddings",
    "latent_to_tokens",
    "load_mage_flow_dit",
    "tokens_to_latent",
]
