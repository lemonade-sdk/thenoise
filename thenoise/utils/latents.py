"""Shared latent (de)packing helpers used by the DiT adapters.

Qwen-Image packs a canonical ``[B, C, H, W]`` latent into the flattened
``[B, H//2*W//2, C*4]`` layout the transformer consumes (2x2 spatial merge).
The same packing/unpacking applies to any model using a 2x2 latent patch, so it
lives here rather than under a single adapter.
"""
from __future__ import annotations

import torch


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack canonical ``[B, C, H, W]`` (or ``[B, C, 1, H, W]``) -> ``[B, H//2*W//2, C*4]``."""
    batch_size = latents.shape[0]
    if latents.ndim == 4 or latents.shape[2] == 1:
        num_channels_latents = latents.shape[1]
        height = latents.shape[-2]
        width = latents.shape[-1]
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    num_layers = latents.shape[1]
    num_channels_latents = latents.shape[2]
    height = latents.shape[-2]
    width = latents.shape[-1]
    latents = latents.view(batch_size, num_layers, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
    return latents.reshape(batch_size, num_layers * (height // 2) * (width // 2), num_channels_latents * 4)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack ``[B, H//2*W//2, C*4]`` -> canonical ``[B, C, H, W]``."""
    batch_size = latents.shape[0]
    num_channels_latents = latents.shape[2] // 4
    height = height // 2
    width = width // 2
    latents = latents.reshape(batch_size, height, width, num_channels_latents, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, num_channels_latents, height * 2, width * 2)


__all__ = ["pack_latents", "unpack_latents"]
