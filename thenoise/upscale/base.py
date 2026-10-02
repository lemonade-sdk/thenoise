"""The latent-upscaler interface.

A ``LatentUpscaler`` owns *everything* that happens to a latent between the DiT
and the next DiT pass in the upscale-and-refine path: the format conversions the
upscaler network needs, the network itself, and the upscale factor.

The contract: input and output are the *canonical* 4D latent ``[B, C, H, W]`` —
the VAE's own latent format — and the output is the input at ``scale`` times the
spatial resolution, in the same space.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class LatentUpscaler(ABC):
    """Latent-domain upscaler: canonical latent in, canonical latent out."""

    scale: int

    @abstractmethod
    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Upscale the canonical latent ``[B, C, H, W]`` to ``[B, C, s*H, s*W]``."""
        ...


__all__ = ["LatentUpscaler"]
