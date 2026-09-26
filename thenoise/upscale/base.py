"""The latent-upscaler interface.

A ``LatentUpscaler`` owns *everything* that happens to a latent between the DiT
and the next DiT pass in the upscale-and-refine path: the format conversions the
upscaler network needs, the network itself, and the upscale factor. The pipeline
hands it the canonical latent the DiT produced and gets back a canonical latent
of the upscaled spatial size, ready to be fed back into the DiT for the refine
(and then to the VAE decode).

Keeping the latent math behind this one call is what lets different upscaler
strategies (Sesqui today, others to come) be swapped by the model adapter alone,
with no change in the pipeline.

The contract is deliberately narrow:

  * input and output are the *canonical* 4D latent ``[B, C, H, W]`` — the VAE's
    own latent format, exactly what ``DiffusionModel.init_latents`` produces and
    ``decode`` consumes;
  * the output is the input at ``scale`` times the spatial resolution, in the
    same (canonical) space, so the refine needs no conversion of its own.
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
