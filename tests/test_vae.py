"""VAE-level contract: each VAE states the canonical latent it produces.

The pipeline's canonical latent ``[B, C, H, W]`` *is* the VAE's output format, so
its geometry lives on the VAE (``z_dim``, ``spatial_compression``) rather than
being re-declared by the model adapters — a DiT's own input width is a different
number (it patchifies the latent further), and two declarations of one fact is how
they drift apart.
"""
from __future__ import annotations

import pytest

from thenoise.vae import AutoencoderKLFlux, AutoencoderKLFlux2, AutoencoderKLQwenImage


def test_class_level_geometry():
    """The weight-free VAEs state their latent geometry as class constants."""
    assert (AutoencoderKLFlux.z_dim, AutoencoderKLFlux.spatial_compression) == (16, 8)
    # Flux.2 packs the 32ch VAE latent 2x2 -> 128ch at 16 pixels per latent cell.
    assert (AutoencoderKLFlux2.z_dim, AutoencoderKLFlux2.spatial_compression) == (128, 16)


def test_qwen_geometry_follows_the_encoder_depth():
    """The shared Qwen-Image VAE derives compression from its downsampling stages."""
    vae = AutoencoderKLQwenImage(base_dim=4)  # tiny random init: no weights needed
    assert vae.z_dim == 16
    assert vae.spatial_compression == 8  # len(dim_mult) - 1 == 3 stages


@pytest.mark.parametrize(
    "dim_mult,compression",
    [([1, 2], 2), ([1, 2, 4], 4), ([1, 2, 4, 4], 8), ([1, 2, 4, 4, 4], 16)],
)
def test_qwen_compression_follows_the_downsampling_stages(dim_mult, compression):
    """One halving per stage past the first: ``2 ** (len(dim_mult) - 1)``."""
    vae = AutoencoderKLQwenImage(base_dim=4, dim_mult=dim_mult)
    assert vae.spatial_compression == compression
