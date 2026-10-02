"""The model catalog's shared contract: defaults, geometry and latent upscalers.

One table-driven test per contract, over every registered adapter, so a new model
is covered by being added to ``MODEL_CATALOG`` rather than by writing yet another
per-model test file.
"""
from __future__ import annotations

import pytest
import torch

from conftest import CATALOG_IDS
from thenoise.models import MODEL_CATALOG
from thenoise.models.base import DiffusionModel
from thenoise.samplers import create_sampler
from thenoise.upscale import (
    SesquiLSRUpscaler,
    _UPSCALER_FORMATS,
)


@pytest.mark.parametrize("model", MODEL_CATALOG, ids=CATALOG_IDS)
def test_model_defaults_are_a_usable_superset_of_the_base(model):
    """A merge that drops a base preference KeyErrors at request time, and a typo'd
    sampler name only blows up when somebody generates: tie both to the machinery.
    """
    assert set(DiffusionModel.DEFAULT_PREFS) <= set(model.DEFAULT_PREFS)
    create_sampler(model.DEFAULT_PREFS["sampler"], model)


@pytest.fixture(scope="module")
def latent_upscalers():
    """Each registered latent format loaded once (a few MB of committed weights)."""
    return {
        fmt: SesquiLSRUpscaler(fmt, device="cpu", dtype=torch.bfloat16)
        for fmt in _UPSCALER_FORMATS
    }


@pytest.mark.parametrize("fmt", sorted(_UPSCALER_FORMATS))
def test_latent_upscaler_matches_its_format_registry(fmt, latent_upscalers):
    """Registry channels agree with the shipped weights, and the round trip works.

    ``SesquiLSRUpscaler`` converts the canonical latent to raw VAE space, upscales and
    converts back: ``to_vae_latent`` must land on the registry's raw channel count and
    one call must hand back a canonical latent at the requested factor.
    """
    upscaler = latent_upscalers[fmt]
    _factory, filename, channels = _UPSCALER_FORMATS[fmt]
    adaptor = upscaler.adaptor

    z = torch.randn(1, adaptor.external_channels, 4, 4)
    assert adaptor.to_vae_latent(z).to(torch.bfloat16).shape[1] == channels, \
        f"{fmt} ({filename}) carries the wrong raw channel count"

    z_up = upscaler(z)
    assert z_up.shape == (
        1,
        adaptor.external_channels,
        upscaler.scale * 4,
        upscaler.scale * 4,
    )

    # An identity-size pass through the adaptor pair must be lossless.
    identity = adaptor.from_vae_latent(adaptor.to_vae_latent(z).float())
    assert torch.allclose(identity, z, atol=1e-4)
