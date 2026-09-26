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
from thenoise.samplers import SAMPLERS, create_sampler
from thenoise.upscale import (
    SesquiLSRUpscaler,
    _UPSCALER_FORMATS,
)

# Per-model public defaults (the values the API/CLI fall back to).
MODEL_DEFAULTS = {
    "anima": {"steps": 8, "guidance": 1, "sampler": "er_sde", "kv_cache": False},
    "krea2": {"steps": 8, "guidance": 1.0, "sampler": "er_sde", "kv_cache": False},
    "zimage": {"steps": 8, "guidance": 1.0, "sampler": "euler", "kv_cache": False},
    "flux_klein": {"steps": 4, "guidance": 1.0, "sampler": "euler", "kv_cache": False},
    "qwen_image": {"steps": 28, "guidance": 2.5, "sampler": "euler", "kv_cache": False},
    "qwen_image21": {"steps": 28, "guidance": 1.0, "sampler": "euler", "kv_cache": True},
}


@pytest.mark.parametrize("model", MODEL_CATALOG, ids=CATALOG_IDS)
def test_model_defaults(model):
    """Every adapter ships the documented defaults and a usable sampler name."""
    expected = MODEL_DEFAULTS[model.name]
    prefs = model.DEFAULT_PREFS
    assert prefs["steps"] == expected["steps"]
    assert prefs["guidance_scale"] == expected["guidance"]
    assert prefs["sampler"] == expected["sampler"]
    assert prefs["kv_cache"] is expected["kv_cache"]
    # A typo'd sampler would only blow up at request time; tie it to the registry.
    assert prefs["sampler"] in SAMPLERS
    assert create_sampler(prefs["sampler"], model) is not None


@pytest.mark.parametrize("model", MODEL_CATALOG, ids=CATALOG_IDS)
def test_model_defaults_extend_the_base_preferences(model):
    """An adapter's override must keep every preference the base declares.

    Defaults are merged (``{**DiffusionModel.DEFAULT_PREFS, ...}``); dropping a key
    would make the pipeline raise ``KeyError`` on that preference at request time.
    """
    assert set(DiffusionModel.DEFAULT_PREFS) <= set(model.DEFAULT_PREFS)


@pytest.fixture(scope="module")
def latent_upscalers():
    """Each registered latent format loaded once (a few MB of committed weights)."""
    return {
        fmt: SesquiLSRUpscaler(fmt, device="cpu", dtype=torch.bfloat16)
        for fmt in _UPSCALER_FORMATS
    }


@pytest.mark.parametrize("fmt", sorted(_UPSCALER_FORMATS))
def test_latent_upscaler_matches_its_format_registry(fmt, latent_upscalers):
    """Registry channels/adaptor agree with the shipped weights and round-trip.

    ``SesquiLSRUpscaler`` converts the canonical latent to raw VAE space, upscales,
    and converts back — so ``to_vae_latent`` must land on the registry's raw channel
    count, and one call on a canonical latent must hand back a canonical latent at
    the requested factor (whatever the raw space's own channel count and spatial
    scale are).
    """
    upscaler = latent_upscalers[fmt]
    _factory, filename, channels = _UPSCALER_FORMATS[fmt]
    adaptor = upscaler.adaptor

    z = torch.randn(1, adaptor.external_channels, 4, 4)
    raw = adaptor.to_vae_latent(z).to(torch.bfloat16)
    assert raw.shape[1] == channels, f"{fmt} ({filename}) carries {raw.shape[1]}ch"

    # The whole transform, in canonical coords in and out.
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
