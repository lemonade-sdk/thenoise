"""The model catalog's shared contract: defaults, geometry and upscale formats.

One table-driven test per contract, over every registered adapter, so a new model
is covered by being added to ``MODEL_CATALOG`` rather than by writing yet another
per-model test file.
"""
from __future__ import annotations

import pytest
import torch

from conftest import CATALOG_IDS
from thenoise.models import MODEL_CATALOG
from thenoise.samplers import SAMPLERS, create_sampler
from thenoise.upscale import _UPSCALER_FORMATS, load_latent_upscaler, upscale_weight_path

# Per-model public defaults (the values the API/CLI fall back to).
MODEL_DEFAULTS = {
    "anima": {"steps": 8, "guidance": 1, "sampler": "er_sde", "channels": 16},
    "krea2": {"steps": 8, "guidance": 1.0, "sampler": "er_sde", "channels": 16},
    "zimage": {"steps": 8, "guidance": 1.0, "sampler": "euler", "channels": 16},
    # Distilled: 4 steps, guidance 1.0 (CFG off), Euler, packed 128ch latent.
    "flux_klein": {"steps": 4, "guidance": 1.0, "sampler": "euler", "channels": 128},
    # Qwen-Image: 50 steps, guidance 1.0 (CFG off), Euler, packed 16ch latent.
    "qwen_image": {"steps": 50, "guidance": 1.0, "sampler": "euler", "channels": 16},
}


@pytest.mark.parametrize("model", MODEL_CATALOG, ids=CATALOG_IDS)
def test_model_defaults(model):
    """Every adapter ships the documented defaults and a usable sampler name."""
    expected = MODEL_DEFAULTS[model.name]
    assert model.DEFAULT_STEPS == expected["steps"]
    assert model.DEFAULT_GUIDANCE_SCALE == expected["guidance"]
    assert model.SAMPLER == expected["sampler"]
    assert model.LATENT_CHANNELS == expected["channels"]
    # A typo'd SAMPLER would only blow up at request time; tie it to the registry.
    assert model.SAMPLER in SAMPLERS
    assert create_sampler(model.SAMPLER, model) is not None


@pytest.mark.parametrize("model", MODEL_CATALOG, ids=CATALOG_IDS)
def test_model_upscale_format_is_registered_with_weights(model):
    """Each adapter names a latent format that has a committed upscaler."""
    instance = object.__new__(model)  # the format is a class constant, no weights
    fmt = instance._upscale_format()
    assert fmt in _UPSCALER_FORMATS, f"{model.name} names unknown format {fmt!r}"
    _factory, filename, channels = _UPSCALER_FORMATS[fmt]
    assert upscale_weight_path(filename).is_file()
    assert channels > 0


@pytest.fixture(scope="module")
def latent_upscalers():
    """Each registered latent format loaded once (a few MB of committed weights)."""
    return {
        fmt: load_latent_upscaler(fmt, device="cpu", dtype=torch.bfloat16)
        for fmt in _UPSCALER_FORMATS
    }


@pytest.mark.parametrize("fmt", sorted(_UPSCALER_FORMATS))
def test_latent_upscaler_matches_its_format_registry(fmt, latent_upscalers):
    """Registry channels/adaptor agree with the shipped weights and round-trip.

    ``PipelineController._upscale_and_refine`` converts the canonical latent to
    raw VAE space, upscales, and converts back — so ``to_vae_latent`` must land on
    the registry's raw channel count and ``from_vae_latent`` must be its inverse.
    """
    model, adaptor = latent_upscalers[fmt]
    _factory, filename, channels = _UPSCALER_FORMATS[fmt]

    z = torch.randn(1, adaptor.external_channels, 4, 4)
    raw = adaptor.to_vae_latent(z).to(torch.bfloat16)
    assert raw.shape[1] == channels, f"{fmt} ({filename}) carries {channels}ch"

    # The pipeline upscales by ``DiffusionModel.UPSCALE_SCALE`` (2) in *external*
    # coords and hands the upscaler a target converted into VAE coords.
    target = adaptor.vae_target_size((2 * 4, 2 * 4))
    out = model(raw, target)
    z_up = adaptor.from_vae_latent(out.float())
    assert z_up.shape == (1, adaptor.external_channels, 8, 8)

    # An identity-size pass through the adaptor pair must be lossless.
    identity = adaptor.from_vae_latent(adaptor.to_vae_latent(z).float())
    assert torch.allclose(identity, z, atol=1e-4)
