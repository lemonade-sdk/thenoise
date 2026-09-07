"""Z-Image adapter tests (no real weights / no GPU needed).

Covers the sigma-based step schedule, the single-file text-encoder validation,
tokenizer-directory discovery and the Flux (Z-Image) latent upscaler. Model
defaults and the upscale-format registration are covered catalog-wide in
``test_catalog.py``.
"""
from __future__ import annotations

import torch

from thenoise.dit.zimage.sampling import get_sigmas
from thenoise.dit.zimage.utils import (
    load_zimage_text_encoder,
)
from thenoise.utils.text_encoder import QWEN25_TOKENIZER_CONFIG_DIR, find_tokenizer_dir


def test_zimage_sigmas_are_static_shifted_grid_with_trailing_zero():
    sigmas = get_sigmas(8, torch.device("cpu"))
    assert sigmas[-1] == 0.0
    # First sigma is exactly 1.0 (linspace(1, 1/8, 8) shifted with shift=3.0 -> 1.0).
    assert sigmas[0] == 1.0
    # Strictly decreasing.
    assert torch.all(sigmas[:-1] > sigmas[1:])
    # Matches the static flow shift formula: sigma = shift*s/(1 + (shift-1)*s).
    s = torch.linspace(1.0, 1.0 / 8, 8)
    expected = 3.0 * s / (1.0 + 2.0 * s)
    assert torch.allclose(sigmas[:-1], expected)


def test_text_encoder_rejects_non_safetensors(tmp_path):
    p = tmp_path / "text_encoder"
    p.mkdir()
    try:
        load_zimage_text_encoder(str(p), device="cpu", dtype=torch.bfloat16)
    except ValueError as e:
        assert ".safetensors" in str(e)
    else:
        raise AssertionError("expected ValueError for a non-.safetensors path")


def test_find_tokenizer_dir(tmp_path):
    # Downloader layout: <out>/tokenizer/ + <out>/split_files/text_encoders/file.safetensors
    out = tmp_path / "models"
    (out / "tokenizer").mkdir(parents=True)
    te = out / "split_files" / "text_encoders" / "qwen_3_4b.safetensors"
    found = find_tokenizer_dir(str(te))
    assert found == str(out / "tokenizer")


def test_find_tokenizer_dir_returns_none_without_tokenizer(tmp_path):
    te = tmp_path / "split_files" / "text_encoders" / "qwen_3_4b.safetensors"
    assert find_tokenizer_dir(str(te)) is None


def test_vendored_tokenizer_config_dir_exists():
    # The tokenizer config files are checked into the package so the tokenizer loads
    # offline without fetching from the Hub (mirrors the anima configs/ pattern).
    from pathlib import Path

    d = Path(QWEN25_TOKENIZER_CONFIG_DIR)
    assert d.is_dir()
    for required in ("tokenizer.json", "tokenizer_config.json"):
        assert (d / required).is_file(), f"missing vendored tokenizer file {required}"


def test_zimage_upscale_format_is_flux():
    # Z-Image uses the Flux VAE -> the affine shift/scale latent format, whose
    # constants must match the VAE's own decode normalization.
    from thenoise.upscale import make_flux
    from thenoise.vae import AutoencoderKLFlux

    adaptor = make_flux()
    assert adaptor.scale == AutoencoderKLFlux.scaling_factor
    assert adaptor.shift == AutoencoderKLFlux.shift_factor


def test_flux_upscaler_loads_and_runs():
    from thenoise.upscale import load_latent_upscaler

    model, adaptor = load_latent_upscaler("flux", device="cpu", dtype=torch.bfloat16)
    # Canonical Z-Image (Flux) latent -> raw VAE latent -> 2x upscale -> back.
    z = torch.randn(1, 16, 8, 8)
    raw = adaptor.to_vae_latent(z).to(torch.bfloat16)
    out = model(raw, (16, 16))
    z_up = adaptor.from_vae_latent(out.float())
    assert z_up.shape == (1, 16, 16, 16)


def test_load_upscaler_rejects_unknown_format():
    import pytest

    from thenoise.upscale import load_latent_upscaler

    with pytest.raises(ValueError):
        load_latent_upscaler("not_a_real_format", device="cpu", dtype=torch.bfloat16)
