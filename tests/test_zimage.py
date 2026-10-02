"""Z-Image adapter tests (no real weights / no GPU needed).

The static step schedule, the single-file text-encoder validation, tokenizer-directory
discovery and the Flux latent-format wiring. Model defaults, detection, the schedule
contract and the upscale-format registration are covered catalog-wide in
``test_catalog.py`` / ``test_detect.py`` / ``test_schedules.py``.
"""
from __future__ import annotations

import pytest
import torch

from thenoise.dit.zimage.sampling import get_sigmas
from thenoise.dit.zimage.utils import load_zimage_text_encoder
from thenoise.utils.text_encoder import find_tokenizer_dir


def test_zimage_sigmas_are_a_static_shifted_grid_with_trailing_zero():
    """Static ``shift=3`` over ``linspace(1, 1/N, N)`` (no resolution dependence):
    ``sigma = 3*s / (1 + 2*s)``, evaluated here for s = 1 ... 1/8.
    """
    sigmas = get_sigmas(8, torch.device("cpu"))

    assert sigmas.tolist() == pytest.approx(
        [1.0, 0.9545455, 0.9, 0.8333333, 0.75, 0.6428571, 0.5, 0.3, 0.0], rel=1e-6
    )


def test_text_encoder_rejects_non_safetensors(tmp_path):
    (tmp_path / "text_encoder").mkdir()
    with pytest.raises(ValueError, match=".safetensors"):
        load_zimage_text_encoder(str(tmp_path / "text_encoder"), device="cpu", dtype=torch.bfloat16)


def test_find_tokenizer_dir(tmp_path):
    # Downloader layout: <out>/tokenizer/ + <out>/split_files/text_encoders/file.safetensors
    out = tmp_path / "models"
    (out / "tokenizer").mkdir(parents=True)
    te = out / "split_files" / "text_encoders" / "qwen_3_4b.safetensors"
    assert find_tokenizer_dir(str(te)) == str(out / "tokenizer")

    # Same layout without the tokenizer directory -> None (the caller fetches).
    (out / "tokenizer").rmdir()
    assert find_tokenizer_dir(str(te)) is None


def test_zimage_upscale_format_is_flux():
    """Z-Image uses the Flux VAE, so the affine latent format must carry that VAE's
    own shift/scale constants.
    """
    from thenoise.upscale import make_flux
    from thenoise.vae import AutoencoderKLFlux

    adaptor = make_flux()
    assert adaptor.scale == AutoencoderKLFlux.scaling_factor
    assert adaptor.shift == AutoencoderKLFlux.shift_factor
