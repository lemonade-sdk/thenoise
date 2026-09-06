"""Qwen-Image adapter tests (no real weights / no GPU needed).

Covers the flow schedule, the token pack/unpack helpers, the vendored
tokenizer config directory, the Qwen-Image latent format and the reference-latent
packing rejection. Detection and the per-model defaults live in the catalog-wide
tables of ``test_detect.py`` / ``test_catalog.py``.
"""
from __future__ import annotations

import pytest
import torch

from conftest import write_safetensors
from thenoise.dit.qwen_image import sampling as qwen_sampling
from thenoise.dit.qwen_image import utils as qwen_utils
from thenoise.models.qwen_image import _detect_zero_cond_t, QwenImageModel
from thenoise.upscale import make_wan21
from thenoise.vae import AutoencoderKLQwenImage


# ---------------------------------------------------------------- flow schedule


def test_schedule_is_flow_grid_1_to_0():
    ts = qwen_sampling.get_schedule(8, 4096)  # 1024x1024 -> 64x64 packed -> 4096 tokens
    assert len(ts) == 9  # num_steps + 1
    assert ts[0] == 1.0
    assert ts[-1] == 0.0
    # Strictly decreasing.
    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1))


def test_schedule_depends_on_token_count():
    ts_small = qwen_sampling.get_schedule(8, 256)
    ts_large = qwen_sampling.get_schedule(8, 4096)
    # A larger image token count gets a larger empirical shift (steeper early steps).
    assert ts_small[1] < ts_large[1]


def test_single_step_schedule_is_finite():
    """``steps == 1`` is the degenerate case: the terminal stretch divides by zero.

    The guard in ``get_sigmas`` leaves the grid untouched so a 1-step denoise
    starts at exactly pure noise (t=1.0) instead of producing ``nan``.
    """
    sigmas = qwen_sampling.get_sigmas(1, 256, qwen_sampling.compute_mu(256))
    assert torch.isfinite(sigmas).all()
    assert sigmas[0] == 1.0


def test_calculate_shift_increases_with_token_count():
    assert qwen_utils.calculate_shift(256) < qwen_utils.calculate_shift(4096)


# ------------------------------------------------------------------ latents


def test_pack_unpack_latents_roundtrip():
    torch.manual_seed(0)
    latent = torch.randn(1, 16, 4, 4)
    packed = qwen_utils.pack_latents(latent)
    # 4x4 grid -> 2x2 patch blocks -> 4 tokens, 16ch * 4 = 64 features each.
    assert packed.shape == (1, 4, 64)
    back = qwen_utils.unpack_latents(packed, 4, 4)
    assert back.shape == (1, 16, 4, 4)
    assert torch.allclose(back, latent)


def test_pack_latents_accepts_frame_axis():
    """The edit path feeds a ``[B, C, 1, H, W]`` reference latent."""
    latent = torch.randn(1, 16, 1, 4, 4)
    packed = qwen_utils.pack_latents(latent)
    assert packed.shape == (1, 4, 64)


# ------------------------------------------------------------- vendored config


def test_vendored_tokenizer_config_dir_exists():
    # The tokenizer config files are checked into the package so the tokenizer and
    # processor load offline without fetching from the Hub (mirrors the anima/zimage
    # ``configs/`` pattern).
    from pathlib import Path

    d = Path(qwen_utils.TOKENIZER_CONFIG_DIR)
    assert d.is_dir()
    for required in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        assert (d / required).is_file(), f"missing vendored tokenizer file {required}"


# ---------------------------------------------------------------- latent format


def test_qwen_upscale_format_is_wan21():
    # Qwen-Image uses the shared Wan21 z-score latent format; the adaptor's
    # per-channel mean/std must match the VAE's own encode/decode normalization.
    adaptor = make_wan21()
    vae = AutoencoderKLQwenImage()
    assert torch.allclose(adaptor.mean.view(-1), torch.tensor(vae.latents_mean))
    assert torch.allclose(adaptor.std.view(-1), torch.tensor(vae.latents_std))


# ----------------------------------------------------------------- reference


def test_pack_reference_latent_rejects_unsupported_method():
    """An unsupported ``ref_latents_method`` raises rather than being ignored."""
    model = QwenImageModel.__new__(QwenImageModel)  # no __init__ (no weights)
    model.device = "cpu"
    model.dtype = torch.float32
    with pytest.raises(ValueError, match="unsupported ref_latents_method"):
        model.pack_reference_latent(torch.randn(1, 16, 4, 4), method="crop")


# ------------------------------------------------------------ zero_cond_t flag


def test_detect_zero_cond_t_reads_checkpoint_header(tmp_path):
    """The edit-2511 checkpoint carries ``__index_timestep_zero__``; older ones do not."""
    plain = tmp_path / "plain.safetensors"
    write_safetensors(plain, {"img_in.weight": torch.zeros(1), "txt_in.weight": torch.zeros(1)})
    assert _detect_zero_cond_t(str(plain)) is False

    zero_cond = tmp_path / "zero_cond.safetensors"
    write_safetensors(zero_cond, {"__index_timestep_zero__": torch.zeros(1)})
    assert _detect_zero_cond_t(str(zero_cond)) is True
