"""SDXL adapter tests (no real weights / no GPU needed).

Covers the discrete euler schedule (timestep indices + sigma grid), the model
defaults, size rounding, the vendored CLIP tokenizer, and the CLIP-G / VAE /
UNet construction with random weights (CPU).
"""
from __future__ import annotations

import pytest
import torch

from thenoise.dit.sdxl.sampling import (
    discrete_timesteps,
    get_alphas_cumprod,
    get_sigmas,
    sigma,
)
from thenoise.dit.sdxl.utils import SDXL_TOKENIZER_CONFIG_DIR
from thenoise.models import SdxlModel


def test_schedule_discrete_timesteps_descending():
    ts = discrete_timesteps(28)
    assert len(ts) == 28
    assert ts[0] == 999  # noise -> clean
    assert ts[-1] == 0
    assert ts == sorted(ts, reverse=True)


def test_schedule_sigmas_descending_with_trailing_zero():
    sigmas = get_sigmas(28)
    assert len(sigmas) == 29
    assert sigmas[-1] == 0.0
    assert all(s > n for s, n in zip(sigmas[:-1], sigmas[1:]))
    # largest sigma at t=999, smallest (t=0) just above 0.
    assert sigma(0) > 0.0
    assert sigma(999) > sigma(0)


def test_alphas_cumprod_bounds():
    abar = get_alphas_cumprod()
    assert abar.shape == (1000,)
    assert 0.0 < abar[0] < 1.0  # nearly clean
    assert abar[-1] < 0.05       # heavily noised


def test_zsnr_rescale_shifts_terminal_to_zero():
    from thenoise.dit.sdxl.sampling import rescale_zero_terminal_snr_alphas_cumprod

    abar = get_alphas_cumprod()
    z = rescale_zero_terminal_snr_alphas_cumprod(abar)
    assert z.shape == (1000,)
    # terminal timestep shifted toward zero-SNR (much smaller abar[-1]).
    assert z[-1] < 1e-4 < abar[-1]
    # first timestep kept near the original clean value.
    assert abs(z[0] - abar[0]) < 1e-4
    # still a valid descending-abar grid.
    assert (z[:-1] > z[1:]).all()


def test_zsnr_schedule_differs_from_plain():
    from thenoise.dit.sdxl.sampling import rescale_zero_terminal_snr_alphas_cumprod

    abar = get_alphas_cumprod()
    z = rescale_zero_terminal_snr_alphas_cumprod(abar)

    def sig(abar, t):
        return ((1 - abar[t]) / abar[t]) ** 0.5

    # sigma at the noisiest step is much larger under zsnr.
    assert sig(z, 999) > sig(abar, 999)


def test_sdxl_defaults():
    assert SdxlModel.DEFAULT_STEPS == 28
    assert SdxlModel.DEFAULT_GUIDANCE_SCALE == 5.5
    assert SdxlModel.SAMPLER == "euler"
    assert SdxlModel.LATENT_CHANNELS == 4
    assert SdxlModel._VAE_SCALE == 8


def test_resolve_size_rounds_to_multiple_of_8():
    m = SdxlModel.__new__(SdxlModel)
    assert m.resolve_size(1000, 1000) == (1000, 1000)
    assert m.resolve_size(1000, 1002) == (1000, 1008)
    assert m.resolve_size(513, 511) == (520, 512)


def test_vendored_tokenizer_config_dir_exists():
    from pathlib import Path

    d = Path(SDXL_TOKENIZER_CONFIG_DIR)
    assert d.is_dir()
    for required in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        assert (d / required).is_file(), f"missing vendored tokenizer file {required}"


def test_find_sdxl_tokenizer_dir(tmp_path):
    from thenoise.dit.sdxl.utils import find_sdxl_tokenizer_dir

    # Downloader layout: <out>/tokenizer/ + <out>/split_files/text_encoders/file.safetensors
    out = tmp_path / "models"
    (out / "tokenizer").mkdir(parents=True)
    te = out / "split_files" / "text_encoders" / "clip_l_g.safetensors"
    assert find_sdxl_tokenizer_dir(str(te)) == str(out / "tokenizer")


def test_find_sdxl_tokenizer_dir_returns_none_without_tokenizer(tmp_path):
    from thenoise.dit.sdxl.utils import find_sdxl_tokenizer_dir

    te = tmp_path / "split_files" / "text_encoders" / "clip_l_g.safetensors"
    assert find_sdxl_tokenizer_dir(str(te)) is None


def test_tokenizer_loads_offline():
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(SDXL_TOKENIZER_CONFIG_DIR, local_files_only=True)
    ids = tok("a cat", padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    assert ids.input_ids.shape == (1, 77)


def test_vae_decodes_latent_to_pixels():
    from thenoise.dit.sdxl.vae import AutoencoderKLSdxl

    vae = AutoencoderKLSdxl()
    latents = torch.randn(1, 4, 8, 8)
    pixels = vae.decode_to_pixels(latents)
    assert pixels.shape == (1, 3, 64, 64)
    assert (-1.0 <= pixels).all() and (pixels <= 1.0).all()


def test_size_embedding_dim():
    # 6 timestep_embeddings of 256 -> 1536, concatenated after pooled (1280) = 2816.
    from thenoise.dit.sdxl.models import timestep_embedding

    t = torch.tensor([1024.0])
    emb = timestep_embedding(t, 256)
    assert emb.shape == (1, 256)
    assert emb.dtype == torch.float32


def test_sigma_at_matches_sampling_grid():
    # ``_sigma_at`` (used for the UNet input scaling) must agree with the
    # standalone ``sigma(t)`` over the whole discrete grid.
    from thenoise.models.sdxl import SdxlModel

    m = object.__new__(SdxlModel)
    m._alphas_cumprod = get_alphas_cumprod()

    t = torch.arange(0, 1000, 37, dtype=torch.int64)
    s_at = m._sigma_at(t)
    assert s_at.shape == t.shape
    for i, ti in enumerate(t.tolist()):
        assert abs(float(s_at[i]) - sigma(ti)) < 1e-6
    # noisy end has the largest sigma; clean end approaches 0.
    assert float(s_at[0]) < float(s_at[-1])


def test_denoise_input_scaling_factor():
    # ComfyUI EPS ``calculate_input`` scales the model input by 1/sqrt(sigma^2+1).
    # At the noisiest step this is ~1/sigma_max (~0.04), so feeding the raw
    # latent (as the old code did) was ~26x too large and collapsed to gray.
    from thenoise.models.sdxl import SdxlModel

    m = object.__new__(SdxlModel)
    m._alphas_cumprod = get_alphas_cumprod()

    t = torch.tensor([999])
    sigma_hat = m._sigma_at(t)
    factor = 1.0 / torch.sqrt(sigma_hat**2 + 1)
    assert 0.0 < float(factor) < 1.0
    # the noisiest step's factor is well below 1 (input must be scaled down).
    assert float(factor) < 0.1


def test_prediction_type_epsilon_by_default():
    assert SdxlModel.prediction_type_from_keys(["input_blocks.0.0.weight"]) == "epsilon"


def test_prediction_type_v_prediction_marker():
    keys = ["input_blocks.0.0.weight", "v_pred"]
    assert SdxlModel.prediction_type_from_keys(keys) == "v_prediction"


def test_prediction_type_v_prediction_wrapped_marker():
    # The marker survives repackaging under the generic wrapper prefix.
    keys = ["input_blocks.0.0.weight", "model.diffusion_model.v_pred"]
    assert SdxlModel.prediction_type_from_keys(keys) == "v_prediction"


def test_prediction_type_edm_raises():
    with pytest.raises(NotImplementedError):
        SdxlModel.prediction_type_from_keys(["edm_mean", "edm_std"])


def test_prediction_type_vpred_edm_raises():
    with pytest.raises(NotImplementedError):
        SdxlModel.prediction_type_from_keys(["edm_vpred.sigma_max"])


def test_prediction_type_zsnr_is_v_prediction():
    # A zsnr checkpoint is still v-prediction; zsnr only changes the schedule.
    keys = ["input_blocks.0.0.weight", "v_pred", "ztsnr"]
    assert SdxlModel.prediction_type_from_keys(keys) == "v_prediction"
    assert SdxlModel.zsnr_from_keys(keys) is True


def test_zsnr_from_keys_absent():
    assert SdxlModel.zsnr_from_keys(["input_blocks.0.0.weight", "v_pred"]) is False


class _FakeDit:
    """Minimal UNet stub returning a constant tensor for denoise_step tests."""

    def __call__(self, scaled, t, y, context):
        return torch.full_like(scaled, 0.5)


def _make_model(prediction_type):
    m = object.__new__(SdxlModel)
    m.device = "cpu"
    m.dtype = torch.bfloat16
    m.dit = _FakeDit()
    m.prediction_type = prediction_type
    m._y = None
    m._y_uncond = None
    m._alphas_cumprod = get_alphas_cumprod(device="cpu")
    return m


def _cond():
    from thenoise.models.base import Conditioning

    return Conditioning(
        cond=torch.randn(1, 77, 2048, dtype=torch.bfloat16),
        pooled=torch.randn(1, 1280, dtype=torch.bfloat16),
        null=None,
        neg_pooled=None,
    )


def test_denoise_epsilon_returns_unet_output():
    m = _make_model("epsilon")
    latents = torch.randn(1, 4, 8, 8, dtype=torch.bfloat16)
    t = torch.tensor(500.0)
    v = m.denoise_step(latents, t, _cond(), guidance_scale=1.0, i=0)
    # epsilon: the velocity is just the UNet output (constant 0.5).
    assert torch.allclose(v, torch.full_like(v, 0.5), atol=1e-2)


def test_denoise_v_prediction_velocity():
    m = _make_model("v_prediction")
    latents = torch.randn(1, 4, 8, 8, dtype=torch.bfloat16)
    t = torch.tensor(500.0)
    v = m.denoise_step(latents, t, _cond(), guidance_scale=1.0, i=0)
    sigma = m._sigma_at(t).to(torch.bfloat16)
    expected = latents * sigma / (sigma**2 + 1) + torch.full_like(latents, 0.5) / torch.sqrt(sigma**2 + 1)
    assert torch.allclose(v, expected, atol=1e-2)


def test_combined_checkpoint_partitions(tmp_path):
    """A single combined SDXL checkpoint is partitioned into UNet/VAE/CLIP."""
    from safetensors.torch import save_file

    from thenoise.dit.sdxl.checkpoint import SDXLCheckpoint

    sd = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1),
        "first_stage_model.decoder.conv_in.weight": torch.zeros(1),
        "first_stage_model.post_quant_conv.weight": torch.zeros(1),
        "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight": torch.zeros(1),
        "conditioner.embedders.1.model.token_embedding.weight": torch.zeros(1),
        "v_pred": torch.zeros(1),
    }
    p = tmp_path / "mix.safetensors"
    save_file(sd, str(p))

    ckpt = SDXLCheckpoint(str(p), device="cpu")
    assert "v_pred" in ckpt.keys
    assert isinstance(ckpt.metadata, dict)

    unet, vae, clip_l, clip_g = ckpt._partition(sd)
    assert "input_blocks.0.0.weight" in unet
    assert "v_pred" in unet  # prediction marker preserved into the UNet partition
    assert "decoder.conv_in.weight" in vae
    assert "text_model.encoder.layers.0.layer_norm1.weight" in clip_l
    assert "token_embedding.weight" in clip_g

    # prediction type autodetected from the combined keys.
    assert SdxlModel.prediction_type_from_keys(ckpt.keys) == "v_prediction"


def test_combined_checkpoint_requires_all_parts(tmp_path):
    from safetensors.torch import save_file

    from thenoise.dit.sdxl.checkpoint import SDXLCheckpoint

    p = tmp_path / "partial.safetensors"
    save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1)}, str(p))
    ckpt = SDXLCheckpoint(str(p), device="cpu")
    with pytest.raises(ValueError):
        ckpt._partition(
            {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1)}
        )


def test_sdxl_upscale_format_registered():
    from thenoise.upscale import _UPSCALER_FORMATS

    assert "sdxl" in _UPSCALER_FORMATS
    assert _UPSCALER_FORMATS["sdxl"][1] == "upscaler_SDXL.safetensors"
    assert _UPSCALER_FORMATS["sdxl"][2] == 4
    # And SDXL now supports latent (refined) upscaling.
    m = object.__new__(SdxlModel)
    assert m._upscale_format() == "sdxl"
    assert m.supports_latent_upscale() is True


def test_sdxl_upscaler_loads_and_runs():
    import torch

    from thenoise.upscale import load_latent_upscaler

    model, adaptor = load_latent_upscaler("sdxl", device="cpu", dtype=torch.bfloat16)
    # SDXL canonical latent (raw * 0.13025) -> raw VAE latent -> 2x upscale -> back.
    z = torch.randn(1, 4, 64, 64)
    raw = adaptor.to_vae_latent(z).to(torch.bfloat16)
    out = model(raw, (128, 128))
    z_up = adaptor.from_vae_latent(out.float())
    assert z_up.shape == (1, 4, 128, 128)
    assert adaptor.external_channels == 4


def test_sdxl_upscaler_adaptor_is_identity():
    # SDXL pipeline latents are already scaled (raw * 0.13025) = Sesqui's trained
    # space, so the adaptor must be identity. Regression: the inverted affine
    # adaptor fed raw latents (~7.68x too large), producing a red/hue-shifted
    # decode.
    import torch

    from thenoise.upscale import load_latent_upscaler

    _, adaptor = load_latent_upscaler("sdxl", device="cpu", dtype=torch.bfloat16)
    z = torch.randn(1, 4, 16, 16)
    raw = adaptor.to_vae_latent(z)
    # to_vae must be a pass-through (no scaling).
    assert torch.allclose(raw.float(), z.float(), atol=1e-6)
    assert torch.allclose(
        adaptor.from_vae_latent(raw).float(), z.float(), atol=1e-6
    )
