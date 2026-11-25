"""The H3 audio codec at a reduced vocoder width: the latent geometry the release
implies, Snake, the alias-free round trip, stereo lanes, decode shape and clamping, the
module tree the checkpoint's names are read into, and the loader dropping the encoder.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from conftest import write_safetensors
from thenoise.vae import AutoencoderKLH3Audio, load_h3_audio_vae
from thenoise.vae.h3_audio import (
    ENCODER_PREFIXES, ENCODER_RATES, LATENTS_PER_SECOND, RESBLOCK_DILATIONS,
    RESBLOCK_KERNELS, SAMPLE_RATE, SAMPLES_PER_LATENT, UPSAMPLE_KERNELS, UPSAMPLE_RATES,
    AMPBlock1, Activation1d, SnakeBeta, kaiser_sinc_filter1d, snake,
)

#: Narrow, but with the released rates: 800 samples per latent frame is what the latent
#: grid and the sample rate both promise, so a test vocoder that kept the rates cannot
#: escape them either.
TINY = dict(decoder_dim=256, upsample_rates=UPSAMPLE_RATES,
            upsample_kernel_sizes=UPSAMPLE_KERNELS,
            resblock_kernel_sizes=RESBLOCK_KERNELS,
            resblock_dilation_sizes=RESBLOCK_DILATIONS)
TINY_METADATA = {"minimax_h3_audio_vae": json.dumps(
    {"kwargs": {"decoder_dim": 256, "latent_dim": 2048,
                "decoder_rates": list(UPSAMPLE_RATES)}})}


def _fill(module: torch.nn.Module) -> torch.nn.Module:
    """Snake's alpha and beta are ``torch.empty`` until a checkpoint fills them, and they
    go through an ``exp``: unfilled, they are inf rather than a small wrong number."""
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, SnakeBeta):
                child.alpha.zero_()
                child.beta.zero_()
    return module


@pytest.fixture(scope="module")
def tiny_vae():
    torch.manual_seed(0)
    return _fill(AutoencoderKLH3Audio(**TINY)).eval().requires_grad_(False)


# --- the latent geometry ------------------------------------------------------


def test_a_latent_frame_is_800_samples_at_32_khz():
    assert SAMPLE_RATE == 32000
    assert SAMPLES_PER_LATENT == 800
    assert LATENTS_PER_SECOND == 40
    assert (AutoencoderKLH3Audio.samples_per_latent,
            AutoencoderKLH3Audio.latents_per_second) == (800, 40)


def test_the_vocoder_undoes_exactly_what_the_encoder_did():
    """The two rate lists are the same product by construction; the latent grid is only
    40 per second if they are."""
    assert math.prod(ENCODER_RATES) == math.prod(UPSAMPLE_RATES) == SAMPLES_PER_LATENT
    assert len(UPSAMPLE_RATES) == len(UPSAMPLE_KERNELS)
    assert len(RESBLOCK_KERNELS) == len(RESBLOCK_DILATIONS)


def test_decode_output_length_is_the_latent_count_in_samples(tiny_vae):
    assert tiny_vae.decode_output_length(6) == 6 * tiny_vae.decoder.upsample_factor
    for latent_t in (1, 3, 12):
        assert tiny_vae.decode_to_pcm(
            torch.randn(1, 32, 2, latent_t)).shape == (1, 2, tiny_vae.decode_output_length(latent_t))


# --- Snake and the alias-free round trip --------------------------------------


def test_snake_is_x_plus_a_periodic_term():
    x = torch.linspace(-2.0, 2.0, 17)
    alpha, beta = torch.tensor(1.3), torch.tensor(0.7)
    want = x + torch.sin(alpha * x) ** 2 / beta
    assert torch.allclose(snake(x.clone(), alpha, beta), want, atol=1e-6)


def test_snake_is_the_identity_when_alpha_is_zero():
    x = torch.randn(1, 4, 8)
    zero = torch.zeros(1, 4, 1)
    assert torch.allclose(snake(x.clone(), zero, zero + 1.0), x, atol=1e-6)


def test_snake_beta_stores_its_parameters_in_the_log_domain():
    act = SnakeBeta(4)
    with torch.no_grad():
        act.alpha.fill_(math.log(2.0))
        act.beta.fill_(math.log(0.5))
    x = torch.randn(1, 4, 8)
    want = snake(x.clone(), torch.tensor(2.0).view(1, -1, 1), torch.tensor(0.5).view(1, -1, 1))
    assert torch.allclose(act(x), want, atol=1e-5)


def test_the_kaiser_filter_is_a_normalized_lowpass():
    for kernel_size in (11, 12):
        f = kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=kernel_size)
        assert f.shape == (1, 1, kernel_size)
        assert torch.allclose(f.sum(), torch.ones(1), atol=1e-6)
    odd = kaiser_sinc_filter1d(0.25, 0.3, 11)
    assert torch.allclose(odd, odd.flip(-1), atol=1e-7), "odd kernels are symmetric"


def test_an_alias_activation_leaves_the_length_alone():
    act = Activation1d(SnakeBeta(6))
    x = torch.randn(1, 6, 64)
    with torch.no_grad():
        assert act(x).shape == x.shape


def test_an_anti_aliased_block_leaves_the_length_alone():
    block = _fill(AMPBlock1(6, kernel_size=3, dilation=(1, 3, 5))).eval()
    x = torch.randn(1, 6, 64)
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# --- decode -------------------------------------------------------------------


def test_decode_is_finite_and_clamped(tiny_vae):
    with torch.no_grad():
        pcm = tiny_vae.decode_to_pcm(torch.randn(1, 32, 2, 5))
    assert torch.isfinite(pcm).all()
    assert pcm.abs().max() <= 1.0


def test_the_stereo_lanes_are_independent(tiny_vae):
    """Folding the lanes into the batch is the whole stereo story, so one lane cannot
    hear the other."""
    latent = torch.randn(1, 32, 2, 5)
    other = latent.clone()
    other[:, :, 1] = torch.randn(1, 32, 5)
    silent = torch.cat([torch.zeros_like(latent[:, :, :1]), latent[:, :, 1:]], dim=2)
    with torch.no_grad():
        assert torch.allclose(tiny_vae.decode_to_pcm(latent)[:, 0],
                              tiny_vae.decode_to_pcm(other)[:, 0], atol=1e-5)
        assert torch.allclose(tiny_vae.decode_to_pcm(latent)[:, 1],
                              tiny_vae.decode_to_pcm(silent)[:, 1], atol=1e-5)


@pytest.mark.parametrize("shape", [(1, 32, 5), (1, 32, 3, 5), (1, 8, 2, 5)])
def test_decode_refuses_a_latent_that_is_not_audio(tiny_vae, shape):
    with pytest.raises(ValueError, match="audio expects|stereo lanes"):
        tiny_vae.decode_to_pcm(torch.zeros(shape))


def test_the_checkpoint_naming_survives_the_module_tree(tiny_vae):
    """The names below are the ones the released file uses. A module attribute renamed is
    a whole activation stack that fails to load, so they are pinned here."""
    names = set(tiny_vae.state_dict())
    for name in ("dec_in_proj.weight", "decoder.conv_pre.weight", "decoder.ups.0.0.weight",
                 "decoder.resblocks.0.convs1.0.weight", "decoder.resblocks.0.convs2.2.weight",
                 "decoder.resblocks.0.activations.0.act.alpha",
                 "decoder.resblocks.0.activations.0.act.beta",
                 "decoder.resblocks.0.activations.0.upsample.filter",
                 "decoder.resblocks.0.activations.0.downsample.lowpass.filter",
                 "decoder.activation_post.act.beta",
                 "decoder.activation_post.downsample.lowpass.filter",
                 "decoder.conv_post.weight", "latents_mean", "latents_std"):
        assert name in names, name
    assert not any(k.startswith(ENCODER_PREFIXES) for k in names)


# --- loader -------------------------------------------------------------------


def test_loader_reads_the_vocoder_and_drops_the_encoder(tmp_path, tiny_vae):
    tensors = dict(tiny_vae.state_dict())
    for i in range(3):
        tensors[f"encoder.block.{i}.weight"] = torch.zeros(4, 4, 1)
        tensors[f"pre_block.mlp.w{i}.weight"] = torch.zeros(4, 4)
    tensors["mean_proj.weight"] = torch.zeros(32, 32, 1)
    tensors["logs_proj.weight"] = torch.zeros(32, 32, 1)
    path = write_safetensors(tmp_path / "h3-audio-vae.safetensors", tensors,
                             metadata=TINY_METADATA)

    loaded = load_h3_audio_vae(path, device="cpu")

    assert loaded.dtype == AutoencoderKLH3Audio.compute_dtype == torch.float32
    assert not any(k.startswith(ENCODER_PREFIXES) for k in loaded.state_dict())
    assert loaded.decode_output_length(3) == tiny_vae.decode_output_length(3)


def test_loader_refuses_a_codec_whose_rates_disagree(tmp_path, tiny_vae, monkeypatch):
    path = write_safetensors(
        tmp_path / "audio.safetensors", dict(tiny_vae.state_dict()),
        metadata={"minimax_h3_audio_vae": json.dumps(
            {"kwargs": {"decoder_dim": 256, "latent_dim": 2048,
                        "decoder_rates": [3, 3, 2]}})})
    with pytest.raises(ValueError, match="rates"):
        load_h3_audio_vae(path, device="cpu")


def test_loader_refuses_a_video_codec(tmp_path):
    from thenoise.vae import AutoencoderKLH3Video

    torch.manual_seed(0)
    tensors = dict(AutoencoderKLH3Video(ch=32, dim=64, num_layers=1).state_dict())
    path = write_safetensors(tmp_path / "video.safetensors", tensors)
    with pytest.raises(ValueError, match="signature"):
        load_h3_audio_vae(path, device="cpu")
