"""The H3 audio codec at a reduced width: the latent geometry the release implies, Snake,
the alias-free round trip, stereo lanes, decode shape and clamping, encoding a sound back
to a latent, the module tree the checkpoint's names are read into, and both halves loading.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from conftest import write_safetensors
from thenoise.vae import AutoencoderKLH3Audio, load_h3_audio_vae
from thenoise.vae.h3_audio import (
    ENCODER_RATES, LATENTS_PER_SECOND, RESBLOCK_DILATIONS,
    RESBLOCK_KERNELS, SAMPLE_RATE, SAMPLES_PER_LATENT, UPSAMPLE_KERNELS, UPSAMPLE_RATES,
    AMPBlock1, Activation1d, AttnProjection, CausalAttention, Encoder, Snake1d, SnakeBeta,
    kaiser_sinc_filter1d, snake,
)

#: Narrow, but with the released rates: 800 samples per latent frame is what the latent
#: grid and the sample rate both promise, so a test vocoder that kept the rates cannot
#: escape them either.
TINY = dict(encoder_dim=2, decoder_dim=256, upsample_rates=UPSAMPLE_RATES,
            upsample_kernel_sizes=UPSAMPLE_KERNELS,
            resblock_kernel_sizes=RESBLOCK_KERNELS,
            resblock_dilation_sizes=RESBLOCK_DILATIONS)
TINY_METADATA = {"minimax_h3_audio_vae": json.dumps(
    {"kwargs": {"decoder_dim": 256, "latent_dim": 2048, "encoder_dim": 2,
                "encoder_rates": list(ENCODER_RATES),
                "decoder_rates": list(UPSAMPLE_RATES)}})}


def _fill(module: torch.nn.Module) -> torch.nn.Module:
    """The checkpoint-supplied parameters the modules leave ``torch.empty``: Snake's,
    which go through an ``exp``, and the attention biases. Unfilled they are garbage, not
    small wrong numbers."""
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, (SnakeBeta, Snake1d)):
                child.alpha.zero_()
            if isinstance(child, SnakeBeta):
                child.beta.zero_()
            if isinstance(child, CausalAttention):
                child.q_bias.zero_()
                child.v_bias.zero_()
                child.zero_k_bias.zero_()
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


# --- encoding a sound ---------------------------------------------------------


def _tone(frames: int, *, samples: int = SAMPLES_PER_LATENT, channels: int = 2,
          rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Whole latent frames of a 440 Hz tone: the posterior head reads layer norms, so a
    silent or constant input has no variance to normalize."""
    step = torch.arange(frames * samples, dtype=torch.float32)
    wave = torch.sin(step * (2 * math.pi * 440.0 / rate))
    return wave.view(1, 1, -1).repeat(channels, 1, 1)[0]


def test_a_sound_becomes_one_latent_frame_per_800_samples(tiny_vae):
    assert tiny_vae.encode_to_latents(_tone(4)).shape == (1, 32, 2, 4)


def test_a_tail_shorter_than_a_frame_still_occupies_one(tiny_vae):
    short = _tone(2)[..., :1200]
    assert tiny_vae.encode_to_latents(short).shape == (1, 32, 2, 2)


def test_a_mono_source_arrives_as_two_equal_lanes(tiny_vae):
    mono = _tone(2, channels=1)
    z = tiny_vae.encode_to_latents(mono)
    assert z.shape == (1, 32, 2, 2)
    assert torch.equal(z[0, :, 0], z[0, :, 1])


def test_a_source_at_another_rate_is_moved_to_the_codecs(tiny_vae):
    """A latent frame is 800 samples *at 32 kHz*, so a 48 kHz clip's own sample count
    says nothing about how many frames of sound it is."""
    from thenoise.utils.media import pcm_resample

    pcm = _tone(3, rate=48000)
    at_source = tiny_vae.encode_to_latents(pcm, sample_rate=48000)
    by_hand = tiny_vae.encode_to_latents(
        pcm_resample(pcm, 48000, SAMPLE_RATE), sample_rate=SAMPLE_RATE)
    assert at_source.shape == by_hand.shape == (1, 32, 2, 2)


def test_encoding_a_sound_never_samples_the_posterior(tiny_vae):
    """The checkpoint trains logs_proj and inference ignores it, so a condition's latent
    is the same whatever the run drew."""
    pcm = _tone(2)
    torch.manual_seed(1)
    first = tiny_vae.encode_to_latents(pcm)
    torch.manual_seed(9999)
    assert torch.equal(first, tiny_vae.encode_to_latents(pcm))


def test_an_encoded_sound_decodes_to_the_length_it_came_in_as(tiny_vae):
    pcm = _tone(3)
    z = tiny_vae.encode_to_latents(pcm)
    out = tiny_vae.decode_to_pcm(z)
    assert out.shape == (1, 2, 3 * SAMPLES_PER_LATENT)
    assert torch.isfinite(out).all()


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
                 "decoder.conv_post.weight", "latents_mean", "latents_std",
                 # encoder side
                 "encoder.block.0.weight",
                 "encoder.block.1.block.0.block.0.alpha",
                 "encoder.block.1.block.0.block.1.weight",
                 "encoder.block.1.block.0.block.3.weight",
                 "encoder.block.1.block.4.weight",
                 "pre_block.norm1.weight", "pre_block.proj.weight",
                 "pre_block.attn.qkv.weight", "pre_block.attn.q_bias",
                 "pre_block.attn.zero_k_bias", "pre_block.attn.proj.weight",
                 "pre_block.mlp.norm.weight", "pre_block.mlp.w0.weight",
                 "mean_proj.weight", "logs_proj.weight"):
        assert name in names, name
    # The posterior head is 32 wide in and out, and the attention reads the 2048 frames.
    assert tiny_vae.mean_proj.weight.shape == (32, 32, 1)
    assert tiny_vae.pre_block.attn.qkv.weight.shape[1] == 2048


# --- loader -------------------------------------------------------------------


def test_loader_reads_both_halves_of_the_codec(tmp_path, tiny_vae):
    path = write_safetensors(tmp_path / "h3-audio-vae.safetensors",
                             dict(tiny_vae.state_dict()), metadata=TINY_METADATA)

    loaded = load_h3_audio_vae(path, device="cpu")

    assert loaded.dtype == AutoencoderKLH3Audio.compute_dtype == torch.float32
    assert set(loaded.state_dict()) == set(tiny_vae.state_dict())
    assert loaded.decode_output_length(3) == tiny_vae.decode_output_length(3)


def test_loader_refuses_a_codec_whose_encoder_squashes_too_far(tmp_path, tiny_vae):
    path = write_safetensors(
        tmp_path / "audio.safetensors", dict(tiny_vae.state_dict()),
        metadata={"minimax_h3_audio_vae": json.dumps(
            {"kwargs": {"decoder_dim": 256, "latent_dim": 2048,
                        "decoder_rates": list(UPSAMPLE_RATES),
                        "encoder_rates": [2, 2, 2]}})})
    with pytest.raises(ValueError, match="encoder rates"):
        load_h3_audio_vae(path, device="cpu")


def test_loader_refuses_a_codec_at_another_sample_rate(tmp_path, tiny_vae):
    """The latent grid is spoken for by the video, so a codec sampling somewhere else
    would put a different amount of sound in every frame."""
    meta = json.loads(TINY_METADATA["minimax_h3_audio_vae"])
    meta["sample_rate"] = 44100
    path = write_safetensors(tmp_path / "audio.safetensors", dict(tiny_vae.state_dict()),
                             metadata={"minimax_h3_audio_vae": json.dumps(meta)})
    with pytest.raises(ValueError, match="44100"):
        load_h3_audio_vae(path, device="cpu")


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
