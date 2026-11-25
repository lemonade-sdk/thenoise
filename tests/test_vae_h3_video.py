"""The H3 video codec at a reduced width: the frame plan the released clip geometry
implies, per-frame norm statistics, the causal conv's temporal taps, streaming decode
against the promised shape, spatial tiling and the loader's geometry sniffing.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import write_safetensors
from torch.testing import assert_close
from thenoise.vae import AutoencoderKLH3Video, load_h3_video_vae
from thenoise.dit.h3.layout import FRAME_PER_TOKEN, FRAME_RESCALE, time_spans
from thenoise.vae.h3 import (
    CLIP_LENGTH, LATENTS_MEAN, LATENTS_STD, TIME_RATIO, TOKEN_DROP, VIDEO_GEOMETRY_SECTION,
    AutoencoderKLH3Image, CausalConv3d, Downsample3D, TemporalIsolatedGroupNorm,
    TemporalPlan, group_norm_3d,
)


def _fill(vae):
    """The decoder's ``register_tokens`` and layer scales are ``torch.empty`` until a
    checkpoint fills them, and they multiply every token."""
    with torch.no_grad():
        vae.decoder.register_tokens.zero_()
        for block in vae.decoder.transformer_blocks:
            block.scale1.fill_(0.5)
            block.scale2.fill_(0.5)
    return vae


def _tiny_vae(**kwargs) -> AutoencoderKLH3Video:
    torch.manual_seed(0)
    return _fill(AutoencoderKLH3Video(ch=32, dim=64, num_layers=1, **kwargs))


@pytest.fixture(scope="module")
def tiny_vae():
    return _tiny_vae().eval().requires_grad_(False)


# --- the frame plan ----------------------------------------------------------


def test_released_clip_geometry():
    assert AutoencoderKLH3Video.z_dim == 24
    assert AutoencoderKLH3Video.spatial_compression == 16
    assert (CLIP_LENGTH, TOKEN_DROP, TIME_RATIO) == (17, 3, 4)


@pytest.mark.parametrize(("latent_t", "frames"), [
    (1, 1), (2, 5), (3, 9), (5, 17), (7, 22), (8, 26), (10, 34),
    (37, 124), (38, 128), (40, 136),
])
def test_frame_plan_follows_the_released_clip(latent_t, frames):
    """The interface asks for ``17k + 5`` frames; the decoder answers whatever it is
    handed with the frames its causal codec actually holds."""
    assert TemporalPlan.of(latent_t).frames == frames


def test_frame_plan_is_the_frames_each_latent_carries():
    """The first latent of every group carries the frame the warm-up took from it."""
    for latent_t in range(1, 60):
        plan = TemporalPlan.of(latent_t)
        assert plan.frames == sum(FRAME_PER_TOKEN[k % plan.chunk_tokens]
                                  for k in range(latent_t))


def test_the_dit_and_the_decoder_share_one_time_axis():
    """The DiT spaces its latent frames by ``FRAME_PER_TOKEN``; the codec derives the same
    schedule from the clip length it was trained on. Two modules, one number, and this is
    where they are tied."""
    per_token = (CLIP_LENGTH - (TemporalPlan.of(1).chunk_tokens - 1) * TIME_RATIO,
                 *([TIME_RATIO] * (TemporalPlan.of(1).chunk_tokens - 1)))
    assert tuple(per_token) == FRAME_PER_TOKEN
    for latent_t in range(1, 40):
        plan = TemporalPlan.of(latent_t)
        assert sum(time_spans(latent_t)) == pytest.approx(plan.frames * FRAME_RESCALE)


@pytest.mark.parametrize(("latent_t", "pad", "clips"), [
    (2, 5, 1), (3, 4, 1), (5, 2, 1), (7, 0, 1), (10, 2, 2), (37, 0, 7), (38, 4, 8),
])
def test_clips_step_over_the_latent_with_overlap(latent_t, pad, clips):
    plan = TemporalPlan.of(latent_t)
    assert (plan.pad_tokens, plan.clips) == (pad, clips)

    starts = [i * plan.chunk_tokens for i in range(plan.clips)]
    assert starts[0] == 0
    assert all(b - a == plan.chunk_tokens for a, b in zip(starts, starts[1:]))
    assert starts[-1] + plan.chunk_tokens + plan.overlap_tokens >= latent_t + pad
    assert (plan.pre_padding, plan.overlap_tokens, plan.frame_overlap) == (3, 2, 5)


# --- the encoder's temporal behaviour ----------------------------------------


def test_per_frame_statistics_do_not_leak_across_frames():
    x = torch.randn(1, 8, 3, 5, 5)
    loud = torch.cat([x[:, :, :1] * 8.0, x[:, :, 1:]], dim=2)
    norm = TemporalIsolatedGroupNorm(num_groups=4, num_channels=8, eps=1e-6, affine=False)
    plain = nn.GroupNorm(num_groups=4, num_channels=8, eps=1e-6, affine=False)
    with torch.no_grad():
        assert torch.equal(norm(x)[:, :, 1:], norm(loud)[:, :, 1:])
        assert not torch.allclose(plain(x)[:, :, 1], plain(loud)[:, :, 1]), \
            "the leak the isolated norm exists to prevent"


def test_a_single_frame_norm_is_the_plain_one():
    norm = group_norm_3d(32)
    x = torch.randn(2, 32, 1, 6, 6)
    with torch.no_grad():
        assert torch.equal(norm(x), F.group_norm(x, 32, norm.weight, norm.bias, eps=norm.eps))


def test_the_conv_is_causal_over_a_clip():
    conv = CausalConv3d(2, 3, kernel_size=3, padding=1).eval()
    x = torch.randn(1, 2, 4, 5, 5)
    later = torch.cat([x[:, :, :1], torch.randn_like(x[:, :, 1:])], dim=2)
    with torch.no_grad():
        assert torch.equal(conv(x)[:, :, 0], conv(later)[:, :, 0]), \
            "the first frame stands in front of zeros"
        assert not torch.allclose(conv(x)[:, :, 3], conv(later)[:, :, 3])
    assert conv(x).shape == (1, 3, 4, 5, 5), "causal padding keeps the frame count"


def test_a_single_frame_sees_what_the_first_of_a_clip_sees():
    """The still shortcut truncates the temporal taps; a clip's first frame reads the
    same taps off the zeros padded in front, so the two answers are one answer.

    One answer, not one bit pattern: these are different conv problems — one tap over one
    frame against three taps over a clip padded with four zeros — and how a given kernel
    sums them is not ours to fix. A wrong tap set misses by units, not by rounding.
    """
    conv = CausalConv3d(2, 3, kernel_size=3, padding=1).eval()
    x = torch.randn(1, 2, 1, 5, 5)
    clip = torch.cat([x, torch.randn_like(x), torch.randn_like(x)], dim=2)
    with torch.no_grad():
        assert_close(conv(x), conv(clip)[:, :, :1], atol=1e-6, rtol=1e-5)


def test_temporal_stride_halves_a_clip():
    down = Downsample3D(2, 4, space_stride=2, time_stride=2).eval()
    with torch.no_grad():
        assert down(torch.zeros(1, 2, 17, 8, 8)).shape == (1, 4, 9, 4, 4)
        assert down(torch.zeros(1, 2, 1, 8, 8)).shape == (1, 4, 1, 4, 4)


# --- decode ------------------------------------------------------------------


def test_decode_answers_the_shape_it_promised(tiny_vae):
    for latent_t in (1, 2, 5, 8):
        latent = torch.randn(1, 24, latent_t, 2, 3)
        with torch.no_grad():
            pixels = tiny_vae.decode_to_pixels(latent)
        assert pixels.shape == tiny_vae.decode_output_shape(tuple(latent.shape))
        assert torch.isfinite(pixels).all()
        assert pixels.min() >= -1.0 and pixels.max() <= 1.0


def test_decode_hands_frames_out_per_clip(tiny_vae):
    """Eight latent frames is two clips, and the pieces are the clips' parts, not one
    buffer handed out twice."""
    latent = torch.randn(1, 24, 8, 2, 3)
    with torch.no_grad():
        pieces = [p.shape[2] for p in tiny_vae.decode_to_pixel_frames(latent)]
    assert len(pieces) == 3
    assert sum(pieces) == tiny_vae.decode_output_shape(tuple(latent.shape))[2]


def test_a_lone_latent_is_decoded_inside_a_clip_of_copies(tiny_vae, monkeypatch):
    """A frame decoded cold is worse than a padded one, and it is one frame out."""
    seen = []

    def spy(z):
        seen.append(z.shape[2])
        return torch.zeros(z.shape[0], 3, z.shape[2] * tiny_vae.patch_size_t,
                           z.shape[-2] * tiny_vae.patch_size, z.shape[-1] * tiny_vae.patch_size,
                           dtype=z.dtype)

    monkeypatch.setattr(tiny_vae, "_decode_pixels", spy)
    with torch.no_grad():
        frames = list(tiny_vae.decode_to_pixel_frames(
            torch.zeros(1, 24, 1, 2, 3)))

    assert [f.shape[2] for f in frames] == [1]
    assert seen == [TemporalPlan.of(1).chunk_tokens]


def test_a_frame_inside_a_tile_is_decoded_as_one_grid(tiny_vae):
    latent = torch.randn(1, 24, 2, 3, 4)
    with torch.no_grad():
        assert torch.equal(tiny_vae._tiled_decode(latent), tiny_vae._decode_pixels(latent))


def test_the_codec_only_ever_sees_a_tile(tiny_vae, monkeypatch):
    monkeypatch.setattr(tiny_vae, "tile_size", 32)
    monkeypatch.setattr(tiny_vae, "tile_blend", 16)
    cells = 32 // tiny_vae.patch_size
    seen = []

    def spy(tile):
        seen.append(tuple(tile.shape[-2:]))
        return torch.zeros(tile.shape[0], 3, tile.shape[2] * tiny_vae.patch_size_t,
                           tile.shape[-2] * 16, tile.shape[-1] * 16, dtype=tile.dtype)

    monkeypatch.setattr(tiny_vae, "_decode_pixels", spy)
    with torch.no_grad():
        pixels = tiny_vae.decode_to_pixels(torch.zeros(1, 24, 2, 4, 4))

    assert pixels.shape[3:] == (64, 64)
    assert len(seen) == 9
    assert set(seen) == {(cells, cells)}


# --- encode ------------------------------------------------------------------


def test_encode_compresses_a_clip_to_the_plan(tiny_vae):
    frames = 5 + CLIP_LENGTH
    pixels = torch.randn(1, 3, frames, 32, 48)
    with torch.no_grad():
        latent = tiny_vae.encode_pixels_to_latents(pixels)
    assert latent.shape == (1, 24, 2 + (frames - 5) // 17 * 5, 2, 3)
    assert torch.isfinite(latent).all()


def test_encode_of_one_frame_is_one_latent(tiny_vae):
    with torch.no_grad():
        latent = tiny_vae.encode_pixels_to_latents(
            torch.zeros(1, 3, 32, 48))
    assert latent.shape == (1, 24, 1, 2, 3)


def test_encode_is_the_mean_so_it_repeats(tiny_vae):
    """Unlike the image codec there is no posterior sample to move with a seed."""
    pixels = torch.randn(1, 3, 32, 48)
    with torch.no_grad():
        assert torch.equal(tiny_vae.encode_pixels_to_latents(pixels),
                           tiny_vae.encode_pixels_to_latents(pixels))


# --- loader ------------------------------------------------------------------


def test_loader_reads_geometry_and_stats(tmp_path, tiny_vae):
    tensors = dict(tiny_vae.state_dict())
    tensors["latents_mean"] = torch.tensor(LATENTS_MEAN)
    tensors["latents_std"] = torch.tensor(LATENTS_STD)
    path = write_safetensors(tmp_path / "h3-video-vae.safetensors", tensors)

    loaded = load_h3_video_vae(path, device="cpu")

    assert (loaded.patch_size, loaded.patch_size_t) == (16, 4)
    assert (loaded.clip_length, loaded.token_drop) == (CLIP_LENGTH, TOKEN_DROP)
    assert torch.allclose(loaded.latents_mean.float().flatten(), torch.tensor(LATENTS_MEAN),
                          rtol=2e-3)
    assert torch.allclose(loaded.latents_std.float().flatten(), torch.tensor(LATENTS_STD),
                          rtol=2e-3)
    assert loaded.dtype == AutoencoderKLH3Video.compute_dtype


def test_loader_takes_the_clip_policy_from_the_metadata(tmp_path, tiny_vae):
    path = write_safetensors(
        tmp_path / "v.safetensors", dict(tiny_vae.state_dict()),
        metadata={VIDEO_GEOMETRY_SECTION: json.dumps({"vae_clip_length": 9,
                                                      "vae_token_drop": 1})})

    loaded = load_h3_video_vae(path, device="cpu")

    assert (loaded.clip_length, loaded.token_drop) == (9, 1)
    here = TemporalPlan.of(5, loaded.clip_length, loaded.token_drop, ratio_t=4)
    assert (here.chunk_tokens, here.pre_padding) == (3, 3)
    assert here.frames != TemporalPlan.of(5).frames


@pytest.mark.parametrize("build", [
    lambda: AutoencoderKLH3Image(ch=32, dim=64, num_layers=1),
    lambda: AutoencoderKLH3Video(ch=32, ch_mult=(1, 1), space_down=(2, 2),
                                 time_down=(1, 1), dim=64, num_layers=1),
])
def test_loader_refuses_a_decoder_it_cannot_fill(tmp_path, build):
    """The image decoder answers one frame per cell; a codec whose strides and decoder
    disagree on that is not this codec."""
    path = write_safetensors(tmp_path / "not-video.safetensors", dict(build().state_dict()))
    with pytest.raises(ValueError, match="proj_out"):
        load_h3_video_vae(path, device="cpu")


def test_loader_refuses_a_sol_decoder(tmp_path):
    """The Sol decoder is another module tree, so it fails the signature by name."""
    tensors = {f"decoder.blocks.{i}.attn.to_qkv.weight": torch.zeros(64, 64)
               for i in range(2)}
    tensors["encoder.conv_in.weight"] = torch.zeros(32, 3, 3, 3, 3)
    path = write_safetensors(tmp_path / "sol-vae.safetensors", tensors)
    with pytest.raises(ValueError, match="signature"):
        load_h3_video_vae(path, device="cpu")
