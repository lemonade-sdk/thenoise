"""VAE-level contract: each VAE states the canonical latent it produces.

The pipeline's canonical latent ``[B, C, H, W]`` *is* the VAE's output format, so
its geometry lives on the VAE (``z_dim``, ``spatial_compression``) rather than
being re-declared by the model adapters — a DiT's own input width is a different
number (it patchifies the latent further), and two declarations of one fact is how
they drift apart.

The second half covers the shared Wan2.1-family loader, which builds ONE
architecture two ways — Qwen-Image (RGB, per-channel z-scored latent) and
Ming-Image (RGBA, scalar scale) — deciding which from the weights themselves.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from conftest import write_safetensors
from thenoise.upscale.inference_adaptors import make_wan21
from thenoise.vae import (
    AutoencoderKLFlux,
    AutoencoderKLFlux2,
    AutoencoderKLQwenImage,
    load_ming_vae,
    load_qwen_vae,
)
from thenoise.vae import qwen_image as qwen_module
from thenoise.vae.qwen_image import (
    MING_IMAGE_SCALE_FACTOR,
    QWEN_IMAGE_LATENTS_MEAN,
    QWEN_IMAGE_LATENTS_STD,
)


def test_class_level_geometry():
    """The weight-free VAEs state their latent geometry as class constants."""
    assert (AutoencoderKLFlux.z_dim, AutoencoderKLFlux.spatial_compression) == (16, 8)
    # Flux.2 packs the 32ch VAE latent 2x2 -> 128ch at 16 pixels per latent cell.
    assert (AutoencoderKLFlux2.z_dim, AutoencoderKLFlux2.spatial_compression) == (128, 16)


def test_qwen_geometry_follows_the_encoder_depth():
    """The shared Qwen-Image VAE derives compression from its downsampling stages."""
    vae = AutoencoderKLQwenImage(base_dim=4)  # tiny random init: no weights needed
    assert vae.z_dim == 16
    assert vae.spatial_compression == 8  # len(dim_mult) - 1 == 3 stages


@pytest.mark.parametrize(
    "dim_mult,compression",
    [([1, 2], 2), ([1, 2, 4], 4), ([1, 2, 4, 4], 8), ([1, 2, 4, 4, 4], 16)],
)
def test_qwen_compression_follows_the_downsampling_stages(dim_mult, compression):
    """One halving per stage past the first: ``2 ** (len(dim_mult) - 1)``."""
    vae = AutoencoderKLQwenImage(base_dim=4, dim_mult=dim_mult)
    assert vae.spatial_compression == compression


# ------------------------------------------------ the shared Wan2.1-family loader


@pytest.fixture
def tiny_family_arch(monkeypatch):
    """Pin the family architecture constant the loaders read to the tiny width these
    checkpoints are built at — the loaders keep their production signature.
    """
    monkeypatch.setattr(
        qwen_module,
        "_WAN21_FAMILY_ARCH",
        dict(qwen_module._WAN21_FAMILY_ARCH, base_dim=4),
    )


def _family_checkpoint(channels: int) -> tuple[dict, dict]:
    """A tiny family checkpoint as a file holds it: ``(stored, expected)``.

    ``channels`` rewrites the two tensors that decide the pixel width; everything is
    re-expanded to the 5D video layout with garbage in all but the LAST time slice.
    """
    source = AutoencoderKLQwenImage(base_dim=4, input_channels=4)
    stored, expected = {}, {}
    for key, val in source.state_dict().items():
        if key == "encoder.conv_in.weight":
            val = val[:, :channels]
        elif key.startswith("decoder.conv_out."):
            val = val[:channels]
        expected[key] = val
        if val.dim() == 4:
            wide = torch.full((val.shape[0], val.shape[1], 3) + val.shape[2:], -12345.0)
            wide[:, :, -1] = val
            stored[key] = wide
        elif key.endswith(".gamma") and val.dim() == 3:
            stored[key] = val.unsqueeze(-1)  # the file's (C, 1, 1, 1)
        else:
            stored[key] = val

    # Video-only ``time_conv`` layers: dropped by the loader, never unexpected keys.
    stored["encoder.down_blocks.2.resample.1.time_conv.weight"] = torch.zeros(8, 8, 3, 1, 1)
    stored["decoder.up_blocks.1.upsamplers.0.resample.1.time_conv.weight"] = torch.zeros(8, 8, 3, 1, 1)
    return stored, expected


def _assert_weights_survived_the_collapse(vae, expected: dict):
    """One 2D weight per checkpoint tensor, bit-exactly, and nothing left over."""
    loaded = vae.state_dict()
    assert set(loaded) == set(expected)
    for key, value in expected.items():
        assert torch.equal(loaded[key], value), key


def test_load_ming_vae_is_rgba_with_the_scalar_scale(tmp_path, tiny_family_arch):
    stored, expected = _family_checkpoint(4)
    path = write_safetensors(tmp_path / "ming_vae.safetensors", stored)

    vae = load_ming_vae(path, device="cpu")

    assert (vae.z_dim, vae.spatial_compression, vae.pixel_channels) == (16, 8, 4)
    # The vendor's scalar pair (``canonical = raw * 8.0064``, shift 0), broadcastable.
    assert vae._latents_mean.shape == (1, 1, 1, 1)
    assert float(vae._latents_mean) == 0.0
    assert float(vae._latents_inv_std) == pytest.approx(MING_IMAGE_SCALE_FACTOR)
    _assert_weights_survived_the_collapse(vae, expected)


def test_load_qwen_vae_is_rgb_with_the_per_channel_z_score(tmp_path, tiny_family_arch):
    stored, expected = _family_checkpoint(3)
    path = write_safetensors(tmp_path / "qwen_vae.safetensors", stored)

    vae = load_qwen_vae(path, device="cpu")

    assert (vae.z_dim, vae.spatial_compression, vae.pixel_channels) == (16, 8, 3)
    assert torch.equal(vae._latents_mean.view(-1), torch.tensor(QWEN_IMAGE_LATENTS_MEAN))
    assert torch.allclose(
        vae._latents_inv_std.view(-1), 1.0 / torch.tensor(QWEN_IMAGE_LATENTS_STD)
    )
    _assert_weights_survived_the_collapse(vae, expected)


def test_family_loader_fails_loudly_on_disagreeing_pixel_widths(tmp_path):
    """An encoder/decoder pixel-width mismatch is not one VAE's round trip, and
    guessing either side would change how many channels every image is written with.
    """
    stored, _ = _family_checkpoint(4)
    stored["decoder.conv_out.weight"] = stored["decoder.conv_out.weight"][:3]
    path = write_safetensors(tmp_path / "split_vae.safetensors", stored)

    with pytest.raises(ValueError, match="encoder takes 4 pixel channels"):
        load_ming_vae(path, device="cpu")


@pytest.mark.parametrize(
    "loader,want,have",
    [(load_ming_vae, 4, 3), (load_qwen_vae, 3, 4)],
    ids=["rgba-loader-on-rgb", "rgb-loader-on-rgba"],
)
def test_each_loader_demands_the_pixel_width_its_model_needs(tmp_path, loader, want, have):
    """A width the file denies is an error, not a silently wrong channel count."""
    stored, _ = _family_checkpoint(have)
    path = write_safetensors(tmp_path / "mismatched_vae.safetensors", stored)

    with pytest.raises(ValueError, match=f"the checkpoint is a {have}-channel model"):
        loader(path, device="cpu")


def test_family_loader_fails_on_a_latent_width_the_file_denies(tmp_path):
    """A wrong ``z_dim`` would feed the DiT a latent it cannot patchify."""
    stored, _ = _family_checkpoint(4)
    stored["post_quant_conv.weight"] = torch.zeros(8, 8, 1, 1)
    path = write_safetensors(tmp_path / "narrow_vae.safetensors", stored)

    with pytest.raises(ValueError, match="the config says a 16ch latent"):
        load_ming_vae(path, device="cpu")


def test_the_scalar_normalisation_is_an_exact_inverse(monkeypatch):
    """``canonical = raw * 8.0064`` in BOTH directions, with the networks stubbed so
    what round-trips is the normalisation alone.
    """
    vae = AutoencoderKLQwenImage(
        base_dim=4,
        scale_factor=MING_IMAGE_SCALE_FACTOR,
        input_channels=4,
    ).eval()
    raw = torch.full((1, 16, 1, 1), 0.05)

    monkeypatch.setattr(
        vae, "encode", lambda x, return_dict=False: (SimpleNamespace(mode=lambda: raw),)
    )
    model = vae.encode_pixels_to_latents(torch.zeros(1, 4, 8, 8))
    assert torch.allclose(model, raw * MING_IMAGE_SCALE_FACTOR)

    monkeypatch.setattr(vae, "decode", lambda z, return_dict=False: (z,))
    assert torch.allclose(vae.decode_to_pixels(model), raw, atol=1e-6)


def test_mixed_normalisations_are_an_error():
    """The z-score and the scalar are alternatives, not layers to compose."""
    with pytest.raises(ValueError, match="not both"):
        AutoencoderKLQwenImage(
            base_dim=4, scale_factor=8.0064, latents_mean=QWEN_IMAGE_LATENTS_MEAN
        )


@pytest.mark.parametrize(
    "factory,vae",
    [
        (
            make_wan21,
            lambda: AutoencoderKLQwenImage(base_dim=4),
        ),
    ],
    ids=["wan21"],
)
def test_the_upscaler_adaptor_means_what_the_vae_means(factory, vae):
    """Sesqui works on RAW latents, so its adaptor must be the VAE's own transform:
    if the two disagree an upscale silently rescales the latent, reading as a
    washed-out picture rather than an error.
    """
    adaptor, vae = factory(), vae()
    z = torch.randn(1, 16, 2, 2)
    mean, inv_std = vae._latents_mean, vae._latents_inv_std

    assert torch.allclose(adaptor.to_vae_latent(z), z / inv_std + mean)
    assert torch.allclose(adaptor.from_vae_latent(z), (z - mean) * inv_std)
