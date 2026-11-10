"""The H3-Image codec at a reduced width: the single-frame causal shortcut, the latent and
pixel affines, the fixed-seed posterior sample, tiling and the loader's geometry
sniffing.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import write_safetensors
from thenoise.vae import AutoencoderKLH3Image, load_h3_image_vae
from thenoise.vae.h3_image import (
    ENCODE_SEED, LATENTS_MEAN, LATENTS_STD, TILE_BLEND, TILE_SIZE, CausalConv3d,
    blend_tiles, split_tiles,
)

def _tiny_vae() -> AutoencoderKLH3Image:
    """The shipped architecture with a 32-channel encoder, the narrowest its 32-group
    GroupNorm accepts.

    ``register_tokens`` and the decoder blocks' layer scales are ``torch.empty`` until a
    checkpoint fills them, and they multiply every token: unfilled, an assertion about
    decoded pixels draws on the allocator.
    """
    torch.manual_seed(0)
    vae = AutoencoderKLH3Image(ch=32, dim=64, num_layers=1)
    with torch.no_grad():
        vae.decoder.register_tokens.zero_()
        for block in vae.decoder.transformer_blocks:
            block.scale1.fill_(0.5)
            block.scale2.fill_(0.5)
    return vae


@pytest.fixture(scope="module")
def tiny_vae():
    return _tiny_vae().eval().requires_grad_(False)


def test_latent_geometry_is_24_channels_at_16_pixels(tiny_vae):
    assert AutoencoderKLH3Image.z_dim == 24
    assert AutoencoderKLH3Image.spatial_compression == 16
    assert len(LATENTS_MEAN) == len(LATENTS_STD) == 24

    with torch.no_grad():
        latent = tiny_vae.encode_pixels_to_latents(torch.zeros(1, 3, 64, 48))
    assert latent.shape == (1, 24, 4, 3)

    with torch.no_grad():
        pixels = tiny_vae.decode_to_pixels(latent)
    assert pixels.shape == (1, 3, 64, 48)
    assert pixels.min() >= -1.0 and pixels.max() <= 1.0


def test_encode_is_a_fixed_seed_posterior_sample(tiny_vae):
    """Twice on the same pixels is the same latent, whatever the ambient RNG says. The seed
    is encode's own, which keeps a reference image's latent from moving with the
    generation's seed.
    """
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        first = tiny_vae.encode_pixels_to_latents(pixels)
        torch.manual_seed(ENCODE_SEED + 1)
        second = tiny_vae.encode_pixels_to_latents(pixels)
    assert torch.equal(first, second)


def test_single_frame_convolution_matches_the_padded_video():
    """The ``T = 1`` shortcut against the same convolution on a clip with two zero frames
    padded in front and the full 3D kernel.
    """
    conv = CausalConv3d(3, 4, kernel_size=3, padding=1).eval()
    x = torch.randn(1, 3, 1, 6, 5)

    video = F.pad(F.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect"), (0, 0, 0, 0, 2, 0))
    with torch.no_grad():
        assert torch.equal(conv(x), F.conv3d(video, conv.weight, conv.bias))


def test_temporal_stride_keeps_one_frame_standing():
    """The stages that stride in time leave a single frame as one frame."""
    conv = CausalConv3d(2, 4, kernel_size=3, padding=(1, 0, 0), stride=(2, 2, 2)).eval()
    with torch.no_grad():
        out = conv(torch.randn(1, 2, 1, 8, 8), spatial_pad=(0, 1, 0, 1))
    assert out.shape == (1, 4, 1, 4, 4)


def test_decode_reads_the_register_suffix_back_out(tiny_vae):
    """The decoder runs over latent cells plus five spare rows and returns the cells."""
    latent = torch.randn(1, 24, 2, 3)
    with torch.no_grad():
        pixels = tiny_vae.decode_to_pixels(latent)
    assert pixels.shape == (1, 3, 32, 48)


def test_dtype_and_device_follow_the_weights(tiny_vae):
    assert tiny_vae.dtype == tiny_vae.decoder.x_embedder.weight.dtype
    assert tiny_vae.device == tiny_vae.decoder.x_embedder.weight.device


def test_split_tiles_lands_on_the_frame_edge():
    """Tiles cover the frame exactly: the slack the minimum overlaps leave is spread a
    latent cell at a time.
    """
    for length in (272, 320, 512, 704, 1024, 1248):
        starts, lengths, overlaps = split_tiles(length, TILE_SIZE, TILE_BLEND, 16)
        assert starts[0] == 0
        assert starts[-1] + lengths[-1] == length
        assert len(starts) == len(lengths) == len(overlaps) + 1
        assert all(o >= TILE_BLEND and o % 16 == 0 for o in overlaps)
        assert all(s + l <= length for s, l in zip(starts, lengths))

    assert split_tiles(240, TILE_SIZE, TILE_BLEND, 16) == ([0], [240], [])
    with pytest.raises(ValueError, match="tile blend"):
        split_tiles(512, TILE_SIZE, TILE_SIZE, 16)


def test_blend_tiles_ramps_then_hands_the_rest_over():
    a, b = torch.zeros(1, 1, 4), torch.ones(1, 1, 4)
    out = blend_tiles(a, b, 2, dim=-1)
    assert out.shape == b.shape
    assert torch.allclose(out, torch.tensor([[[0.0, 0.5, 1.0, 1.0]]]))
    assert torch.equal(blend_tiles(a, b, 0, dim=-1), b)


def test_the_codec_only_ever_sees_a_tile(tiny_vae, monkeypatch):
    """The rope spans whatever grid it is handed, so nothing may reach either half wider
    than a tile.
    """
    monkeypatch.setattr(tiny_vae, "tile_size", 64)
    monkeypatch.setattr(tiny_vae, "tile_blend", 32)
    cells = 64 // tiny_vae.patch_size
    seen = []

    def spy(tile):
        seen.append(tuple(tile.shape[-2:]))
        return torch.zeros(1, 3, tile.shape[-2] * 16, tile.shape[-1] * 16)

    monkeypatch.setattr(tiny_vae, "_decode_pixels", spy)
    with torch.no_grad():
        pixels = tiny_vae.decode_to_pixels(torch.zeros(1, 24, 12, 12))

    assert pixels.shape == (1, 3, 192, 192)
    assert len(seen) == 25, "192 pixels of 64-pixel tiles with overlap is 5 tiles per side"
    assert set(seen) == {(cells, cells)}

    seen.clear()
    monkeypatch.setattr(tiny_vae, "_encode_moments",
                        lambda x: seen.append(tuple(x.shape[-2:]))
                        or torch.zeros(1, 48, 1, cells, cells))
    with torch.no_grad():
        tiny_vae.encode_pixels_to_latents(torch.zeros(1, 3, 192, 192))
    assert len(seen) == 25
    assert set(seen) == {(64, 64)}


def test_a_frame_inside_a_tile_is_decoded_as_one_grid(tiny_vae):
    """Below a tile there is nothing to cut, and tiling must not disturb the result."""
    latent = torch.randn(1, 24, 3, 4)
    pixels = torch.randn(1, 3, 48, 64)
    with torch.no_grad():
        assert torch.equal(tiny_vae._tiled_decode(latent), tiny_vae._decode_pixels(latent))
        assert torch.equal(tiny_vae._tiled_moments(pixels), tiny_vae._encode_moments(pixels))


def _write_checkpoint(tmp_path, vae, drop=()):
    tensors = {k: v for k, v in vae.state_dict().items() if k not in drop}
    return write_safetensors(tmp_path / "h3-vae.safetensors", tensors)


def test_loader_reads_the_geometry_off_the_file(tmp_path, tiny_vae):
    path = _write_checkpoint(tmp_path, tiny_vae)
    loaded = load_h3_image_vae(path, device="cpu")

    assert loaded.encoder.conv_in.weight.shape == (32, 3, 3, 3, 3)
    assert loaded.decoder.x_embedder.weight.shape == (64, 24)
    assert len(loaded.decoder.transformer_blocks) == 1
    loaded = loaded.decoder.proj_out.weight.float()
    assert torch.allclose(loaded, tiny_vae.decoder.proj_out.weight, atol=1e-3)


def test_loader_lands_on_the_codec_dtype(tmp_path, tiny_vae):
    loaded = load_h3_image_vae(_write_checkpoint(tmp_path, tiny_vae), device="cpu")
    assert loaded.dtype == AutoencoderKLH3Image.compute_dtype


def test_loader_refuses_a_foreign_checkpoint(tmp_path):
    path = write_safetensors(tmp_path / "other.safetensors", {"decoder.conv_in.weight": torch.zeros(3, 3)})
    with pytest.raises(ValueError, match="not a H3-Image VAE"):
        load_h3_image_vae(path, device="cpu")


def test_loader_is_strict(tmp_path, tiny_vae):
    path = _write_checkpoint(tmp_path, tiny_vae, drop=("decoder.proj_out.bias",))
    with pytest.raises(RuntimeError):
        load_h3_image_vae(path, device="cpu")
