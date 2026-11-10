"""The H3-Image codec: 16x in both directions, on a reduced-width instance.

The encoder is the H3 video encoder held to a single frame, so its interesting property
is that the causal-temporal shortcut — convolving with the kernel's last tap instead of
padding zeros in front — is the same convolution. Then the latent and pixel affine
transformations, the fixed-seed posterior sample, and the loader's geometry sniffing.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import write_safetensors
from thenoise.vae import AutoencoderKLH3Image, load_h3_image_vae
from thenoise.vae.h3_image import ENCODE_SEED, LATENTS_MEAN, LATENTS_STD, CausalConv3d

def _tiny_vae() -> AutoencoderKLH3Image:
    """The shipped architecture with a 32-channel encoder and a narrow decoder.

    32 is the narrowest the codec's 32-group GroupNorm accepts.

    ``register_tokens`` and the decoder blocks' layer scales are ``torch.empty`` until a
    checkpoint fills them — and they are multiplied into every token, so uninitialized
    they turn each assertion about decoded pixels into a draw on whatever the allocator
    left behind.
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
    """Twice on the same pixels is the same latent, whatever the ambient RNG says.

    The decoder is trained on sampled latents, so encode samples — and on a seed of its
    own, which is what keeps a reference image's latent from moving with the
    generation's seed.
    """
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        first = tiny_vae.encode_pixels_to_latents(pixels)
        torch.manual_seed(ENCODE_SEED + 1)
        second = tiny_vae.encode_pixels_to_latents(pixels)
    assert torch.equal(first, second)


def test_single_frame_convolution_matches_the_padded_video():
    """The ``T = 1`` shortcut drops the front padding and the taps it isolates.

    The reference is the same convolution on a clip of two zero frames in front, run
    with the full 3D kernel: only the kernel's last temporal tap can see the frame, and
    the borders stay reflect-padded either way.
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
