"""The Qwen-Image codec (Wan 2.1 family), row-strip decode: what each strip layer answers,
strip invariance of the whole codec, and the in-place blocks leaving inputs alone.

The latent format and the loader are ``test_vae.py``; the DiT is ``test_qwen_image.py``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from thenoise.vae import AutoencoderKLQwenImage
from thenoise.vae.qwen_image import (
    QwenImageResample,
    QwenImageResidualBlock,
    QwenImageRMS_norm,
)
import thenoise.vae.fused as fused_module


def plain_block(blk, x):
    """``QwenImageResidualBlock.forward`` as the reference implementation writes it."""
    h = blk.conv_shortcut(x)
    x = blk.norm1(x)
    x = blk.nonlinearity(x)
    x = blk.conv1(x)
    x = blk.norm2(x)
    x = blk.nonlinearity(x)
    x = blk.conv2(x)
    return x + h


@pytest.fixture(scope="module")
def tiny_vae() -> AutoencoderKLQwenImage:
    """The shipped geometry (three downsampling stages, 16ch latent) at 4-wide blocks."""
    return AutoencoderKLQwenImage(base_dim=4).eval().requires_grad_(False)


# ------------------------------------------------------------------- strip layers


def test_the_norm_is_the_one_a_strip_can_compute_for_itself():
    """``F.normalize`` over channels: a strip can compute this norm for itself.

    Where a GroupNorm codec has to carry whole-frame statistics in, this one needs none.
    """
    norm = QwenImageRMS_norm(6).float().eval().requires_grad_(False)
    x = torch.randn(1, 6, 8, 3)
    with torch.no_grad():
        whole = norm(x)
        strips = torch.cat([norm(x.narrow(-2, 0, 3)), norm(x.narrow(-2, 3, 5))], dim=-2)
    assert torch.equal(whole, strips)


@pytest.mark.parametrize("out_dim", [6, 9])
def test_residual_block_matches_its_own_layers(out_dim, monkeypatch):
    """Both shapes of the block in strips, and which of them returns the tensor it read.

    That identity is what keeps a level at one feature map instead of one per block.
    """
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", 1 << 8)
    blk = QwenImageResidualBlock(6, out_dim).float().eval().requires_grad_(False)
    x = torch.randn(1, 6, 12, 5)
    with torch.no_grad():
        want = plain_block(blk, x)
        pristine = x.clone()
        got = blk(x)
    assert torch.allclose(got, want, atol=1e-5)
    inplace = isinstance(blk.conv_shortcut, nn.Identity)
    assert (got is x) is inplace
    assert torch.equal(x, pristine) is not inplace


@pytest.mark.parametrize("mode", ["upsample2d", "downsample2d", "same"])
def test_resample_matches_the_plain_module_it_wraps(mode, monkeypatch):
    monkeypatch.setattr(fused_module, "STRIP_ELEMS", 1 << 8)
    resample = QwenImageResample(6, mode).float().eval().requires_grad_(False)
    x = torch.randn(1, 6, 12, 7)
    with torch.no_grad():
        assert torch.allclose(resample(x), resample.resample(x), atol=1e-5)


# --------------------------------------------------------------- the whole codec


def test_encode_and_decode_are_strip_invariant(tiny_vae, monkeypatch):
    """Strips are a memory device, not an approximation: same numbers either way.

    The tiny threshold leaves the mid block a single strip, so a seam would show up as a
    border between levels — where an in-place block is most likely to make one.
    """
    z = torch.randn(1, 16, 4, 4)
    pixels = torch.rand(1, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        ref_decode = tiny_vae.decode(z)["sample"]
        ref_encode = tiny_vae.encode(pixels)["latent_dist"].mode()
        monkeypatch.setattr(fused_module, "STRIP_ELEMS", 64)
        got_decode = tiny_vae.decode(z)["sample"]
        got_encode = tiny_vae.encode(pixels)["latent_dist"].mode()
    assert torch.allclose(got_decode, ref_decode, atol=1e-4)
    assert torch.allclose(got_encode, ref_encode, atol=1e-4)


def test_the_codec_leaves_what_the_caller_handed_it(tiny_vae):
    """In-place blocks write to activations the codec made itself, never to an input."""
    z = torch.randn(1, 16, 4, 4)
    pixels = torch.rand(1, 3, 64, 64) * 2 - 1
    z0, pixels0 = z.clone(), pixels.clone()
    with torch.no_grad():
        tiny_vae.decode(z)
        tiny_vae.encode(pixels)
        tiny_vae.decode_to_pixels(z)
        tiny_vae.encode_pixels_to_latents(pixels)
    assert torch.equal(z, z0)
    assert torch.equal(pixels, pixels0)
