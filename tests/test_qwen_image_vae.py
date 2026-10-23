"""The Qwen-Image codec (Wan 2.1 family): Qwen-Image and Ming-Image, still-image port.

This covers what the row-strip decode depends on and nothing else: that each strip layer
answers exactly as its plain module would, that the norm is the one kind a strip can
compute for itself, that the whole codec is strip-invariant, and that a block updating
its own input leaves the caller's tensors alone. The latent format, the loader and the
normalisation are ``test_vae.py``; the DiT is ``test_qwen_image.py``.
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
    """``F.normalize`` over channels: every pixel's scale comes from that pixel alone.

    The reason this codec strips without needing whole-frame statistics, where a
    GroupNorm codec has to carry them in.
    """
    norm = QwenImageRMS_norm(6).float().eval().requires_grad_(False)
    x = torch.randn(1, 6, 8, 3)
    with torch.no_grad():
        whole = norm(x)
        strips = torch.cat([norm(x.narrow(-2, 0, 3)), norm(x.narrow(-2, 3, 5))], dim=-2)
    assert torch.equal(whole, strips)


@pytest.mark.parametrize("out_dim", [6, 9])
def test_residual_block_matches_its_own_layers(out_dim, monkeypatch):
    """Both shapes of the block in strips: the one that adds into ``x``, and the one that
    needs a buffer of its own because it changes width.

    The block's return identity is the memory claim — a width-preserving block updating
    its input is what keeps a decoder level at one feature map instead of one per block —
    so it is pinned here rather than left to the camera's ``GiB`` column.
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

    The tiny threshold puts the full-resolution levels over several strips while the mid
    block stays a single one, so a strip reading a synthetic edge would show as a seam at
    the border of a level — which is where an in-place block is most likely to make one.
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
    """In-place blocks write to activations the codec made itself, never to an input.

    A caller reusing one latent for two decodes would otherwise get a different picture
    the second time, which no single-decode test can see.
    """
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
