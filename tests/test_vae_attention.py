"""The attention the VAE decoders share instead of SDPA, and its query tiling.

Two contracts: the helper computes what SDPA computes, scale included, and no VAE decode
asks a fused backend for its pixels. The patched blocks in ``mage_flow`` and ``flux2``
are held to their upstreams, and to the one property that exposes a wrong axis with no
upstream to copy — attention over positions must commute with reordering them.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from thenoise.utils import attention as attn
from thenoise.utils.attention import single_head_attention
from thenoise.vae import AutoencoderKLFlux
from thenoise.vae.flux2 import _AttnBlock
from thenoise.vae.mage_flow import _attention as mage_attention


def _sdpa_formula(q, k, v):
    """SDPA's definition, in fp32: ``softmax(q k^T / sqrt(E)) v``."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    q, k, v = q.float(), k.float(), v.float()
    return ((q @ k.transpose(-2, -1)) * scale).softmax(-1) @ v


@pytest.mark.parametrize("shape", [(2, 24, 8), (1, 3, 32, 6), (1, 2, 5, 7)])
def test_it_is_sdpa_down_to_the_default_scale(shape):
    """``(..., N, E)`` at any rank, scaled by ``1/sqrt(E)`` as SDPA scales."""
    q, k, v = (torch.randn(*shape) for _ in range(3))
    out = single_head_attention(q, k, v)
    assert out.shape == q.shape
    assert torch.allclose(out, _sdpa_formula(q, k, v), atol=1e-5, rtol=1e-4)


def test_it_agrees_with_the_backend_it_replaces():
    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    assert torch.allclose(single_head_attention(q, k, v),
                          F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-4)


def test_it_survives_the_compute_dtype_the_decoders_run_in():
    q, k, v = (torch.randn(1, 1, 48, 64, dtype=torch.bfloat16) for _ in range(3))
    out = single_head_attention(q, k, v)
    assert out.dtype == torch.bfloat16 and out.shape == q.shape
    assert torch.isfinite(out).all()


def test_the_tile_policy_leaves_the_small_shapes_alone():
    """Where tiling starts, in rows, at the shapes the real decoders hit."""
    assert attn.score_tile_rows(1024) == 1024     # mage windows
    assert attn.score_tile_rows(16384) == 16384   # every codec's 1024x1024 rung
    assert attn.score_tile_rows(65536) == 1024    # qwen/ming/Flux at 2048x2048
    assert attn.score_tile_rows(262144) == 256    # 4K latents


@pytest.mark.parametrize("rows", [1, 63, 128, 257, 10_000])
def test_tiling_the_queries_does_not_change_the_answer(rows):
    """Exact, because a query's output depends on no other query. Odd tile counts are
    where a slice bound or the reused scratch buffer shows up; small ones land on the
    ``MIN_TILE_ROWS`` floor.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 601, 24) for _ in range(3))
    whole = single_head_attention(q, k, v, rows=q.shape[-2])
    tiled = single_head_attention(q, k, v, rows=rows)
    assert tiled.shape == whole.shape and tiled.dtype == whole.dtype
    assert torch.allclose(tiled, whole, atol=1e-6, rtol=1e-5)


@pytest.fixture
def tiny_flux():
    """The shipped Flux decoder at 8-channel blocks; its mid block is an attention block."""
    return AutoencoderKLFlux(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                             layers_per_block=0).eval().requires_grad_(False)


def test_flux_decode_never_reaches_for_sdpa(tiny_flux, monkeypatch):
    """Whatever a fused backend does on a given ROCm build, a decode must not inherit it."""
    def no_backend(*args, **kwargs):
        raise AssertionError("the VAE decode asked for an SDPA backend")

    monkeypatch.setattr(F, "scaled_dot_product_attention", no_backend)
    with torch.no_grad():
        pixels = tiny_flux.decode_to_pixels(torch.randn(1, 16, 4, 4))
    assert pixels.shape == (1, 3, 32, 32)


def _reference_patched_attention(q, k, v):
    """Mage-Flow's windowed block, per Comfy-Org/ComfyUI ``comfy/ldm/mage_flow/vae.py``."""
    b, c, length = q.shape
    q, k, v = (t.view(b, 1, c, length).transpose(2, 3).contiguous() for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(b, c, length)


def test_mage_flow_attention_attends_over_positions_not_memory():
    """``_attention`` takes ``(B, C, L)`` and must transpose into ``(…, L, C)``.

    ``view(b, 1, L, C)`` reaches that shape by re-reading the buffer instead, which is a
    different tensor whenever ``C != L``: attention over a permuted reading of the
    window. Entry and exit were both that mistake, and they looked like they cancelled.
    """
    q, k, v = (torch.randn(2, 6, 16) for _ in range(3))  # C != L: the case that can tell
    out = mage_attention(q, k, v)
    assert out.shape == q.shape
    assert torch.allclose(out, _reference_patched_attention(q, k, v), atol=1e-5)

    perm = torch.tensor([13, 0, 7, 4, 15, 2, 9, 1, 12, 6, 11, 3, 14, 8, 5, 10])
    assert torch.allclose(mage_attention(q[..., perm], k[..., perm], v[..., perm]),
                          out[..., perm], atol=1e-5)


def test_flux2_attention_matches_upstream():
    """Positions are the tokens and channels the features (BFL and kohya both say so).

    A view into ``(b, 1, h*w, c)`` looks like upstream's rearrange and is not: it
    re-reads the channel-major buffer, and the scale then follows the pixel count
    rather than the channel count.
    """
    torch.manual_seed(0)
    block = _AttnBlock(32).eval().requires_grad_(False)
    x = torch.randn(1, 32, 5, 3)  # 15 positions, 32 channels

    def upstream(x):
        """BFL/kohya ``AttnBlock.attention``, transcribed: rearrange, sdpa, rearrange back."""
        h_ = block.norm(x)
        q, k, v = (op(h_).flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
                   for op in (block.q, block.k, block.v))
        return F.scaled_dot_product_attention(q, k, v).squeeze(1).transpose(1, 2).reshape(x.shape)

    with torch.no_grad():
        out = block.attention(x)
        assert torch.allclose(out, upstream(x), atol=1e-6)

        perm = torch.randperm(15)
        shuffled = x.flatten(2)[..., perm].reshape(x.shape)
        assert torch.allclose(block.attention(shuffled),
                              out.flatten(2)[..., perm].reshape(x.shape), atol=1e-6)
