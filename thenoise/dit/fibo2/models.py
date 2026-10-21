"""Fibo 2.0 DiT — one stream of gist, image and reference tokens, text by cross-attention.

The Z-Image block (fused QKV self-attention with per-head QK-norm, a SwiGLU MLP, four
AdaLN rows) with three things bolted on:

  * One token is one latent cell: the DiT takes the canonical ``[B, 128, H, W]`` latent
    and answers with the velocity in the same layout, so the adapter packs nothing.
  * **No prompt token enters the stream.** A Perceiver squeezes the first of the six
    text bundles into gist tokens, and the five remaining bundles are cross-attended
    by one injection block each — so the sequence length is the image's, plus a gist
    count that follows the prompt length (:func:`gist_count`).
  * The prompt and the references are **clean**: AdaLN carries two rows, ``t = 0`` and
    ``t``, and the noisy image span is the only part read at ``t``. Which tokens that
    is is a property of the layout, so every block is told the span (:func:`scale_tokens`).

The RoPE ``t`` axis keeps the streams apart: gist on ``1 .. num_gist``, the image on a
plane past every possible gist count, each reference on the plane after it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from thenoise.dit.lumina.models import Attention as RotatedAttention
from thenoise.dit.lumina.models import FinalLayer, TimestepEmbedder
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import attention
from thenoise.utils.positions import grid_from_axes, grid_positions
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rope import RopeCache, matrix_rope
from thenoise.utils.rms_norm import RMSNorm

#: A gist run of 256 tokens covers a prompt of this many tokens; a prompt of
#: ``GIST_FULL_LENGTH`` gets every token the Perceiver can emit.
GIST_MIN_TOKENS = 256
GIST_FULL_LENGTH = 2048
#: The count only moves in whole blocks of queries.
GIST_STEP = 64


@dataclass
class Fibo2Params:
    """Architecture knobs, read back off the checkpoint by ``detect_params``.

    The rotary split, ``norm_eps`` and the flow shift are not in the file: they are
    the released model's constants.
    """

    dim: int = 3584
    ffn_dim: int = 9728
    num_heads: int = 28
    num_layers: int = 44
    num_refiner_layers: int = 2
    perceiver_layers: int = 2
    #: Main blocks that cross-attend to text, in order of the bundle they read.
    injection_layers: Tuple[int, ...] = (6, 14, 22, 30, 38)
    latent_channels: int = 128
    #: One text bundle: three tapped Qwen3-VL layers, concatenated.
    cap_dim: int = 7680
    max_gist_tokens: int = 512
    #: AdaLN input width, the timestep MLP's own width, and its sinusoid width.
    adaln_dim: int = 256
    t_embed_dim: int = 1024
    freq_dim: int = 256
    #: ``w1``/``w3`` stored as one ``w13`` (the release) rather than apart.
    fused_ffn: bool = True
    norm_eps: float = 1e-5
    rope_theta: float = 256.0
    axes_dims: Tuple[int, ...] = (32, 48, 48)

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def num_bundles(self) -> int:
        """Text bundles: the Perceiver's, plus one per injection block."""
        return 1 + len(self.injection_layers)


def gist_count(text_length: int, max_gist: int) -> int:
    """Gist tokens to spend on a prompt of ``text_length`` tokens.

    ``GIST_MIN_TOKENS`` up to ``GIST_MIN_TOKENS`` tokens, then rising linearly to
    ``max_gist`` at ``GIST_FULL_LENGTH``, rounded up to :data:`GIST_STEP`. It is a
    function of the prompt length alone, which is why no branch can be padded.
    """
    ratio = min(1.0, max(0.0, (text_length - GIST_MIN_TOKENS)
                         / (GIST_FULL_LENGTH - GIST_MIN_TOKENS)))
    want = GIST_MIN_TOKENS + ratio * (max_gist - GIST_MIN_TOKENS)
    return max(GIST_MIN_TOKENS, min(max_gist, math.ceil(want / GIST_STEP) * GIST_STEP))


def scale_tokens(y: Tensor, rows: Tensor, span: Optional[Tuple[int, int]] = None) -> Tensor:
    """``y`` ``[B, L, dim]`` scaled per token from AdaLN ``rows`` of ``[2B, 1, dim]``.

    The AdaLN embedding is computed over a batch of clean and noisy timesteps stacked
    along the batch axis, and ``span`` is the token range the noisy half scales while
    the clean half scales the rest: the image reads ``t``, the gist and the reference
    tokens read ``t = 0``. With no span — the noise refiner of a run with no
    references, fed the noisy half alone — every token gets the same rows.
    """
    if span is None:
        return y * rows
    batch = y.shape[0]
    start, stop = span
    return torch.cat(
        (
            y[:, :start] * rows[:batch],
            y[:, start:stop] * rows[batch:],
            y[:, stop:] * rows[:batch],
        ),
        dim=1,
    )


class Attention(nn.Module):
    """Unrotated multi-head attention with separate query and context streams.

    Neither the cross-attention to a text bundle nor the Perceiver's self-attention is
    given positions: order and content carry the meaning, not position.
    """

    def __init__(self, dim: int, num_heads: int, kv_dim: int, eps: float) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.to_q = QuantizedLinear(dim, dim, bias=False)
        self.to_kv = QuantizedLinear(kv_dim, 2 * dim, bias=False)
        self.to_out = QuantizedLinear(dim, dim, bias=False)
        self.qk_norm = QKNorm(dim // num_heads, eps=eps)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        heads = self.num_heads
        q = self.to_q(x).unflatten(-1, (heads, -1)).transpose(1, 2)
        # ``to_kv`` stacks the key rows over the value rows.
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.unflatten(-1, (heads, -1)).transpose(1, 2)
        v = v.unflatten(-1, (heads, -1)).transpose(1, 2)
        q, k = self.qk_norm(q, k)
        return self.to_out(attention([q, k, v]))


class FeedForward(nn.Module):
    """SwiGLU.

    The release fuses ``w1``/``w3`` into ``w13`` and zero-pads the hidden width to
    :data:`Fibo2Params.ffn_dim`, which is what lets the int8 export quantize the
    activation inside ``w2``'s GEMM.
    """

    def __init__(self, dim: int, hidden_dim: int, fused: bool) -> None:
        super().__init__()
        self.fused = fused
        if fused:
            self.w13 = QuantizedLinear(dim, 2 * hidden_dim, bias=False)
        else:
            self.w1 = QuantizedLinear(dim, hidden_dim, bias=False)
            self.w3 = QuantizedLinear(dim, hidden_dim, bias=False)
        self.w2 = QuantizedLinear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.fused:
            gate, up = self.w13(x).chunk(2, dim=-1)
        else:
            gate, up = self.w1(x), self.w3(x)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    """One stream block: rotated self-attention, optionally a text cross-attention, MLP.

    ``cap_dim`` is the cross-attention's context width, so a block that is not an
    injection block simply has no cross-attention to carry. The residuals are added out
    of place: the reference writes them into the buffer it is handed, inside a memory
    compiler region that has no counterpart here.
    """

    def __init__(self, params: Fibo2Params, cap_dim: Optional[int] = None) -> None:
        super().__init__()
        dim, heads, eps = params.dim, params.num_heads, params.norm_eps
        self.attention = RotatedAttention(dim, heads, eps)
        self.feed_forward = FeedForward(dim, params.ffn_dim, params.fused_ffn)
        self.attention_norm1 = RMSNorm(dim, eps=eps)
        self.ffn_norm1 = RMSNorm(dim, eps=eps)
        self.attention_norm2 = RMSNorm(dim, eps=eps)
        self.ffn_norm2 = RMSNorm(dim, eps=eps)
        self.adaLN_modulation = nn.Sequential(
            QuantizedLinear(params.adaln_dim, 4 * dim, bias=True)
        )
        self.cross_attention = None
        if cap_dim is not None:
            self.cross_attn_norm1 = RMSNorm(dim, eps=eps)
            self.cross_attn_norm2 = RMSNorm(dim, eps=eps)
            self.cross_attention = Attention(dim, heads, cap_dim, eps)
            # Closed rather than uninitialized until a checkpoint gives it a value.
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    @torch.compile(fullgraph=True)
    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        adaln: Tensor,
        span: Optional[Tuple[int, int]] = None,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        scale_attn, gate_attn, scale_ffn, gate_ffn = self.adaLN_modulation(adaln).chunk(
            4, dim=-1
        )

        y = self.attention(
            scale_tokens(self.attention_norm1(x), (1.0 + scale_attn).unsqueeze(1), span),
            freqs_cis=pe,
        )
        x = x + scale_tokens(self.attention_norm2(y), gate_attn.tanh().unsqueeze(1), span)

        if self.cross_attention is not None:
            y = self.cross_attention(self.cross_attn_norm1(x), context)
            x = x + self.cross_attn_norm2(y) * self.cross_attn_gate.tanh()

        y = self.feed_forward(
            scale_tokens(self.ffn_norm1(x), (1.0 + scale_ffn).unsqueeze(1), span)
        )
        return x + scale_tokens(self.ffn_norm2(y), gate_ffn.tanh().unsqueeze(1), span)


class PerceiverLayer(nn.Module):
    """Gist queries read the prompt, then each other, then their MLP."""

    def __init__(self, params: Fibo2Params) -> None:
        super().__init__()
        dim, heads, eps = params.dim, params.num_heads, params.norm_eps
        self.cross_attn_norm = RMSNorm(dim, eps=eps)
        self.cross_attention = Attention(dim, heads, params.cap_dim, eps)
        self.self_attn_norm = RMSNorm(dim, eps=eps)
        self.self_attention = Attention(dim, heads, dim, eps)
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.ffn = FeedForward(dim, params.ffn_dim, params.fused_ffn)

    @torch.compile(fullgraph=True)
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = x + self.cross_attention(self.cross_attn_norm(x), context)
        y = self.self_attn_norm(x)
        x = x + self.self_attention(y, y)
        return x + self.ffn(self.ffn_norm(x))


class Perceiver(nn.Module):
    """Prompt bundle -> gist tokens, through ``num_layers`` Perceiver layers."""

    def __init__(self, params: Fibo2Params) -> None:
        super().__init__()
        # Zeroed for the same reason as a cross-attention gate: these are shipped
        # weights, and a module that never loaded one should be inert, not uninitialized.
        self.latent_queries = nn.Parameter(torch.zeros(params.max_gist_tokens, params.dim))
        self.layers = nn.ModuleList(
            [PerceiverLayer(params) for _ in range(params.perceiver_layers)]
        )
        self.norm_out = RMSNorm(params.dim, eps=params.norm_eps)

    def forward(self, context: Tensor, num_gist: int) -> Tensor:
        x = self.latent_queries[:num_gist].unsqueeze(0).expand(context.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, context)
        return self.norm_out(x)


class Fibo2Transformer2DModel(nn.Module):
    """The Fibo 2.0 transformer: two noise refiners, the Perceiver, the main blocks.

    :meth:`gist_tokens` runs once per CFG branch; every step after that is
    :meth:`forward`, which takes that branch's own gist and rotary table.
    """

    def __init__(self, params: Fibo2Params) -> None:
        super().__init__()
        if sum(params.axes_dims) != params.head_dim:
            raise ValueError(
                f"the rope split must cover the head dim: {params.axes_dims} sums to "
                f"{sum(params.axes_dims)}, head dim is {params.head_dim}"
            )
        bad = [d for d in params.axes_dims if d % 2]
        if bad:
            raise ValueError(f"rope axis dims must be even, got {bad}")

        self.params = params
        self.latent_channels = params.latent_channels
        # block index -> the text bundle it cross-attends to; bundle 0 is the Perceiver's
        self._bundle_of = {i: n + 1 for n, i in enumerate(params.injection_layers)}

        self.pe_embedder = RopeCache(matrix_rope(list(params.axes_dims), params.rope_theta))
        self.x_embedder = QuantizedLinear(params.latent_channels, params.dim)
        self.t_embedder = TimestepEmbedder(
            params.adaln_dim, mid_size=params.t_embed_dim,
            frequency_embedding_size=params.freq_dim,
        )
        self.noise_refiner = nn.ModuleList(
            [Block(params) for _ in range(params.num_refiner_layers)]
        )
        self.perceiver = Perceiver(params)
        self.layers = nn.ModuleList(
            [
                Block(params, params.cap_dim if i in self._bundle_of else None)
                for i in range(params.num_layers)
            ]
        )
        self.final_layer = FinalLayer(params.dim, params.latent_channels)

    def gist_tokens(self, context: Tensor) -> Tensor:
        """The Perceiver's read-out of the prompt's first bundle: ``[B, num_gist, dim]``.

        Nothing about it depends on the timestep, so the adapter runs it once per
        branch — and ``num_gist`` comes out of it, since the count follows the length.
        """
        bundle = context.chunk(self.params.num_bundles, dim=-1)[0]
        return self.perceiver(
            bundle, gist_count(context.shape[1], self.params.max_gist_tokens)
        )

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        context: Tensor,
        gist: Tensor,
        pe: Tensor,
        refs: Sequence[Tensor] = (),
    ) -> Tensor:
        """The velocity of the whole latent, ``[B, latent, H, W]``.

        ``context`` is the prompt's six text bundles concatenated, ``gist`` this
        branch's Perceiver output for it and ``pe`` the rotary table over the whole
        ``[gist, image, references]`` sequence. ``refs`` are reference latents in the
        same canonical layout as ``x``.
        """
        batch, channels, height, width = x.shape
        num_img = height * width
        num_gist = gist.shape[1]
        bundles = context.chunk(self.params.num_bundles, dim=-1)

        # Two AdaLN rows: the clean tokens read the first, the noisy image the second.
        adaln = self.t_embedder(torch.cat((torch.zeros_like(timestep), timestep)))

        tokens = [x.flatten(2).transpose(1, 2)]
        tokens += [ref.flatten(2).transpose(1, 2) for ref in refs]
        img = self.x_embedder(torch.cat(tokens, dim=1))

        if refs:
            refiner_adaln, refiner_span = adaln, (0, num_img)
        else:
            refiner_adaln, refiner_span = adaln[batch:], None
        for block in self.noise_refiner:
            img = block(img, pe[:, num_gist:], refiner_adaln, refiner_span)

        x = torch.cat((gist, img), dim=1)
        del img
        noisy_span = (num_gist, num_gist + num_img)
        for i, block in enumerate(self.layers):
            bundle = self._bundle_of.get(i)
            x = block(
                x, pe, adaln, noisy_span, None if bundle is None else bundles[bundle]
            )

        out = self.final_layer(x[:, num_gist:num_gist + num_img], adaln[batch:])
        return out.transpose(1, 2).reshape(batch, channels, height, width)


# --------------------------------------------------------------- positions


def sequence_positions(
    num_gist: int,
    height: int,
    width: int,
    ref_sizes: Sequence[Tuple[int, int]],
    max_gist: int,
    *,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> Tensor:
    """``[1, num_gist + H*W + refs, 3]`` ``(t, h, w)`` coordinates of one branch's sequence.

    One row per token of :meth:`Fibo2Transformer2DModel.forward`, batched because that
    is the layout the rotary table is built from: a run has one sequence per branch.

    The RoPE ``t`` axis is what tells the streams apart: gist tokens ride ``1`` to
    ``num_gist`` with no spatial position, the image sits on ``max_gist + 1`` — the
    plane past every gist count the model can emit, so a shorter prompt does not move
    it — and reference ``k`` on the plane after it. Spatial axes start at ``0``.
    """
    zero = torch.zeros(1, dtype=dtype, device=device)
    streams = [
        grid_from_axes(
            [torch.arange(1, num_gist + 1, dtype=dtype, device=device), zero, zero]
        )
    ]
    plane = max_gist + 1
    streams.append(
        grid_positions([1, height, width], start=[plane, 0, 0], dtype=dtype, device=device)
    )
    for k, (ref_h, ref_w) in enumerate(ref_sizes):
        streams.append(
            grid_positions(
                [1, ref_h, ref_w], start=[plane + 1 + k, 0, 0], dtype=dtype, device=device
            )
        )
    return torch.cat(streams, dim=0).unsqueeze(0)


__all__ = [
    "GIST_FULL_LENGTH",
    "GIST_MIN_TOKENS",
    "GIST_STEP",
    "Attention",
    "Block",
    "Fibo2Params",
    "Fibo2Transformer2DModel",
    "FeedForward",
    "Perceiver",
    "PerceiverLayer",
    "gist_count",
    "scale_tokens",
    "sequence_positions",
]
