"""MOSS-Image DiT — one joint stream, a refined text front-end, split-half RoPE.

  * One token is one latent cell: the DiT takes the canonical ``[B, 128, H, W]`` and
    answers with the velocity in the same layout, so the adapter packs nothing.
  * **Every** token runs in one stream. The prompt is refined by a small front-end (a
    projection plus two attention blocks), concatenated to the image tokens, and the
    joint blocks modulate the whole sequence with one shared adaLN row set.
  * The timestep conditions nothing but those modulations, so the front-end is
    **step-invariant**: it runs once per run, per CFG branch
    (:meth:`MossImage.text_states`).
  * RoPE is the **split-half** convention over four ``(r, f, h, w)`` axes (see
    :func:`thenoise.utils.rope.split_half_rope_4d`). The image sits *after* the text
    on ``f`` — its whole grid shares ``f = text_length`` — so a branch's rotary table
    depends on its own prompt length: the CFG branches cannot share one table, and
    neither can be padded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import attention as sdpa_attention
from thenoise.utils.dynamo import mark_token_axis
from thenoise.utils.positions import grid_from_axes
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rope import RopeCache, apply_rope_split_half, split_half_rope_4d
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.timestep import timestep_embedding

NORM_EPS = 1e-6

#: Not a knob: the checkpoint carries this many front-end blocks.
TEXT_DEPTH = 2


@dataclass
class MossImageParams:
    """Architecture knobs, read back off the checkpoint by ``detect_params``."""

    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096  # the Qwen3.5-9B hidden size
    latent_channels: int = 128
    freq_dim: int = 256
    rope_base: float = 10000.0


#: The two caps of :func:`situ`: ``4`` on the gate, ``25`` on the hidden factor.
SITU_CAP = 100.0


def situ(hidden: Tensor, gate: Tensor) -> Tensor:
    """Soft-capped gated activation: both factors are ``tanh``-limited before gating.

    The joint blocks use :func:`situ_unscaled` and carry :data:`SITU_CAP` on their own
    gate instead, which is one fewer pass over the ffn-width tensor.
    """
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def situ_unscaled(hidden: Tensor, gate: Tensor) -> Tensor:
    """``situ`` without :data:`SITU_CAP`, for a caller that can carry it for free."""
    return torch.tanh(gate / 4.0) * torch.sigmoid(gate) * torch.tanh(hidden / 25.0)


def modulated_norm(norm: nn.Module, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """``(1 + scale) * norm(x) + shift``, with ``[B, 1, dim]`` rows over the tokens."""
    return norm(x) * (1 + scale) + shift


def attention(
    q: Tensor, k: Tensor, v: Tensor, qk_norm: QKNorm, pe: Tuple[Tensor, Tensor], heads: int
) -> Tensor:
    """Normalize the heads, rotate them, attend: ``[B, L, dim]`` in and out.

    Never masked — the sequence is unpadded, each CFG branch carrying its own length.
    """
    batch, length, _ = q.shape
    q = q.view(batch, length, heads, -1).transpose(1, 2)
    k = k.view(batch, length, heads, -1).transpose(1, 2)
    v = v.view(batch, length, heads, -1).transpose(1, 2)

    q, k = qk_norm(q, k)
    cos, sin = pe
    q, k = apply_rope_split_half(q, cos, sin), apply_rope_split_half(k, cos, sin)
    return sdpa_attention([q, k, v])


class TextBlock(nn.Module):
    """Front-end block: attention + SiTU MLP, both plain residual (no adaLN)."""

    def __init__(self, dim: int, ffn_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSNorm(dim, eps=NORM_EPS)
        self.norm2 = RMSNorm(dim, eps=NORM_EPS)
        self.qk_norm = QKNorm(dim // num_heads, eps=NORM_EPS)
        self.qkv = QuantizedLinear(dim, 3 * dim)
        self.proj = QuantizedLinear(dim, dim)
        self.ffn_up = QuantizedLinear(dim, ffn_dim)
        self.ffn_gate = QuantizedLinear(dim, ffn_dim)
        self.ffn_down = QuantizedLinear(ffn_dim, dim)

    def forward(self, x: Tensor, pe: Tuple[Tensor, Tensor]) -> Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(attention(q, k, v, self.qk_norm, pe, self.num_heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextEmbedder(nn.Module):
    """Prompt embeddings -> the tokens the joint blocks see."""

    def __init__(self, text_dim: int, dim: int, ffn_dim: int, num_heads: int) -> None:
        super().__init__()
        self.proj = QuantizedLinear(text_dim, dim)
        self.blocks = nn.ModuleList(
            [TextBlock(dim, ffn_dim, num_heads) for _ in range(TEXT_DEPTH)]
        )

    def forward(self, x: Tensor, pe: Tuple[Tensor, Tensor]) -> Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, pe)
        return x


class Block(nn.Module):
    """One joint block: modulated attention + a modulated SiTU MLP."""

    def __init__(self, dim: int, ffn_dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(
                f"dim {dim} must be divisible by num_heads {num_heads}"
            )
        self.num_heads = num_heads
        self.norm1 = RMSNorm(dim, eps=NORM_EPS)
        self.norm2 = RMSNorm(dim, eps=NORM_EPS)
        self.qk_norm = QKNorm(dim // num_heads, eps=NORM_EPS)
        # The nesting mirrors the checkpoint's; q/k/v arrive fused.
        self.proj = nn.ModuleDict(
            {
                "qkv": QuantizedLinear(dim, 3 * dim, bias=False),
                "o": QuantizedLinear(dim, dim, bias=False),
                "ffn_up_gate": QuantizedLinear(dim, 2 * ffn_dim, bias=False),
                "ffn_down": QuantizedLinear(ffn_dim, dim, bias=False),
            }
        )

    @torch.compile(fullgraph=True)
    def forward(
        self, x: Tensor, mod: Tuple[Tensor, ...], pe: Tuple[Tensor, Tensor]
    ) -> Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        y = modulated_norm(self.norm1, x, scale_a, shift_a)
        q, k, v = self.proj["qkv"](y).chunk(3, dim=-1)
        out = attention(q, k, v, self.qk_norm, pe, self.num_heads)
        x = x + gate_a * self.proj["o"](out)

        y = modulated_norm(self.norm2, x, scale_f, shift_f)
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, dim=-1)
        # situ's cap goes on the gate instead: legal because ffn_down has no bias.
        out = self.proj["ffn_down"](situ_unscaled(hidden, gate))
        return x + (gate_f * SITU_CAP) * out


class MossImage(nn.Module):
    """The MOSS-Image transformer: text front-end, joint blocks, image-only head.

    :meth:`text_states` runs once per CFG branch; every step after that is
    :meth:`forward`, which takes its output and that branch's own rotary table.
    """

    def __init__(self, params: MossImageParams) -> None:
        super().__init__()
        head_dim = params.dim // params.num_heads
        if head_dim % 8:
            raise ValueError(
                f"MOSS-Image splits the head dim into 4 RoPE axes of 2 groups each, "
                f"so head_dim must divide by 8, got {head_dim}"
            )
        if params.freq_dim % 2:
            raise ValueError(f"the timestep sinusoid needs an even freq_dim, got {params.freq_dim}")

        self.latent_channels = params.latent_channels
        self.freq_dim = params.freq_dim
        self.pe_embedder = RopeCache(split_half_rope_4d(head_dim, params.rope_base))

        self.video_embedder = QuantizedLinear(params.latent_channels, params.dim)
        self.text_embedder = TextEmbedder(
            params.text_dim, params.dim, params.ffn_dim, params.num_heads
        )
        self.time_embedding = nn.Sequential(
            QuantizedLinear(params.freq_dim, params.dim),
            nn.SiLU(),
            QuantizedLinear(params.dim, params.dim),
        )
        # Six rows: the attention and MLP shift/scale/gate triplets, in that order.
        self.shared_block_modulation = QuantizedLinear(params.dim, 6 * params.dim)
        self.layers = nn.ModuleList(
            [Block(params.dim, params.ffn_dim, params.num_heads) for _ in range(params.num_layers)]
        )
        self.final_norm_video = RMSNorm(params.dim, eps=NORM_EPS)
        self.final_linear_video = QuantizedLinear(params.dim, params.latent_channels, bias=False)
        self.video_head_modulation = QuantizedLinear(params.dim, 2 * params.dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def text_states(self, context: Tensor, pe_text: Tuple[Tensor, Tensor]) -> Tensor:
        """Prompt embeddings ``[B, L, text_dim]`` -> the ``[B, L, dim]`` joint tokens."""
        return self.text_embedder(context, pe_text)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        txt_states: Tensor,
        pe: Tuple[Tensor, Tensor],
    ) -> Tensor:
        """The velocity of the whole latent, ``[B, latent, H, W]``.

        ``pe`` is this branch's rotary table over the **joint** ``[image, text]``
        sequence and ``txt_states`` its own front-end output.
        """
        batch, _, height, width = x.shape
        num_img = height * width

        vec = self.time_embedding(timestep_embedding(timestep, self.freq_dim))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, 1, -1)).unbind(1)
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(
            -1, (2, 1, -1)
        ).unbind(1)

        video = self.video_embedder(x.flatten(2).transpose(1, 2))
        tokens = torch.cat((video, txt_states), dim=1)
        for block in self.layers:
            mark_token_axis(tokens, *pe)
            tokens = block(tokens, mod, pe)

        # Only the image predicts: the text tokens leave the sequence here.
        out = modulated_norm(
            self.final_norm_video, tokens[:, :num_img], head_scale, head_shift
        )
        out = self.final_linear_video(out)
        return out.transpose(1, 2).reshape(batch, self.latent_channels, height, width)


# --------------------------------------------------------------- positions


def joint_positions(
    height: int,
    width: int,
    text_length: int,
    *,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> Tensor:
    """``[H*W + text_length, 4]`` ``(r, f, h, w)`` coordinates, image tokens first.

    The image grid is centred on ``0`` and shares one ``f``, ``text_length``; the text
    runs ``f = 0 .. text_length-1`` with no spatial position. ``r`` is the video axis,
    which this model never uses.
    """
    image = grid_from_axes(
        [
            torch.zeros(1, dtype=dtype, device=device),
            torch.full((1,), text_length, dtype=dtype, device=device),
            torch.arange(height, dtype=dtype, device=device) - (height - 1) / 2,
            torch.arange(width, dtype=dtype, device=device) - (width - 1) / 2,
        ]
    )
    text = torch.zeros(text_length, 4, dtype=dtype, device=device)
    text[:, 1] = torch.arange(text_length, dtype=dtype, device=device)
    return torch.cat((image, text), dim=0)


__all__ = [
    "NORM_EPS",
    "SITU_CAP",
    "TEXT_DEPTH",
    "Block",
    "MossImage",
    "MossImageParams",
    "TextBlock",
    "TextEmbedder",
    "attention",
    "joint_positions",
    "modulated_norm",
    "situ",
    "situ_unscaled",
]
