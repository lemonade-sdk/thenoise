"""Flux.3 DiT architecture — one block type, three token *modes*.

Follows Comfy-Org/ComfyUI's ``comfy/ldm/flux3/model.py`` (the reference Flux.3
implementation) with this repo's conventions: quantized linears, the shared RoPE
cache/attention/timestep helpers, and the early-block outputs handed in by the
adapter so the step-invariant streams run once per run instead of once per step.

What makes Flux.3 different from Flux.2 (``dit/flux2``):

  * **one** block, in which attention and a SwiGLU MLP run *in parallel* under a
    single adaLN and a single gate — no double stream, no fused ``linear1``;
  * tokens belong to a **mode** — ``image`` (the target), ``image_cond`` (reference
    latents) or ``txt`` — and each mode has its own input embedding, its own
    ``depth`` *early* blocks that attend only within the mode, and its own
    modulation row in the ``depth_single`` joint blocks;
  * text and references are modulated at **timestep zero**, the target at ``t``;
  * a **flow map** (``flow_map=True``, signalled by ``ref_time_in``) additionally
    embeds the timestep each step *jumps to* into every modulation, distilled models
    being step-conditioned rather than time-conditioned alone.

A block never sees a whole-sequence modulation: the joint sequence is described by
``Span``\\ s, each an exclusive token range plus that mode's ``(shift, scale, gate)``,
so a mode's rows are modulated and gated by its own vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Sequence

import torch
from einops import rearrange
from torch import Tensor, nn

from thenoise.dit.flux2.models import (
    LastLayer,
    MLPEmbedder,
    Modulation,
    SiLUActivation,
)
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import attention as sdpa_attention
from thenoise.utils.dynamo import mark_token_axis
from thenoise.utils.positions import grid_from_axes
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rope import RopeCache, apply_rope, matrix_rope
from thenoise.utils.setup_logging import setup_logging
from thenoise.utils.timestep import timestep_embedding

setup_logging()
import logging

logger = logging.getLogger(__name__)

#: The content modes and the text mode. The mode name is part of the checkpoint's key
#: paths (``emb_in.image``, ``single_stream_modulations.txt``, ...), so these strings
#: are the weight layout.
CONTENT_MODES = ("image", "image_cond")
TEXT_MODE = "txt"

#: Flux.3 scales the image row/column position axes by a half, and separates the
#: target (0) from its references (-1, -2, ...) on the ``l`` axis.
IMAGE_ROPE_SCALE = 0.5


@dataclass
class Flux3Params:
    """Flux.3 architecture knobs (the released ``f3-image-small`` values)."""

    in_channels: int = 128  # Flux.2 packed latent channels
    context_in_dim: int = 20480  # 8 Qwen3-VL-4B taps x 2560 hidden
    hidden_size: int = 3072
    num_heads: int = 24  # head_dim is 128 in every released variant
    depth: int = 5  # early blocks, per mode
    depth_single_blocks: int = 28  # joint blocks
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 10000
    mlp_ratio: float = 3.0
    flow_map: bool = False  # a step-distilled flow map embeds each step's target time
    fused_qkv: bool = True  # how the CHECKPOINT stores q/k/v (this module always fuses)


class ModOut(NamedTuple):
    """One mode's adaLN triplet, as a ``[1, 1, hidden]`` row broadcast over its tokens."""

    shift: Tensor
    scale: Tensor
    gate: Tensor


class Span(NamedTuple):
    """``[start, end)`` of the token sequence and the modulation driving those rows."""

    start: int
    end: int
    mod: ModOut


def attention(tokens: Sequence[Tensor], pe: Tensor) -> Tensor:
    """Apply RoPE, then attend through :mod:`thenoise.utils.attention`.

    Output is token-major ``[B, L, H*D]``. Flux.3 attention is never masked: every
    token of every mode reads every other one, which is why the text stream carries
    its padding into the DiT (see ``dit/flux3/encoder.py``).
    """
    q, k, v = tokens
    q, k = apply_rope(q, k, pe)
    return sdpa_attention([q, k, v])


def _modulate(x: Tensor, spans: list[Span], norm: nn.Module) -> Tensor:
    """Pre-norm + adaLN, mode by mode (ComfyUI's ``modulated_norm`` per span)."""
    if len(spans) == 1:
        shift, scale, _ = spans[0].mod
        return (1 + scale) * norm(x) + shift
    parts = []
    for start, end, (shift, scale, _) in spans:
        parts.append((1 + scale) * norm(x[:, start:end]) + shift)
    return torch.cat(parts, dim=1)


def _gated(x: Tensor, out: Tensor, spans: list[Span]) -> Tensor:
    """``x + gate * out``, one gate row per mode."""
    if len(spans) == 1:
        return x + spans[0].mod.gate * out
    return x + torch.cat(
        [span.mod.gate * out[:, span.start:span.end] for span in spans], dim=1
    )


class Block(nn.Module):
    """One Flux.3 block: parallel attention + SwiGLU MLP under one adaLN and one gate."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 3.0):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        self.num_heads = num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Fused even for checkpoints that store the three projections apart; the fold
        # happens at load time (see ``dit/flux3/utils.py``).
        self.qkv = QuantizedLinear(hidden_size, 3 * hidden_size, bias=False)
        self.mlp_in = QuantizedLinear(hidden_size, 2 * mlp_hidden_dim, bias=False)
        self.attn_out = QuantizedLinear(hidden_size, hidden_size, bias=False)
        self.mlp_out = QuantizedLinear(mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(hidden_size // num_heads, eps=1e-6)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    @torch.compile(fullgraph=False)
    def forward(self, x: Tensor, spans: list[Span], pe: Tensor) -> Tensor:
        x_mod = _modulate(x, spans, self.pre_norm)
        q, k, v = rearrange(self.qkv(x_mod), "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        mlp = self.mlp_in(x_mod)
        q, k = self.norm(q, k)

        out = self.attn_out(attention([q, k, v], pe))
        # ComfyUI folds the two into one GEMM epilogue; the sum is the same residual.
        out = out + self.mlp_out(self.mlp_act(mlp))
        return _gated(x, out, spans)


class Flux3(nn.Module):
    """The Flux.3 transformer: per-mode early blocks, then joint blocks, then one head.

    The three streams are fed in already tokenized. Call the pieces around
    ``forward`` rather than expecting one mega-call:

      * :meth:`timestep_vec` — the ``[2, hidden]`` adaLN source (row 0 = ``t``,
        row 1 = timestep zero), plus the flow map's target time;
      * :meth:`text_states` / :meth:`reference_states` — the timestep-zero streams;
        step-invariant unless the DiT is a flow map, so the adapter runs them once;
      * :meth:`forward` — the target's early blocks, the joint blocks and the head.
    """

    def __init__(self, params: Flux3Params) -> None:
        super().__init__()
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.flow_map = params.flow_map
        head_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != head_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {head_dim}")

        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = RopeCache(matrix_rope(params.axes_dim, params.theta))

        self.emb_in = nn.ModuleDict(
            {m: QuantizedLinear(self.in_channels, params.hidden_size, bias=False)
             for m in CONTENT_MODES}
        )
        self.txt_in = QuantizedLinear(params.context_in_dim, params.hidden_size, bias=False)
        self.time_in = MLPEmbedder(256, params.hidden_size, disable_bias=True)
        if self.flow_map:
            self.ref_time_in = MLPEmbedder(256, params.hidden_size, disable_bias=True)

        modulation = lambda: Modulation(params.hidden_size, double=False, disable_bias=True)
        self.early_stream_modulations = nn.ModuleDict(
            {m: modulation() for m in CONTENT_MODES + (TEXT_MODE,)}
        )
        self.single_stream_modulations = nn.ModuleDict(
            {m: modulation() for m in CONTENT_MODES + (TEXT_MODE,)}
        )

        block = lambda: Block(params.hidden_size, params.num_heads, params.mlp_ratio)
        self.content_mode_blocks = nn.ModuleDict(
            {m: nn.ModuleList([block() for _ in range(params.depth)]) for m in CONTENT_MODES}
        )
        self.txt_mode_blocks = nn.ModuleList([block() for _ in range(params.depth)])
        self.single_blocks = nn.ModuleList(
            [block() for _ in range(params.depth_single_blocks)]
        )

        # Only the target predicts; the mode key is part of the checkpoint layout.
        self.final_layer = nn.ModuleDict(
            {"image": LastLayer(params.hidden_size, self.out_channels)}
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # ------------------------------------------------------------ conditioning
    def timestep_vec(self, timestep: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """``[2, hidden]`` adaLN source: row 0 is ``timestep``, row 1 is timestep zero.

        A flow map adds the time the step *jumps to* to **both** rows, which is what
        makes even its timestep-zero streams change every step.
        """
        # ``timestep_embedding`` upcasts to fp32; the vector joins the blocks in the
        # compute dtype, which is the dtype the caller's ``t`` arrives in.
        vec = self.time_in(
            timestep_embedding(torch.cat([timestep, timestep.new_zeros(1)]), 256).to(timestep.dtype)
        )
        if self.flow_map:
            if target is None:
                raise ValueError(
                    "a flow-map Flux.3 modulates every stream with the timestep this "
                    "step jumps to, so forward() needs it (see the adapter's schedule)"
                )
            vec = vec + self.ref_time_in(timestep_embedding(target, 256).to(target.dtype))
        return vec

    @staticmethod
    def _one_span(length: int, mod: tuple[Tensor, Tensor, Tensor]) -> list[Span]:
        return [Span(0, length, ModOut(*mod))]

    def text_states(self, ctx: Tensor, vec_zero: Tensor, pe_ctx: Tensor) -> Tensor:
        """Text embedding ``[B, L, context_in_dim]`` -> the stream joining the joint blocks."""
        txt = self.txt_in(ctx)
        spans = self._one_span(txt.shape[1], self.early_stream_modulations[TEXT_MODE](vec_zero)[0])
        for block in self.txt_mode_blocks:
            mark_token_axis(txt, pe_ctx)
            txt = block(txt, spans, pe_ctx)
        return txt

    def reference_states(self, ref_tokens: Tensor, vec_zero: Tensor, pe_ref: Tensor) -> Tensor:
        """Reference latent tokens ``[B, n_ref, in_channels]`` -> their joint-stream tokens."""
        refs = self.emb_in["image_cond"](ref_tokens)
        spans = self._one_span(
            refs.shape[1], self.early_stream_modulations["image_cond"](vec_zero)[0]
        )
        for block in self.content_mode_blocks["image_cond"]:
            mark_token_axis(refs, pe_ref)
            refs = block(refs, spans, pe_ref)
        return refs

    # ------------------------------------------------------------ denoise step
    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe_x: Tensor,
        txt_states: Tensor,
        pe_ctx: Tensor,
        ref_states: Optional[Tensor] = None,
        pe_ref: Optional[Tensor] = None,
    ) -> Tensor:
        """One denoise step; returns the target tokens' velocity ``[B, L_img, out_channels]``.

        ``vec`` is :meth:`timestep_vec`'s ``[2, hidden]`` pair, ``txt_states`` /
        ``ref_states`` the early-block outputs of the timestep-zero streams, and the
        three ``pe_*`` their rotary frequencies. The joint sequence is
        ``[image, image_cond, txt]``.
        """
        num_img = x.shape[1]
        vec_img, vec_zero = vec[:1], vec[1:]

        img = self.emb_in["image"](x)
        early = ModOut(*self.early_stream_modulations["image"](vec_img)[0])

        # The joint sequence's spans: the target reads the ``t`` row, everything else
        # the timestep-zero row.
        parts: list[Tensor] = [img]
        mods: list[ModOut] = [ModOut(*self.single_stream_modulations["image"](vec_img)[0])]
        pe = pe_x
        if ref_states is not None:
            parts.append(ref_states)
            mods.append(ModOut(*self.single_stream_modulations["image_cond"](vec_zero)[0]))
            pe = torch.cat([pe, pe_ref], dim=1)
        parts.append(txt_states)
        mods.append(ModOut(*self.single_stream_modulations[TEXT_MODE](vec_zero)[0]))
        pe = torch.cat([pe, pe_ctx], dim=1)

        spans, start = [], 0
        for tokens, mod in zip(parts, mods):
            spans.append(Span(start, start + tokens.shape[1], mod))
            start += tokens.shape[1]

        for block in self.content_mode_blocks["image"]:
            mark_token_axis(img, pe_x)
            img = block(img, self._one_span(num_img, early), pe_x)

        x = torch.cat([img] + parts[1:], dim=1)
        for block in self.single_blocks:
            mark_token_axis(x, pe)
            x = block(x, spans, pe)

        # Only the target predicts: drop the references and the text again.
        return self.final_layer["image"](x[:, :num_img], vec_img)


# --------------------------------------------------------------- tokens / positions


def latent_to_tokens(latents: Tensor) -> Tensor:
    """Canonical packed latent ``[B, 128, H, W]`` -> tokens ``[B, H*W, 128]``."""
    return rearrange(latents, "b c h w -> b (h w) c")


def tokens_to_latent(tokens: Tensor, height: int, width: int) -> Tensor:
    """Target tokens -> canonical packed latent ``[B, 128, height, width]``."""
    return rearrange(tokens, "b (h w) c -> b c h w", h=height, w=width)


def image_ids(
    height: int,
    width: int,
    index: int = 0,
    *,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> Tensor:
    """``[1, H*W, 4]`` ``(t, row, col, l)`` ids of one image's tokens.

    The row and column axes carry :data:`IMAGE_ROPE_SCALE` times the grid index;
    ``l`` is ``index``, which is 0 for the target and negative per reference, so a
    reference never lands on a target position. Positions are fractional, so the ids
    are float — unlike Flux.2's integer ones.
    """
    return grid_from_axes(
        [
            torch.zeros(1, dtype=dtype, device=device),
            torch.arange(height, dtype=dtype, device=device) * IMAGE_ROPE_SCALE,
            torch.arange(width, dtype=dtype, device=device) * IMAGE_ROPE_SCALE,
            torch.full((1,), index, dtype=dtype, device=device),
        ]
    ).unsqueeze(0)


def ref_ids(
    shapes: Sequence[tuple[int, int]],
    *,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> Tensor:
    """``[1, sum(h*w), 4]`` ids for several references, on ``l`` axes -1, -2, ..."""
    return torch.cat(
        [
            image_ids(h, w, -(i + 1), dtype=dtype, device=device)
            for i, (h, w) in enumerate(shapes)
        ],
        dim=1,
    )


def text_ids(length: int, *, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
    """``[1, L, 4]`` ids of a text stream: every axis 0 except ``l = 0..L-1``."""
    ids = torch.zeros(1, length, 4, dtype=dtype, device=device)
    ids[0, :, 3] = torch.arange(length, dtype=dtype, device=device)
    return ids


__all__ = [
    "CONTENT_MODES",
    "IMAGE_ROPE_SCALE",
    "TEXT_MODE",
    "Block",
    "Flux3",
    "Flux3Params",
    "ModOut",
    "Span",
    "attention",
    "image_ids",
    "latent_to_tokens",
    "ref_ids",
    "text_ids",
    "tokens_to_latent",
]
