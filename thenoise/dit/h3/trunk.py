"""The H3 trunk: a packed single-stream transformer, shared by the image and the
audiovisual model.

The time embedding is a curve stored in the checkpoint and read between its rows, so it
stays FP32. What a variant adds on top of this trunk is its output head and the packing
its forward runs over: see :mod:`.image`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from thenoise.dit.h3.layout import MODALITIES, PATCH_SIZE, Layout
from thenoise.dit.kvcache import KVBuffers, attend
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.fp32_linear import Fp32Linear
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.rope import RopeCache, apply_rope_split_half, split_half_rope_axes

#: Whether this model's blocks want the relay store path (``thenoise.utils.layout``).
RELAY_STORES = True

#: A conditioned row's timestep label: a reference's noise is mixed to match it, and its
#: own rows are read at it instead of the step's.
COND_TIMESTEP = 0.999


@dataclass(frozen=True)
class H3Params:
    """The architecture, as read out of the checkpoint header.

    ``pdd_steps`` is the output head bank's row count and ``pdd_sigmas`` the sigma grid it
    was distilled on; a model without step-distilled heads reports one row and the
    two-point grid, which is the plain flow schedule's own endpoints.
    """

    hidden_size: int
    num_layers: int
    token_refiner_layers: int
    num_heads: int
    head_dim: int
    ffn_hidden_size: int
    latents_dim: int
    text_dim: int
    rope_inv_freq: int
    adaln_grid: int
    adaln_dim: int
    pdd_steps: int
    pdd_sigmas: Tuple[float, ...]
    #: Rows of the second stream's width, or ``None`` for a file with no audio stream:
    #: an audio row is a whole latent frame of it, so this is both the projection's
    #: input and the head's output width.
    audio_dim: Optional[int] = None
    patch_size: Tuple[int, int, int] = PATCH_SIZE
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @property
    def video_dim(self) -> int:
        """Latent channels times patch cells: the output head's width per PDD row."""
        return self.latents_dim * math.prod(self.patch_size)

    @property
    def rope_dim(self) -> int:
        """Rotated dimensions of a head: three axes sharing the checkpoint's frequencies."""
        return 6 * self.rope_inv_freq


class Mod(NamedTuple):
    """One modulation row, already projected: the shift/scale/gate of a span."""

    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Span(NamedTuple):
    """``[start, end)`` of the packed sequence and the modulation driving those rows."""

    start: int
    end: int
    attn: Mod
    mlp: Mod


class Attention(nn.Module):
    """Full attention over the packed sequence, with per-head Q/K norms and rope."""

    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = QuantizedLinear(hidden, inner * 3, bias=False)
        self.qk_norm = QKNorm(head_dim, eps=eps)
        self.out_proj = QuantizedLinear(inner, hidden, bias=False)

    def forward(self, x: torch.Tensor,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                bufs: Optional[KVBuffers] = None) -> torch.Tensor:
        """``x`` is ``[B, S, hidden]``; ``rope`` the run's ``(cos, sin)``, ``[1, 1, S, rot/2]``;
        ``bufs`` this block's run-long K/V (see :func:`.kvcache.attend`).
        """
        batch, rows, _ = x.shape
        inner = self.heads * self.head_dim
        q, k, v = self.qkv_proj(x).split(inner, dim=-1)

        def heads_(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, rows, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = heads_(q), heads_(k), heads_(v)
        # The checkpoint's q_norm/k_norm: without them the QK magnitudes grow with the
        # trunk width and the softmax saturates.
        q, k = self.qk_norm(q, k)
        if rope is not None:
            cos, sin = rope
            q = apply_rope_split_half(q, cos, sin)
            k = apply_rope_split_half(k, cos, sin)
        return self.out_proj(attend(q, k, v, bufs))


class MLP(nn.Module):
    """SwiGLU: one fused projection to ``2 * ffn``, gate half of it, project back."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = QuantizedLinear(hidden, ffn * 2, bias=False)
        self.fc2 = QuantizedLinear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class AdalnProj(nn.Module):
    """Modulation coordinates -> the per-(timestep, modality) shift/scale/gate rows.

    ``out_dtype`` brings the FP32 projection back to the trunk dtype: an FP32 shift
    would widen the residual stream behind it.
    """

    def __init__(self, t_dim: int, hidden: int, expand: int, modalities: int,
                 out_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.linear = Fp32Linear(t_dim, expand * hidden * modalities, bias=True,
                                 out_dtype=out_dtype)

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.linear(t_emb)
        x = x.view(-1, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _modulate(x: torch.Tensor, spans: Sequence[Span], norm: nn.Module, attn: bool
              ) -> torch.Tensor:
    """Pre-norm + AdaLN shift/scale, span by span: each span carries its own row."""
    parts = []
    for span in spans:
        mod = span.attn if attn else span.mlp
        parts.append((1.0 + mod.scale) * norm(x[:, span.start:span.end]) + mod.shift)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def _gate(x: torch.Tensor, spans: Sequence[Span], update: torch.Tensor, attn: bool
          ) -> torch.Tensor:
    """Accumulate a fresh attention/MLP output into the stream, gated, span by span."""
    parts = []
    for span in spans:
        mod = span.attn if attn else span.mlp
        sl = slice(span.start, span.end)
        parts.append(x[:, sl] + update[:, sl] * mod.gate)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


class DiTBlock(nn.Module):
    def __init__(self, params: H3Params, compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.norm1 = RMSNorm(params.hidden_size, eps=params.norm_eps)
        self.norm2 = RMSNorm(params.hidden_size, eps=params.norm_eps)
        self.attn = Attention(params.hidden_size, params.num_heads, params.head_dim,
                              params.qk_norm_eps)
        self.mlp = MLP(params.hidden_size, params.ffn_hidden_size)
        self.adaln_proj = AdalnProj(params.adaln_dim, params.hidden_size, 6, MODALITIES,
                                    out_dtype=compute_dtype)

    @torch.compile(fullgraph=False)
    def forward(self, x: torch.Tensor, spans: Sequence[Span],
                rope: Tuple[torch.Tensor, torch.Tensor],
                bufs: Optional[KVBuffers]) -> torch.Tensor:
        h = _modulate(x, spans, self.norm1, attn=True)
        x = _gate(x, spans, self.attn(h, rope, bufs), attn=True)
        h = _modulate(x, spans, self.norm2, attn=False)
        return _gate(x, spans, self.mlp(h), attn=False)


class RefinerBlock(nn.Module):
    """A plain pre-norm block: the prompt's own stream, unmodulated and unrotated."""

    def __init__(self, params: H3Params):
        super().__init__()
        self.norm1 = RMSNorm(params.hidden_size, eps=params.norm_eps)
        self.norm2 = RMSNorm(params.hidden_size, eps=params.norm_eps)
        self.attn = Attention(params.hidden_size, params.num_heads, params.head_dim,
                              params.qk_norm_eps)
        self.mlp = MLP(params.hidden_size, params.ffn_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm1(x)).add_(x)
        return self.mlp(self.norm2(x)).add_(x)


class TokenRefiner(nn.Module):
    """The prompt's encoder: unmodulated blocks over the projected text states, which
    do not depend on the timestep.
    """

    def __init__(self, params: H3Params):
        super().__init__()
        self.blocks = nn.ModuleList(
            RefinerBlock(params) for _ in range(params.token_refiner_layers))
        self.final_norm = RMSNorm(params.hidden_size, eps=params.final_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


def _spans(rows: Sequence[Tuple[int, int, int]],
           modulation: Tuple[torch.Tensor, ...]) -> list[Span]:
    """One block's modulation projection -> the packed sequence's spans.

    Row indexing stays out of the block: a row number inside a compiled graph is a
    guard on it.
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    return [
        Span(start, stop,
             Mod(shift_msa[row], scale_msa[row], gate_msa[row]),
             Mod(shift_mlp[row], scale_mlp[row], gate_mlp[row]))
        for start, stop, row in rows
    ]


class H3DiT(nn.Module):
    """What a variant has before it has an output head.

    ``video_patch_proj`` turns latent rows into stream rows, ``condition_proj`` and
    ``token_refiner`` turn the text encoder's states into prompt rows, and
    ``adaln_t_table`` is the modulation curve every block reads.
    """

    def __init__(self, params: H3Params,
                 compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.params = params
        self.video_patch_proj = Fp32Linear(params.video_dim, params.hidden_size,
                                           bias=True, out_dtype=None)
        self.condition_proj = QuantizedLinear(params.text_dim, params.hidden_size, bias=True)
        self.register_buffer("adaln_t_table",
                             torch.empty(params.adaln_grid, params.adaln_dim, dtype=torch.float32))
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq",
                                  torch.empty(params.rope_inv_freq, dtype=torch.float32))
        self.token_refiner = TokenRefiner(params)
        self.blocks = nn.ModuleList(
            DiTBlock(params, compute_dtype) for _ in range(params.num_layers))
        #: Rotary tables by conditioning branch; filled by :meth:`prepare_rope`.
        self.pe_embedder = RopeCache(self._rope_table)

    @property
    def rope_dim(self) -> int:
        return 6 * self.params.rope_inv_freq

    def _apply(self, fn, recurse=True):
        """Move the model, keeping the curve and the rope frequencies in FP32: every
        modulation vector is a projection read between the curve's rows, and a rope
        frequency is multiplied by a coordinate that grows with the canvas.
        """
        out = super()._apply(fn, recurse)
        if self.adaln_t_table.is_floating_point():
            self.adaln_t_table = self.adaln_t_table.float()
        if self.rope.inv_freq.is_floating_point():
            self.rope.inv_freq = self.rope.inv_freq.float()
        return out

    def refine_text(self, text_states: torch.Tensor) -> torch.Tensor:
        """``[1, L, text_dim]`` from the text encoder -> ``[1, L, hidden]``."""
        return self.token_refiner(self.condition_proj(text_states))

    def _rope_table(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``[S, 3]`` coordinates -> ``(cos, sin)``, each ``[1, 1, S, rope_dim/2]``, in
        float32 on the model's device: a coordinate is multiplied by an inverse
        frequency long before anything is rounded down to the compute dtype.
        """
        pos = position_ids.to(self.rope.inv_freq.device, torch.float32)
        return split_half_rope_axes(self.rope.inv_freq)(pos)

    def prepare_rope(self, layout: Layout, dtype: torch.dtype, name: str = "cond") -> None:
        """Build one branch's rotary table for a run; a forward then takes its ``name``."""
        self.pe_embedder.store(name, layout.position_ids, dtype=dtype)

    def time_embedding(self, timesteps: Sequence[float], device: torch.device) -> torch.Tensor:
        """The curve read at each timestep, interpolated between rows; ``t`` clamps to
        the curve's ends.
        """
        t = torch.tensor(list(timesteps), dtype=torch.float32, device=device)
        table = self.adaln_t_table.to(device)
        pos = t.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    def patchify(self, latent: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` -> one row per 2x2 patch, channels-major within a row."""
        _, c, h_full, w_full = latent.shape
        _, ph, pw = self.params.patch_size
        h, w = h_full // ph, w_full // pw
        rows = latent.reshape(-1, c, h, ph, w, pw)
        return rows.permute(0, 2, 4, 1, 3, 5).reshape(-1, c * ph * pw)

    def unpatchify(self, rows: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
        """The inverse of :meth:`patchify`, batch size 1."""
        _, ph, pw = self.params.patch_size
        h, w = latent_h // ph, latent_w // pw
        x = rows.reshape(1, h, w, self.params.latents_dim, ph, pw)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(1, self.params.latents_dim, latent_h, latent_w)


__all__ = [
    "COND_TIMESTEP",
    "RELAY_STORES",
    "AdalnProj",
    "Attention",
    "DiTBlock",
    "H3DiT",
    "H3Params",
    "MLP",
    "Mod",
    "RefinerBlock",
    "Span",
    "TokenRefiner",
    "_spans",
]
