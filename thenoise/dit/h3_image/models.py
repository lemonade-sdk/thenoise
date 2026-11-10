"""The H3-Image DiT: a single-stream packed-token transformer with a PDD head bank.

Architecture facts a checkpoint pins and this module reads off it (see
:func:`detect_params`): 50 blocks of ``hidden = 5376`` over 56 heads of 128, a
SwiGLU MLP at 8/3 width, and no cross-attention anywhere — the prompt, the reference
pictures and the target share the one sequence :mod:`.layout` builds, so a block is
attention plus MLP with nothing conditioning on which token is which. The only place
modality enters is the AdaLN modulation, and that is a *lookup*, not a projection of
a computed time embedding:

  * ``adaln_t_table`` is a 1025-row sampling of the time-embedding curve, and a step
    reads it at ``t * 1024`` with linear interpolation between rows. The exported
    checkpoint therefore carries no time embedder at all, and the per-block heads
    consume an 8-wide coordinate vector rather than a full-width embedding.
  * The table's rows are indexed by ``timestep_row * 3 + modality_tag``, so one
    projection yields the modulation of every (timestep, modality) pair this forward
    needs — which is how the reference rows of an edit, pinned at
    :data:`COND_TIMESTEP`, get their own shift/scale/gate without a second pass.

The output head is a **PDD head bank**: ``pdd_steps`` stacked heads, one per interval
of the checkpoint's own sigma grid, where row 0 is a complete head and every later row
is the offset from it for that interval. A step that spans several grid intervals —
which is what happens the moment the sampler's schedule is not the distilled one —
takes the dt-weighted mean of the heads it spans. This is why the sampler is fixed:
the heads, the grid in :mod:`.sampling` and the 6-step default are one artifact.

Two dtype islands survive from the release, both where rounding would be a quality
regression rather than a rounding error: the input patch projection and the output
head bank run in FP32 (:class:`~thenoise.utils.fp32_linear.Fp32Linear`), as does the
modulation path that feeds them.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights

from thenoise.dit.h3_image.layout import MODALITIES, PATCH_SIZE, Layout, pad_to_patch
from thenoise.dit.kvcache import KVBuffers, KVCache, attend, cache_mode
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.dynamo import mark_token_axis
from thenoise.utils.fp32_linear import Fp32Linear
from thenoise.utils.loader import load_dit
from thenoise.utils.qk_norm import QKNorm, qk_norm_key_map
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.rope import RopeCache, apply_rope_split_half, split_half_rope_axes
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key

logger = logging.getLogger(__name__)

#: Reference rows are fed as if they were almost pure noise: the timestep they are
#: labelled with, and the fraction of noise mixed into them to match that label.
COND_TIMESTEP = 0.999


@dataclass(frozen=True)
class H3ImageParams:
    """The architecture, as read out of the checkpoint header."""

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
    """``[start, end)`` of the packed sequence and the modulation driving those rows.

    Both halves of a block are carried because a span is a property of the sequence,
    not of the projection: which rows share a modulation row is decided once per
    forward, outside the compiled blocks, so that no row index reaches a graph.
    """

    start: int
    end: int
    attn: Mod
    mlp: Mod


class Attention(nn.Module):
    """Full attention over the packed sequence, with per-head Q/K norms and rope.

    The projections are wider than the residual stream (``heads * head_dim`` is
    7168 against a 5376 trunk), and there is no gate on the attention output: the
    modulation's gate is the only one.
    """

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
        """``x`` is ``[B, S, hidden]``; ``rope`` the run's ``(cos, sin)``, ``[S, 1, 1, rot]``.

        ``bufs`` are this block's run-long K/V buffers (see :func:`.kvcache.attend`),
        and the refiner calls with neither: its stream is unrotated and never cached.
        """
        batch, rows, _ = x.shape
        inner = self.heads * self.head_dim
        q, k, v = self.qkv_proj(x).split(inner, dim=-1)

        def heads_(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, rows, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = heads_(q), heads_(k), heads_(v)
        # Per-head norms before the rotation: without them the QK magnitudes grow with
        # the trunk width and the softmax saturates, which is what the checkpoint's
        # q_norm/k_norm weights exist to hold down.
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

    ``forward`` turns ``[M, t_dim]`` into ``expand`` tensors of ``[M * modalities,
    hidden]``, so the ``M`` timesteps of one forward — the target's and the pinned
    one the references carry — are projected together and indexed by row.
    """

    def __init__(self, t_dim: int, hidden: int, expand: int, modalities: int):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.linear = Fp32Linear(t_dim, expand * hidden * modalities, bias=True)

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
    def __init__(self, params: H3ImageParams):
        super().__init__()
        self.norm1 = RMSNorm(params.hidden_size, eps=params.norm_eps)
        self.norm2 = RMSNorm(params.hidden_size, eps=params.norm_eps)
        self.attn = Attention(params.hidden_size, params.num_heads, params.head_dim,
                              params.qk_norm_eps)
        self.mlp = MLP(params.hidden_size, params.ffn_hidden_size)
        self.adaln_proj = AdalnProj(params.adaln_dim, params.hidden_size, 6, MODALITIES)

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

    def __init__(self, params: H3ImageParams):
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
    """The prompt's encoder: a few unmodulated blocks over the projected text states.

    Runs once per sampling rather than once per step (the adapter caches its output in
    :meth:`H3ImageDiT.refine_text`), because the prompt does not depend on the
    timestep.
    """

    def __init__(self, params: H3ImageParams):
        super().__init__()
        self.blocks = nn.ModuleList(
            RefinerBlock(params) for _ in range(params.token_refiner_layers))
        self.final_norm = RMSNorm(params.hidden_size, eps=params.final_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class FinalLayer(nn.Module):
    """Modulated output head over the target's rows, as a PDD head bank.

    ``pdd_sigmas`` is the grid the bank was distilled on, one point per interval
    boundary; ``pdd_steps`` rows of head are stored in the projection, row 0 complete
    and the rest offsets from it.
    """

    def __init__(self, params: H3ImageParams):
        super().__init__()
        self.video_dim = params.video_dim
        self.pdd_steps = params.pdd_steps
        self.pdd_sigmas = tuple(params.pdd_sigmas)
        self.norm = RMSNorm(params.hidden_size, eps=params.final_norm_eps)
        self.adaln_proj = AdalnProj(params.adaln_dim, params.hidden_size, 2, 1)
        self.video_out = Fp32Linear(params.hidden_size, params.video_dim * params.pdd_steps,
                                    bias=True)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, target: Tuple[int, int],
                sigma: float, sample_sigmas: Optional[torch.Tensor]) -> torch.Tensor:
        """The target rows' velocity; ``target`` is their ``(start, stop)``.

        The head is modulated by the step's timestep row alone: reference rows never
        reach it, and the prompt's tags are a block-level concern.
        """
        shift, scale = self.adaln_proj(t_emb)
        start, stop = target
        h = (self.norm(x[:, start:stop]) * (1.0 + scale[0]) + shift[0]).to(torch.float32)
        if self.pdd_steps == 1:
            return self.video_out(h)
        weight, bias = self._bank(sigma, sample_sigmas)
        return F.linear(h, weight, bias)

    def _bank(self, sigma: float, sample_sigmas: Optional[torch.Tensor]):
        """The dt-weighted head for a step, folded onto the shape of one head.

        The step's endpoints are located on both the sampler's schedule and the
        distilled grid, and every interval between the two is averaged with weights
        proportional to the sigma distance it covers. The start is clamped into the
        bank first, so a step at the last interval still has one to its name.
        """
        if sample_sigmas is None:
            raise ValueError(
                "the PDD head bank needs the sigma schedule this run is stepping "
                "through; pass it as ``sample_sigmas``"
            )
        n = self.pdd_steps
        grid = self.pdd_sigmas
        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = float(sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)])
        start = min(range(n + 1), key=lambda k: abs(grid[k] - sigma))
        stop = min(range(n + 1), key=lambda k: abs(grid[k] - sigma_next))
        start = min(start, n - 1)
        stop = max(stop, start + 1)

        dt = [grid[k] - grid[k + 1] for k in range(start, stop)]
        weight_of = self.video_out.weight
        w = torch.tensor(dt, dtype=weight_of.dtype, device=weight_of.device) / sum(dt)
        rows = weight_of.reshape(len(grid) - 1, -1, weight_of.shape[1])
        bias_rows = self.video_out.bias.reshape(len(grid) - 1, -1)
        first = max(start, 1)
        return (
            rows[0] + torch.einsum("n,noi->oi", w[first - start:], rows[first:stop]),
            bias_rows[0] + torch.einsum("n,no->o", w[first - start:], bias_rows[first:stop]),
        )


def _spans(rows: Sequence[Tuple[int, int, int]],
           modulation: Tuple[torch.Tensor, ...]) -> list[Span]:
    """One block's modulation projection -> the packed sequence's spans.

    The row indexing happens here rather than in the block because a row number baked
    into a compiled graph is a guard on it, and the rows of a run are not constants of
    the architecture.
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    return [
        Span(start, stop,
             Mod(shift_msa[row], scale_msa[row], gate_msa[row]),
             Mod(shift_mlp[row], scale_mlp[row], gate_mlp[row]))
        for start, stop, row in rows
    ]


class H3ImageDiT(nn.Module):
    """The image DiT: ``[text | references | target]`` in, the target's velocity out.

    ``context`` is the *refined* prompt — :meth:`refine_text` is meant to run once per
    sampling, so the forward never sees raw text-encoder states.
    """

    def __init__(self, params: H3ImageParams):
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
        self.blocks = nn.ModuleList(DiTBlock(params) for _ in range(params.num_layers))
        self.final_layer = FinalLayer(params)
        #: Rotary tables by conditioning branch; filled by :meth:`prepare_rope`.
        self.pe_embedder = RopeCache(self._rope_table)

    @property
    def rope_dim(self) -> int:
        """Rotated dimensions of a head: three axes sharing the checkpoint's frequencies."""
        return 6 * self.params.rope_inv_freq

    def _apply(self, fn, recurse=True):
        """Move the model, keeping the curve and the rope frequencies in FP32.

        The table is read by interpolating between neighbouring rows, and every
        modulation vector in the model is a projection of the result: rounding the
        curve to the compute dtype would move the conditioning of all 50 blocks, not
        just one tensor. An inverse frequency is multiplied by a coordinate that grows
        with the canvas, so its rounding is amplified the same way. The FP32 linears
        pin themselves by the same mechanism.
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
        """``[S, 3]`` coordinates -> ``(cos, sin)``, each ``[S, 1, 1, rope_dim]``.

        All three axes share one frequency vector. The coordinates are float64, so
        they are brought to float32 here and the result is rounded once, by the cache.
        """
        return split_half_rope_axes(self.rope.inv_freq)(position_ids.to(torch.float32))

    def prepare_rope(self, layout: Layout, dtype: torch.dtype, name: str = "cond") -> None:
        """Build one branch's rotary table for a run; a forward then takes its ``name``."""
        self.pe_embedder.store(name, layout.position_ids, dtype=dtype)

    def time_embedding(self, timesteps: Sequence[float], device: torch.device) -> torch.Tensor:
        """The curve table read at each timestep, interpolated between its rows.

        ``t`` outside ``[0, 1]`` clamps to the curve's ends; ``t = 1`` stays on the
        last interval rather than reading past the table.
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

    def _codec_rows(self, target: torch.Tensor, refs: Sequence[torch.Tensor],
                    seed: int) -> torch.Tensor:
        """Patchified rows in packing order: the target, then the references.

        A reference is labelled as a nearly-noise row, so it is mixed with the noise
        that label promises — :data:`COND_TIMESTEP` of signal, the rest noise. Each
        one restarts the same RNG stream on purpose: a second reference must not
        inherit the random draws the first one consumed.
        """
        rows = [self.patchify(target.to(torch.float32))]
        for ref in refs:
            padded, _ = pad_to_patch(ref.to(torch.float32), self.params.patch_size)
            r = self.patchify(padded)
            gen = torch.Generator("cpu").manual_seed(seed)
            noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
            rows.append(COND_TIMESTEP * r + (1.0 - COND_TIMESTEP) * noise.to(r.device))
        return torch.cat(rows, dim=0)

    def forward(
        self,
        latent: torch.Tensor,
        sigma: float,
        context: torch.Tensor,
        layout: Layout,
        refs: Sequence[torch.Tensor] = (),
        tags: Optional[torch.Tensor] = None,
        seed: int = 0,
        sample_sigmas: Optional[torch.Tensor] = None,
        rope: str = "cond",
        kv: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """One forward pass; returns the target's velocity, ``[1, C, H, W]``.

        ``latent`` is the canonical noisy latent (any H/W; it is repeat-padded to a
        whole patch and the padding cut back off the result), ``sigma`` the flow
        matching noise level, ``context`` the refined prompt ``[1, L, hidden]``,
        ``refs`` the reference latents in :attr:`Layout` order, ``tags`` the text
        encoder's per-token modality tags, ``sample_sigmas`` the schedule the run is
        stepping through (needed whenever the head bank has more than one row),
        ``rope`` the name this branch's rotary table was stored under, and ``kv`` this
        branch's K/V cache.

        A cached run walks the whole sequence once, references included, and afterwards
        runs its leading rows — the prompt and the target — reading the references back
        out of the buffers. That is the slice a cache may freeze: a reference is fed the
        same latent, the same noise and :data:`COND_TIMESTEP` at every step, while the
        prompt follows the step's timestep and is therefore recomputed.
        """
        device = latent.device
        dtype = context.dtype
        padded, (orig_h, orig_w) = pad_to_patch(latent, self.params.patch_size)
        lat_h, lat_w = padded.shape[-2], padded.shape[-1]
        if (lat_h, lat_w) != (layout.latent_h, layout.latent_w):
            raise ValueError(
                f"layout was built for a {layout.latent_h}x{layout.latent_w} latent, "
                f"got {lat_h}x{lat_w}"
        )

        mode = cache_mode(kv, bool(refs))
        rows = layout.modulation_rows(tags)
        cos, sin = self.pe_embedder[rope]
        if mode == "read":
            n_rows = layout.target_span[1]
            rows = [row for row in rows if row[0] < n_rows]
            cos, sin = cos[:n_rows], sin[:n_rows]
            patched = self.video_patch_proj(self.patchify(padded)).to(dtype)
            h = torch.cat([context, patched.unsqueeze(0)], dim=1)
        else:
            patched = self.video_patch_proj(self._codec_rows(padded, refs, seed)).to(dtype)
            h = torch.empty(1, layout.seq_len, self.params.hidden_size,
                            dtype=dtype, device=device)
            codec = 0
            for start, stop, kind in layout.segments:
                if kind == "text":
                    h[:, start:stop] = context[0]
                else:
                    n = stop - start
                    h[:, start:stop] = patched[codec:codec + n]
                    codec += n

        # Row 0 of the modulation is this step's timestep and row 1 the reference's
        # pinned one; the pair keeps its shape at the last step, where they coincide.
        t = 1.0 - float(sigma)
        t_emb = self.time_embedding((t, max(t, COND_TIMESTEP)), device)
        kv_shape = (1, self.params.num_heads, layout.seq_len, self.params.head_dim)
        mark_token_axis(h, dim=1)
        mark_token_axis(cos, sin, dim=0)
        for i, block in enumerate(self.blocks):
            bufs = None if kv is None else kv.buffers(("attn", i), mode, kv_shape, dtype, device)
            h = block(h, _spans(rows, block.adaln_proj(t_emb)), (cos, sin), bufs)
        if kv is not None and mode == "fill":
            kv.set_filled()

        out = self.final_layer(h, t_emb, layout.target_span, sigma, sample_sigmas)
        out = self.unpatchify(out[0], lat_h, lat_w)
        return -out[:, :, :orig_h, :orig_w].to(latent.dtype)


def detect_params(dit_path: str) -> H3ImageParams:
    """Read the architecture, the PDD head count and its sigma grid out of a file.

    The head count is a shape (``video_out`` holds one head per grid interval), but
    the grid itself is only in the checkpoint's ``config`` metadata — and the two have
    to agree, because a head bank indexed by a grid it does not span predicts a
    plausible-but-wrong velocity rather than failing.
    """
    with MemoryEfficientSafeOpen(dit_path) as f:
        shapes = {unwrap_key(k): tuple(f.header[k]["shape"]) for k in f.keys()}
        metadata = f.metadata()

    video_out = shapes["final_layer.video_out.weight"]
    patch_proj = shapes["video_patch_proj.weight"]
    adaln_t_table = shapes["adaln_t_table"]
    qkv = shapes["blocks.0.attn.qkv_proj.weight"]
    head_dim = shapes["blocks.0.attn.q_norm.weight"][0]

    config = _transformer_config(metadata)
    pdd_steps = video_out[0] // patch_proj[1]
    pdd_sigmas = tuple(config.get("pdd_sigmas", (1.0, 0.0)))
    if len(pdd_sigmas) != pdd_steps + 1:
        raise ValueError(
            f"{dit_path}: the head bank is {pdd_steps} rows but the distilled sigma "
            f"grid has {len(pdd_sigmas)} points ({pdd_sigmas}); expected grid length "
            f"{pdd_steps + 1}"
        )

    return H3ImageParams(
        hidden_size=patch_proj[0],
        num_layers=_block_count(shapes),
        token_refiner_layers=_block_count(shapes, "token_refiner.blocks."),
        num_heads=qkv[0] // (3 * head_dim),
        head_dim=head_dim,
        ffn_hidden_size=shapes["blocks.0.mlp.fc2.weight"][1],
        latents_dim=patch_proj[1] // math.prod(PATCH_SIZE),
        text_dim=shapes["condition_proj.weight"][1],
        rope_inv_freq=shapes["rope.inv_freq"][0],
        adaln_grid=adaln_t_table[0],
        adaln_dim=adaln_t_table[1],
        pdd_steps=pdd_steps,
        pdd_sigmas=pdd_sigmas,
    )


def _transformer_config(metadata: dict) -> dict:
    """The ``config`` metadata blob's ``transformer`` section, empty when absent."""
    raw = metadata.get("config")
    if not raw:
        return {}
    try:
        return json.loads(raw).get("transformer", {})
    except (TypeError, ValueError):
        logger.warning("Could not parse the checkpoint's transformer config metadata")
        return {}


def _block_count(shapes: dict, prefix: str = "blocks.") -> int:
    """How many ``<prefix><i>.`` blocks a key-set contains."""
    indices = []
    for key in shapes:
        if not key.startswith(prefix):
            continue
        index = key[len(prefix):].split(".", 1)[0]
        if index.isdigit():
            indices.append(int(index))
    return max(indices) + 1 if indices else 0


def load_h3_image_dit(
    dit_path: str,
    params: H3ImageParams,
    device,
    dtype: torch.dtype,
) -> H3ImageDiT:
    """Build the DiT on meta from the checkpoint's own geometry, then load its weights.

    The file is loaded with its own dtypes — BF16 trunk, FP32 patch projection and head
    bank, FP16 modulation heads — and only then moved to ``dtype``, because that move is
    where the FP32 islands pin themselves (:class:`~thenoise.utils.fp32_linear.Fp32Linear`
    and :meth:`H3ImageDiT._apply`) and where the quantized weights stay quantized.
    Casting the state dict instead would round the modulation factors the whole model is
    conditioned on before they ever arrived.

    The checkpoint stores the QK norms as ``attn.q_norm`` / ``attn.k_norm``; they are
    remapped onto the shared :class:`~thenoise.utils.qk_norm.QKNorm` layout.
    """
    with init_empty_weights():
        model = H3ImageDiT(params)
    load_dit(model, dit_path, device=device, dtype=None, key_map=qk_norm_key_map)
    model.to(dtype=dtype)
    logger.info(
        "Loaded H3-Image DiT (%d layers, %dx%d heads, %d-step PDD head bank) from %s",
        params.num_layers, params.num_heads, params.head_dim, params.pdd_steps, dit_path,
    )
    return model


__all__ = [
    "COND_TIMESTEP",
    "H3ImageDiT",
    "H3ImageParams",
    "detect_params",
    "load_h3_image_dit",
]
