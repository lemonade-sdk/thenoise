"""Qwen-Image 2.1 DiT — single-stream transformer with a *causal* text/reference prefix.

Ported from ComfyUI's ``comfy/ldm/qwen_image21/model.py`` (itself a port of the
diffusers Qwen-Image 2.1 implementation, Apache-2.0) onto this engine's shared
parts: ``QuantizedLinear``, the shared ``RMSNorm``/``QKNorm``, ``matrix_rope`` +
``apply_rope`` and ``thenoise.dit.kvcache``. Weight names follow the released
checkpoints so they load as-is (the only renames are the QK norms onto the shared
``QKNorm`` layout — see ``thenoise.dit.qwen_image21.utils``).

Three things set this architecture apart from the other adapters, and they are all
visible from the sequence layout:

* **The latent is the DiT input.** ``img_in`` maps the VAE's 64 channels straight
  to the model width, so one token is one latent cell of the 16x-compressed grid —
  no pack/patchify step anywhere (the canonical latent IS the model-internal one).
* **One modulation drives every block**, computed from two timestep rows: the
  sampled ``t`` for the target image tokens and ``t = 0`` for the whole text +
  reference prefix. That prefix modulation is what makes the prefix K/V
  step-invariant, i.e. cacheable.
* **Attention is block-causal.** The sequence is ``[text … references … target]``
  with each reference spliced into the text at the slot its tokenizer run recorded
  and the target image LAST. A text segment attends causally to what precedes it
  and an image segment reads everything up to its own end, so no prefix token ever
  sees the target. The cached prefix is therefore *exact* rather than the
  approximation Flux.2 Klein / Qwen-Image accept with their bidirectional
  references — and it also means the target-only steps (everything after the fill)
  need no mask at all.

The reference implementation fuses the QK-norm + RoPE and the AdaLN into
``comfy_kitchen`` kernels over a ``(B, N, H, D)`` layout; here they are the shared
PyTorch modules over the repo's ``(B, H, N, D)`` attention layout, which is what
``thenoise.utils.attention`` and the KV cache speak.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from thenoise.dit.kvcache import KVBuffers, KVCache, attend, cache_mode
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import AttentionParams, attention as sdpa_attention
from thenoise.utils.positions import broadcast_positions, grid_from_axes
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.rope import RopeCache, apply_rope, matrix_rope
from thenoise.utils.timestep import timestep_embedding

#: RoPE theta of the released checkpoints.
ROPE_THETA = 10000.0


@dataclass
class QwenImage21Params:
    """Architecture knobs, read back from the checkpoint by ``detect_params``."""

    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, ...] = (16, 56, 56)
    eps: float = 1e-6
    #: ``img_mlp`` stores ``[gate; up]`` as one ``gate_up`` matrix in the released
    #: checkpoints; the split form is the diffusers/original layout.
    fused_mlp: bool = True


@dataclass(frozen=True)
class QwenImage21Sequence:
    """One conditioning branch's step-invariant sequence, built once per run.

    Everything the denoise loop would otherwise redo every step lives here: the
    *projected* prefix tokens (text + references — step-invariant because they are
    modulated at ``t = 0``), the RoPE table for the full sequence, the block-causal
    attention segments of the fill pass, and the split point between prefix and
    target. The target image tokens are the only thing ``forward`` adds.
    """

    prefix: Optional[Tensor]
    pe: Tensor
    segments: tuple[tuple[int, int, Optional[Tensor]], ...] = ()
    prefix_len: int = 0
    target_len: int = 0
    height: int = 0
    width: int = 0


@dataclass(frozen=True)
class AttentionPlan:
    """How the blocks attend on one forward, per block (``bufs`` differs per block).

    ``fill`` / ``off`` run the block-causal segments (the text segments carry a
    causal mask, the image segments need none, so the biggest attention of the run
    stays mask-free); ``read`` drops the prefix from the sequence and attends over
    the run's cached prefix + this step's target tokens through the shared
    ``kvcache.attend``.
    """

    mode: str
    segments: Sequence[tuple[int, int, Optional[Tensor]]] = ()
    bufs: Optional[KVBuffers] = None

    def run(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Attention for one block; ``q``/``k``/``v`` are ``[B, H, N, D]`` post-RoPE."""
        if self.mode == "read":
            return attend(q, k, v, self.bufs)
        if self.bufs is not None:
            # A fill runs the whole sequence, so what it writes IS the cache.
            self.bufs.write(k, v)
        outs = [
            sdpa_attention(
                [q[:, :, start:end], k[:, :, :end], v[:, :, :end]],
                attn_params=None if mask is None else AttentionParams(mask),
            )
            for start, end, mask in self.segments
        ]
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)


class TimestepEmbedding(nn.Module):
    """``linear_1 -> SiLU -> linear_2``, no biases (diffusers ``TimestepEmbedding``)."""

    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.linear_1 = QuantizedLinear(in_dim, dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = QuantizedLinear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class TimestepProjEmbeddings(nn.Module):
    """Sinusoidal timestep features -> the modulation's input embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, dim)

    def forward(self, timestep: Tensor) -> Tensor:
        return self.timestep_embedder(timestep_embedding(timestep, 256))


class TextProjection(nn.Module):
    """Text-encoder hidden states -> DiT width.

    ``text_norm`` is a *zero-centred* RMSNorm in the checkpoint (its weight stores
    ``scale - 1``); the loader adds the 1 back so the shared ``RMSNorm`` can be used
    unmodified — the same reconciliation Krea 2 applies.
    """

    def __init__(self, in_dim: int, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.text_norm = RMSNorm(in_dim, eps=eps)
        self.in_layer = QuantizedLinear(in_dim, dim, bias=False)
        self.out_layer = QuantizedLinear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU MLP; ``fused`` keeps ``[gate; up]`` in one GEMM, as the checkpoints do."""

    def __init__(self, dim: int, hidden_dim: int, fused: bool = True) -> None:
        super().__init__()
        self.fused = fused
        if fused:
            self.gate_up = QuantizedLinear(dim, 2 * hidden_dim, bias=False)
        else:
            self.proj = QuantizedLinear(dim, hidden_dim, bias=False)
            self.gate_layer = QuantizedLinear(dim, hidden_dim, bias=False)
        self.out = QuantizedLinear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.fused:
            gate, up = self.gate_up(x).chunk(2, dim=-1)
        else:
            gate, up = self.gate_layer(x), self.proj(x)
        return self.out(F.silu(gate) * up)


class Attention(nn.Module):
    """Attention over one joint sequence: separate QKV, QK-norm, RoPE, out proj."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        inner = num_heads * head_dim
        self.to_q = QuantizedLinear(dim, inner, bias=False)
        self.to_k = QuantizedLinear(dim, inner, bias=False)
        self.to_v = QuantizedLinear(dim, inner, bias=False)
        self.qk_norm = QKNorm(head_dim, eps=eps)
        # ``to_out`` is a ModuleList to match the checkpoint's ``to_out.0`` key.
        self.to_out = nn.ModuleList([QuantizedLinear(inner, dim, bias=False)])

    def forward(self, x: Tensor, pe: Tensor, plan: AttentionPlan) -> Tensor:
        B, N, _ = x.shape

        def split(t: Tensor) -> Tensor:
            return t.view(B, N, self.num_heads, -1).transpose(1, 2)

        q = split(self.to_q(x))
        k = split(self.to_k(x))
        v = split(self.to_v(x))
        q, k = self.qk_norm(q, k)
        q, k = apply_rope(q, k, pe)
        return self.to_out[0](plan.run(q, k, v))


class QwenImage21TransformerBlock(nn.Module):
    """Pre-norm attention + SwiGLU block with per-token AdaLN *scale* (no shift)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
        fused_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, num_heads, head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio, fused=fused_mlp)

    def forward(
        self,
        x: Tensor,
        scale1: Tensor,
        gate1: Tensor,
        scale2: Tensor,
        gate2: Tensor,
        pe: Tensor,
        plan: AttentionPlan,
    ) -> Tensor:
        x = torch.addcmul(x, self.attn(self.img_norm1(x) * (1.0 + scale1), pe, plan), gate1)
        x = torch.addcmul(x, self.img_mlp(self.img_norm2(x) * (1.0 + scale2)), gate2)
        if x.dtype == torch.float16:
            x = x.clip(-65504.0, 65504.0)
        return x


class LastLayer(nn.Module):
    """Output AdaLN — scale only, no shift — before ``proj_out``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.linear = QuantizedLinear(dim, dim, bias=False)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        return self.norm(x) * (1.0 + self.linear(F.silu(temb)).unsqueeze(1))


def modulation_rows(rows: Tensor, prefix_len: int, n_tokens: int) -> Tensor:
    """``[2, D]`` (sampled-``t`` row, then the ``t = 0`` row) -> per-token ``[1, N, D]``.

    The prefix (text + references) modulates from ``t = 0`` and the target image
    from ``t``. With no prefix the ``t`` row is returned unsqueezed so it broadcasts
    over every token instead of materialising a per-token copy.
    """
    if prefix_len:
        target_len = n_tokens - prefix_len
        return torch.cat(
            [rows[1:].expand(prefix_len, -1), rows[:1].expand(target_len, -1)], dim=0
        )[None]
    return rows[:1, None]


class QwenImage21Transformer2DModel(nn.Module):
    """The Qwen-Image 2.1 transformer (text-to-image and reference editing)."""

    def __init__(self, params: Optional[QwenImage21Params] = None) -> None:
        super().__init__()
        params = params or QwenImage21Params()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.num_heads = params.num_attention_heads
        self.head_dim = params.attention_head_dim
        dim = self.num_heads * self.head_dim
        if sum(params.axes_dims_rope) != self.head_dim:
            raise ValueError(
                f"axes_dims_rope {tuple(params.axes_dims_rope)} must sum to the head "
                f"dim {self.head_dim}"
            )
        self.dim = dim

        self.pe_embedder = RopeCache(matrix_rope(list(params.axes_dims_rope), ROPE_THETA))
        self.time_text_embed = TimestepProjEmbeddings(dim)
        self.txt_in = TextProjection(params.context_in_dim, dim, eps=params.eps)
        self.img_in = QuantizedLinear(params.in_channels, dim, bias=False)

        # One modulation shared by every block: scale/gate for the two sub-layers.
        self.modulation = nn.Sequential(
            nn.SiLU(), QuantizedLinear(dim, 4 * dim, bias=False)
        )
        self.transformer_blocks = nn.ModuleList(
            QwenImage21TransformerBlock(
                dim, self.num_heads, self.head_dim,
                mlp_ratio=params.mlp_ratio, eps=params.eps, fused_mlp=params.fused_mlp,
            )
            for _ in range(params.num_layers)
        )
        self.norm_out = LastLayer(dim, eps=params.eps)
        self.proj_out = QuantizedLinear(dim, params.out_channels, bias=False)

    # ------------------------------------------------------------------ sequence
    def build_sequence(
        self,
        x: Tensor,
        context: Tensor,
        ref_latents: Optional[Sequence[Tensor]] = None,
        image_slots: Optional[Sequence[int]] = None,
        *,
        name: str = "seq",
        dtype: Optional[torch.dtype] = None,
    ) -> QwenImage21Sequence:
        """Project text + references + target into one causal sequence, ONCE per run.

        ``context`` is the text-encoder embedding and ``image_slots`` the token index
        at which each reference image's (removed) vision tokens sat in the prompt, so
        the reference latents are spliced back into the text stream exactly where the
        language model saw the picture. A branch without references just gets the
        ``image_slots``-free layout (text first, target last).

        ``name`` keys the computed RoPE table in ``pe_embedder``, so the conditional
        and unconditional branches (different prompt lengths) keep their own entry —
        same convention as the other adapters' ``"txt"`` / ``"txt_uncond"``.
        """
        x = x.to(device=self.device)
        refs = [r.to(device=x.device) for r in (ref_latents or [])]
        H, W = x.shape[-2:]
        dev = x.device

        txt = self.txt_in(context.to(device=dev, dtype=self._dtype))
        txt_len = txt.shape[1]
        slots = list(image_slots or []) + [txt_len] * len(refs)

        parts: list[Tensor] = []
        ids: list[Tensor] = []
        segments: list[tuple[int, int, Optional[Tensor]]] = []
        pos = length = cursor = 0

        def push_text(end: int) -> None:
            """Text up to ``end``, attending causally to everything before itself."""
            nonlocal pos, length, cursor
            n = end - cursor
            if n > 0:
                parts.append(txt[:, cursor:end])
                ids.append(broadcast_positions(n, 3, offset=pos, device=dev))
                # bool SDPA mask, True = attend: row i (global index length + i) may
                # read keys 0..length+i, i.e. causal over the accumulated sequence.
                mask = torch.ones(n, length + n, dtype=torch.bool, device=dev).tril(length)
                segments.append((length, length + n, mask[None, None]))
                pos += n
                length += n
            cursor = end

        def push_image(img: Tensor) -> int:
            """One latent grid -> tokens, centred on the target grid; returns its length."""
            nonlocal pos, length
            h, w = img.shape[-2:]
            parts.append(self.img_in(img.to(dtype=self._dtype).flatten(2).transpose(1, 2)))
            # Reference grids are offset by half a token whenever their parity differs
            # from the target, which keeps them centred on it (the target itself lands
            # on the plain ``-ceil(n/2)`` centring).
            hh = torch.arange(h, device=dev, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - H % 2)
            ww = torch.arange(w, device=dev, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - W % 2)
            ids.append(grid_from_axes(
                [torch.full((1,), float(pos), device=dev, dtype=torch.float32), hh, ww]
            ))
            # An image segment reads everything up to its own end — no mask needed.
            segments.append((length, length + h * w, None))
            pos += max(h, w)
            length += h * w
            return h * w

        for i, ref in enumerate(refs):
            push_text(slots[i])
            push_image(ref)
        push_text(txt_len)
        target_len = push_image(x)

        tokens = torch.cat(parts, dim=1)
        pe = self.pe_embedder.store(name, torch.cat(ids, dim=0)[None], dtype=dtype)
        prefix_len = tokens.shape[1] - target_len
        return QwenImage21Sequence(
            prefix=tokens[:, :prefix_len] if prefix_len else None,
            pe=pe,
            segments=tuple(segments),
            prefix_len=prefix_len,
            target_len=target_len,
            height=H,
            width=W,
        )

    # -------------------------------------------------------------------- forward
    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        seq: QwenImage21Sequence,
        kv: Optional[KVCache] = None,
    ) -> Tensor:
        """One forward on the target latents ``[B, C, H, W]`` -> velocity, same shape.

        ``seq`` carries the per-run prefix built by :meth:`build_sequence`; ``kv`` is
        the run's cache for this conditioning branch. Once the cache is filled the
        prefix leaves the sequence entirely (``cache_mode`` says ``read``) and each
        block attends over ``[cached prefix, this step's target]``.
        """
        B, _, H, W = x.shape
        target = self.img_in(x.flatten(2).transpose(1, 2))

        mode = cache_mode(kv, seq.prefix is not None)
        if mode == "read":
            hidden, pe, prefix_len = target, seq.pe[:, seq.prefix_len:], 0
        else:
            hidden = target if seq.prefix is None else torch.cat([seq.prefix, target], dim=1)
            pe, prefix_len = seq.pe, seq.prefix_len
        n_tokens = hidden.shape[1]

        # Two modulation rows: the sampled timestep for the target, t = 0 for the
        # prefix. (The reference rounds ``t * 1000`` to the compute dtype to match its
        # pipeline's rounding; this engine feeds the flow timestep straight in.)
        t = timesteps.reshape(-1)[:1].to(device=hidden.device, dtype=hidden.dtype)
        temb = self.time_text_embed(torch.cat([t, t * 0]))
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)

        def rows(r: Tensor) -> Tensor:
            return modulation_rows(r, prefix_len, n_tokens)

        mods = (rows(scale1), rows(gate1.tanh()), rows(scale2), rows(gate2.tanh()))

        plan = AttentionPlan(mode=mode, segments=seq.segments)
        cached = mode in ("fill", "read")
        for i, block in enumerate(self.transformer_blocks):
            bufs = None
            if cached:
                bufs = kv.buffers(
                    ("block", i), mode,
                    (B, self.num_heads, n_tokens, self.head_dim),
                    hidden.dtype, hidden.device,
                )
            hidden = block(hidden, *mods, pe, replace(plan, bufs=bufs))
        if cached and mode == "fill":
            kv.set_filled()

        out = self.proj_out(self.norm_out(hidden[:, prefix_len:], temb[:1]))
        return out.transpose(1, 2).reshape(B, self.out_channels, H, W)

    @property
    def _dtype(self) -> torch.dtype:
        return self.img_in.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.img_in.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


__all__ = [
    "AttentionPlan",
    "QwenImage21Params",
    "QwenImage21Sequence",
    "QwenImage21Transformer2DModel",
    "modulation_rows",
]
