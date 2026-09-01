"""Flux.2 (Flux Klein) DiT architecture — single/double-stream MMDiT.

Ported from kohya-ss/musubi-tuner's ``flux2_models`` (itself a copy of the Black
Forest Labs FLUX repo, Apache-2.0). Only the still-image text-to-image path is
kept: no gradient checkpointing, no block swapping, no training hooks. Weight key
names are unchanged so the official ``flux2-klein-*.safetensors`` checkpoints load
as-is. Attention uses this engine's shared SDPA helper
(``thenoise.utils.attention``), which handles the ``[B, L, H, D]`` layout.

Flux Klein is the Flux.2 family: a packed 128-channel latent (the Flux.2 VAE packs
32ch -> 128ch via a 2x2 patchify), a 4-axis RoPE, and Qwen3 text conditioning. The
Klein variants (4B / 9B) share the architecture but differ in width/depth and
context width; they are selected at load time from the checkpoint's ``img_in``
width.
"""

# Copyright 2023 The HuggingFace Team. Licensed under the Apache-2.0 License.
# Copyright 2024 Black Forest Labs. Flux is released under the Apache-2.0 License.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from einops import rearrange
from torch import Tensor, nn

from thenoise.dit.quantized import QuantizedLinear
from thenoise.dit.reference import concat_reference, slice_reference_output
from thenoise.utils.attention import attention as sdpa_attention
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


@dataclass
class Flux2Params:
    """Shared Flux.2 architecture knobs (dev defaults)."""

    in_channels: int = 128  # packed latent channels
    context_in_dim: int = 15360
    hidden_size: int = 6144
    num_heads: int = 48
    depth: int = 8
    depth_single_blocks: int = 48
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = True


@dataclass
class Klein9BParams(Flux2Params):
    context_in_dim: int = 12288  # 3 * Qwen3-8B hidden (4096)
    hidden_size: int = 4096
    num_heads: int = 32
    depth: int = 8
    depth_single_blocks: int = 24
    use_guidance_embed: bool = False


@dataclass
class Klein4BParams(Flux2Params):
    context_in_dim: int = 7680  # 3 * Qwen3-4B hidden (2560)
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    use_guidance_embed: bool = False


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    """Sinusoidal timestep embedding (scaled by ``time_factor``, 1000.0)."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """Rotary embedding for one of the 4 position axes."""
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(qkv_list: list[Tensor], pe: Tensor) -> Tensor:
    """Apply RoPE then the shared SDPA attention, returning ``[B, L, H*D]``."""
    q, k, v = qkv_list
    q, k = apply_rope(q, k, pe)
    q = q.transpose(1, 2)  # B, H, L, D -> B, L, H, D
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    return sdpa_attention([q, k, v])


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = QuantizedLinear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = QuantizedLinear(hidden_dim, hidden_dim, bias=not disable_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))], dim=-3)
        return emb.unsqueeze(1)


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = QuantizedLinear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: torch.Tensor):
        org_dtype = vec.dtype
        vec = vec.to(torch.float32)  # numerical stability
        out = self.lin(nn.functional.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.to(org_dtype)
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = QuantizedLinear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), QuantizedLinear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        org_dtype = x.dtype
        vec = vec.to(torch.float32)  # numerical stability
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = x.to(torch.float32)  # numerical stability
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x.to(org_dtype)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = QuantizedLinear(dim, dim * 3, bias=False)
        self.norm = QKNorm(head_dim)
        self.proj = QuantizedLinear(dim, dim, bias=False)


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2

        self.linear1 = QuantizedLinear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=False)
        self.linear2 = QuantizedLinear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor]) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor], dim=-1
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = attention([q, k, v], pe)

        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_mult_factor = 2

        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            QuantizedLinear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            QuantizedLinear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            QuantizedLinear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            QuantizedLinear(mlp_hidden_dim, hidden_size, bias=False),
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[Tensor, Tensor],
        mod_txt: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        # image attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # text attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        txt_len = txt_q.shape[2]
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        pe = torch.cat((pe_ctx, pe), dim=2)
        attn = attention([q, k, v], pe)
        txt_attn, img_attn = attn[:, :txt_len], attn[:, txt_len:]

        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp((1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift)

        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp((1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift)
        return img, txt


class Flux2(nn.Module):
    """Flux.2 transformer (Flux Klein). Text-to-image path only."""

    def __init__(self, params: Flux2Params) -> None:
        super().__init__()
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"hidden_size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = QuantizedLinear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
        self.txt_in = QuantizedLinear(params.context_in_dim, self.hidden_size, bias=False)

        self.use_guidance_embed = params.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio) for _ in range(params.depth)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio) for _ in range(params.depth_single_blocks)]
        )

        self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)

        self.final_layer = LastLayer(self.hidden_size, self.out_channels)

        self.num_double_blocks = len(self.double_blocks)
        self.num_single_blocks = len(self.single_blocks)

        # Per-block compiled-forward cache: ``(blocks, index, dynamic)``. t2i
        # compiles static (fixed shape); edit (variable ref count) compiles
        # dynamic to avoid Inductor's recompile/`CantSplit` failure on the grown
        # sequence. Keyed by the owning ModuleList + position (not ``id(block)``),
        # which also keeps the double vs single streams from colliding.
        self._compiled_blocks: dict[tuple[nn.ModuleList, int, bool], Callable] = {}

    def _compile_block(self, blocks: nn.ModuleList, index: int, dynamic: bool) -> Callable:
        """Cached ``torch.compile``d forward for ``blocks[index]``."""
        block = blocks[index]
        key = (blocks, index, dynamic)
        fn = self._compiled_blocks.get(key)
        if fn is None:
            fn = torch.compile(block.forward, fullgraph=True, dynamic=dynamic)
            self._compiled_blocks[key] = fn
        return fn

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None = None,
        ref_tokens: Tensor | None = None,
        ref_ids: Tensor | None = None,
    ) -> Tensor:
        num_txt_tokens = ctx.shape[1]
        num_img_tokens = x.shape[1]

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        if self.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        # Reference-latent editing (Flux2 Klein): append the reference image
        # tokens+ids to the image stream (in packed-latent space) so the DiT
        # attends to them alongside the text instruction. ``None`` (plain t2i)
        # is a no-op. Must happen before ``img_in``/``pe_embedder`` so the refs
        # are embedded like the image tokens.
        x, x_ids = concat_reference(x, x_ids, ref_tokens, ref_ids)

        img = self.img_in(x)
        txt = self.txt_in(ctx)
        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        # Edit varies seq length (ref tokens appended), so compile blocks
        # dynamically there; t2i keeps static-specialized kernels.
        dynamic = ref_tokens is not None or ref_ids is not None

        for i in range(len(self.double_blocks)):
            fwd = self._compile_block(self.double_blocks, i, dynamic)
            img, txt = fwd(img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt)

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        for i in range(len(self.single_blocks)):
            fwd = self._compile_block(self.single_blocks, i, dynamic)
            img = fwd(img, pe, single_block_mod)

        img = img.to(vec.device)
        img = img[:, num_txt_tokens:, ...]  # drop the text tokens
        img = slice_reference_output(img, num_img_tokens)  # drop ref tokens
        img = self.final_layer(img, vec)
        return img


__all__ = [
    "Flux2",
    "Flux2Params",
    "Klein4BParams",
    "Klein9BParams",
]
