from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from thenoise.dit.feed_forward import FeedForward
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import attention
from thenoise.utils.loader import load_dit
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.timestep import timestep_embedding

from .keys import DIT_PREFIX, Hunyuan3DShapeParams, shape_key_map, take_prefix


def _heads(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``[B, L, H*D]`` -> ``[B, H, L, D]``, the layout SDPA wants."""
    batches, tokens, _ = value.shape
    return value.view(batches, tokens, num_heads, -1).transpose(1, 2)


class Attention(nn.Module):
    """Attention over latent tokens; a ``context`` makes it cross-attention."""

    def __init__(self, dim: int, context_dim: int, num_heads: int, qkv_bias: bool,
                 qk_norm_eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.to_q = QuantizedLinear(dim, dim, bias=qkv_bias)
        self.to_k = QuantizedLinear(context_dim, dim, bias=qkv_bias)
        self.to_v = QuantizedLinear(context_dim, dim, bias=qkv_bias)
        self.qk_norm = QKNorm(dim // num_heads, eps=qk_norm_eps)
        self.out_proj = QuantizedLinear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        source = x if context is None else context
        q = _heads(self.to_q(x), self.num_heads)
        k = _heads(self.to_k(source), self.num_heads)
        v = _heads(self.to_v(source), self.num_heads)
        q, k = self.qk_norm(q, k)
        return self.out_proj(attention(q, k=k, v=v))


class MoE(nn.Module):
    """Top-k experts weighted unrenormalized, plus a shared expert added to every token."""

    def __init__(self, dim: int, num_experts: int, top_k: int, mult: int):
        super().__init__()
        self.top_k = top_k
        self.gate = QuantizedLinear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForward(dim, mult=mult, approximate="none") for _ in range(num_experts)
        )
        self.shared_experts = FeedForward(dim, mult=mult, approximate="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.reshape(-1, x.shape[-1])
        weights, chosen = self.gate(tokens).softmax(dim=-1).topk(self.top_k, dim=-1)
        out = self.shared_experts(tokens)
        for expert_id, expert in enumerate(self.experts):
            picked = chosen == expert_id
            rows = picked.any(dim=1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            scale = (weights[rows] * picked[rows]).sum(dim=1, keepdim=True)
            out.index_add_(0, rows, expert(tokens.index_select(0, rows)) * scale)
        return out.view_as(x)


class ShapeBlock(nn.Module):
    """Self-attention, cross-attention to the picture, then a feed-forward."""

    def __init__(self, params: Hunyuan3DShapeParams, *, moe: bool, skip: bool):
        super().__init__()
        dim = params.hidden_size
        self.norm1 = nn.LayerNorm(dim, eps=params.norm_eps)
        self.attn1 = Attention(dim, dim, params.num_heads, params.qkv_bias, params.qk_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=params.norm_eps)
        self.attn2 = Attention(dim, params.context_dim, params.num_heads, params.qkv_bias,
                               params.qk_norm_eps)
        self.norm3 = nn.LayerNorm(dim, eps=params.norm_eps)
        self.moe = MoE(dim, params.num_experts, params.top_k, params.mlp_ratio) if moe else None
        self.mlp = None if moe else FeedForward(dim, mult=params.mlp_ratio, approximate="none")
        self.skip_linear = QuantizedLinear(2 * dim, dim) if skip else None
        self.skip_norm = nn.LayerNorm(dim, eps=params.norm_eps) if skip else None

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                skip_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.skip_linear is not None:
            x = self.skip_norm(self.skip_linear(torch.cat([skip_value, x], dim=-1)))
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        ffn = self.moe if self.moe is not None else self.mlp
        return x + ffn(self.norm3(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal features of the timestep, then an MLP: the token that leads the sequence."""

    def __init__(self, hidden_size: int, features: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            QuantizedLinear(hidden_size, features),
            nn.GELU(),
            QuantizedLinear(features, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        sine = timestep_embedding(t, self.hidden_size, time_factor=1.0, sin_first=True)
        return self.mlp(sine.to(self.mlp[0].weight.dtype)).unsqueeze(1)


class FinalLayer(nn.Module):
    """Norm, drop the leading timestep token, project back to latent channels."""

    def __init__(self, dim: int, out_channels: int, eps: float):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, eps=eps)
        self.linear = QuantizedLinear(dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x)[:, 1:])


class Hunyuan3DShapeDiT(nn.Module):
    """Latent tokens, unpositioned, led by a timestep token; ``t`` is 0 at noise, 1 at clean."""

    def __init__(self, params: Hunyuan3DShapeParams):
        super().__init__()
        self.params = params
        self.x_embedder = QuantizedLinear(params.in_channels, params.hidden_size)
        self.t_embedder = TimestepEmbedder(params.hidden_size, params.hidden_size * 4)
        self.blocks = nn.ModuleList(
            ShapeBlock(params, moe=params.uses_moe(layer), skip=params.uses_skip(layer))
            for layer in range(params.depth)
        )
        self.final_layer = FinalLayer(params.hidden_size, params.out_channels, params.norm_eps)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """``x`` ``[B, N, in_channels]``, ``t`` ``[B]``, ``context`` ``[B, L, context_dim]``."""
        x = torch.cat([self.t_embedder(t), self.x_embedder(x)], dim=1)
        half = self.params.depth // 2
        skips: List[torch.Tensor] = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= half else skips.pop()
            x = block(x, context, skip_value)
            if layer < half:
                skips.append(x)
        return self.final_layer(x)


def load_hunyuan3d_shape_dit(dit_path: str, params: Hunyuan3DShapeParams, device,
                            dtype: torch.dtype) -> Hunyuan3DShapeDiT:
    """Build on meta from the header's geometry, then load the ``model.`` subtree."""
    with init_empty_weights():
        model = Hunyuan3DShapeDiT(params)
    load_dit(model, dit_path, device=device, dtype=None,
             state_map=take_prefix(DIT_PREFIX), key_map=shape_key_map)
    model.to(dtype=dtype)
    return model


__all__ = [
    "Attention",
    "FinalLayer",
    "Hunyuan3DShapeDiT",
    "MoE",
    "ShapeBlock",
    "TimestepEmbedder",
    "load_hunyuan3d_shape_dit",
]
