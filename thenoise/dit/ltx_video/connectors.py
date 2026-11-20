# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Lightricks and Hugging Face contributors.
"""Native LTX 2.3/2.5 Gemma text connectors, no Diffusers import.

Keep checkpoint ABI and TheNoise's native AV attention/FFN. There is no second
attention implementation, loader, tokenizer or offloading machinery here.
Numerical flow follows the Apache-2 LTX2TextConnectors reference.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from thenoise.dit.quantized import QuantizedLinear

from thenoise.dit.ltx_video.transformer import _AVRoPE, _FeedForward, _LTXAttention


class _ConnectorBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, gated: bool):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn1 = _LTXAttention(dim, dim, heads, head_dim, gated)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = _FeedForward(dim, True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x), mask=mask, q_rope=rope)
        return x + self.ff(self.norm2(x))


class _Connector(nn.Module):
    def __init__(self, heads: int, head_dim: int, layers: int,
                 registers: int, rope_base: int, theta: float,
                 double_precision: bool, gated: bool):
        super().__init__()
        if registers <= 0:
            raise ValueError("LTX connector needs positive learned register count")
        dim = heads * head_dim
        self.num_learnable_registers = registers
        self.learnable_registers = nn.Parameter(torch.rand(registers, dim) * 2 - 1)
        # Connector positions use token index / 4096, without midpoint sampling.
        # _AVRoPE implements exactly that fractional LTX split-half rotation.
        self.rope = _AVRoPE(dim, heads, modality="audio", base_num_frames=rope_base,
                            scale_factors=(1,), theta=theta,
                            double_precision=double_precision)
        self.transformer_blocks = nn.ModuleList([
            _ConnectorBlock(dim, heads, head_dim, gated) for _ in range(layers)
        ])
        self.norm_out = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, dim = x.shape
        if seq % self.num_learnable_registers:
            raise ValueError("LTX text sequence must be divisible by learnable register count")
        if mask.ndim != 4 or mask.shape != (batch, 1, 1, seq):
            raise ValueError("LTX connector expects additive attention mask [B,1,1,T]")
        valid = (mask >= -9000).int().squeeze(1).squeeze(1)
        order = torch.argsort(1 - valid, dim=1, stable=True)
        aligned = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, dim))
        count = valid.sum(1, keepdim=True)
        pos = torch.arange(seq, device=x.device).unsqueeze(0)
        front = (pos < count).unsqueeze(-1)
        registers = self.learnable_registers.repeat(seq // self.num_learnable_registers, 1)
        registers = registers[None].expand(batch, -1, -1)
        x = torch.where(front, aligned, registers.to(x.dtype))
        mask = torch.zeros_like(mask)  # All tokens now usable, including the registers.
        pos = torch.arange(seq, device=x.device, dtype=torch.float32)[None, None].expand(batch, 1, -1)
        rope = self.rope(pos)
        for block in self.transformer_blocks:
            x = block(x, mask, rope)
        return self.norm_out(x), mask


class LTX2TextConnectors(nn.Module):
    """Native LTX text projection + video/audio connector pair.

    Constructor signature follows the existing `make_connectors` dictionary.
    The feature-aggregation and module attribute names are checkpoint ABI.
    """
    def __init__(self, caption_channels: int, text_proj_in_factor: int,
                 video_connector_num_attention_heads: int = 32,
                 video_connector_attention_head_dim: int = 128,
                 video_connector_num_layers: int = 8,
                 video_connector_num_learnable_registers: int = 128,
                 video_gated_attn: bool = True,
                 audio_connector_num_attention_heads: int = 32,
                 audio_connector_attention_head_dim: int = 64,
                 audio_connector_num_layers: int = 8,
                 audio_connector_num_learnable_registers: int = 128,
                 audio_gated_attn: bool = True,
                 connector_rope_base_seq_len: int = 4096,
                 rope_theta: float = 10000.0,
                 rope_double_precision: bool = True,
                 causal_temporal_positioning: bool = False,
                 rope_type: str = "split",
                 per_modality_projections: bool = True,
                 video_hidden_dim: int = 4096,
                 audio_hidden_dim: int = 2048,
                 proj_bias: bool = True):
        super().__init__()
        if not per_modality_projections or rope_type != "split" or causal_temporal_positioning:
            raise ValueError("native LTX connectors support only distilled LTX 2.3/2.5")
        self.config = SimpleNamespace(
            caption_channels=caption_channels, text_proj_in_factor=text_proj_in_factor,
            video_hidden_dim=video_hidden_dim, audio_hidden_dim=audio_hidden_dim,
            per_modality_projections=per_modality_projections,
        )
        dim = caption_channels * text_proj_in_factor
        self.video_text_proj_in = QuantizedLinear(dim, video_hidden_dim, bias=proj_bias)
        self.audio_text_proj_in = QuantizedLinear(dim, audio_hidden_dim, bias=proj_bias)
        self.video_connector = _Connector(
            video_connector_num_attention_heads, video_connector_attention_head_dim,
            video_connector_num_layers, video_connector_num_learnable_registers,
            connector_rope_base_seq_len, rope_theta, rope_double_precision, video_gated_attn,
        )
        self.audio_connector = _Connector(
            audio_connector_num_attention_heads, audio_connector_attention_head_dim,
            audio_connector_num_layers, audio_connector_num_learnable_registers,
            connector_rope_base_seq_len, rope_theta, rope_double_precision, audio_gated_attn,
        )

    @property
    def dtype(self) -> torch.dtype:
        # Choose an unquantized normalisation tensor rather than the ConvRot
        # projection: QuantizedTensor storage dtype may be INT8.
        return self.video_connector.transformer_blocks[0].attn1.norm_q.weight.dtype

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor,
                padding_side: str = "left", scale_factor: int = 8
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geometry = (self.config.caption_channels, self.config.text_proj_in_factor)
        if hidden.ndim == 3:
            if hidden.shape[-1] != math.prod(geometry):
                raise ValueError(f"LTX text features have {hidden.shape[-1]} channels, expected {math.prod(geometry)}")
            hidden = hidden.unflatten(2, geometry)
        if hidden.ndim != 4 or tuple(hidden.shape[2:]) != geometry:
            raise ValueError(f"LTX text connector expects [B,T,{geometry[0]},{geometry[1]}] Gemma features")
        if attention_mask.ndim != 2 or attention_mask.shape != hidden.shape[:2]:
            raise ValueError("LTX text connector needs matching binary [B,T] mask")
        # LTX 2.3/2.5 use per-token RMS over the hidden-channel axis, not
        # per-layer masked/range norm used by legacy LTX 2.0.
        norm = hidden * torch.rsqrt(torch.mean(hidden.square(), dim=2, keepdim=True) + 1e-6)
        packed = norm.flatten(2, 3)
        packed = torch.where(attention_mask.bool().unsqueeze(-1), packed, torch.zeros_like(packed))
        vscale = math.sqrt(self.config.video_hidden_dim / self.config.caption_channels)
        ascale = math.sqrt(self.config.audio_hidden_dim / self.config.caption_channels)
        v = self.video_text_proj_in(packed * vscale)
        a = self.audio_text_proj_in(packed * ascale)
        # Learned-register attention takes additive padding bias.
        mask = (attention_mask.to(torch.int64) - 1).to(v.dtype)
        mask = mask[:, None, None, :] * torch.finfo(v.dtype).max
        v, vmask = self.video_connector(v, mask)
        vmask = (vmask < 1e-6).to(torch.int64).reshape(v.shape[0], v.shape[1], 1)
        v = v * vmask
        a, _ = self.audio_connector(a, mask)
        return v, a, vmask.squeeze(-1)


__all__ = ["LTX2TextConnectors"]
