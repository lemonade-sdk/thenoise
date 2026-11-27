from __future__ import annotations

from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import attention
from thenoise.utils.loader import build_dit, take_prefix
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.rope import apply_rope, matrix_rope
from thenoise.utils.timestep import timestep_embedding

from .keys import (
    ADALN_CHUNKS,
    ROPE_AXES,
    ROPE_THETA,
    Pixal3DStageParams,
    pixal3d_key_map,
    stage_params,
    stage_prefix,
)
from thenoise.utils.safetensors import Header, as_header

#: The gain divides by the head's own norm; the checkpoint has no epsilon.
HEAD_NORM_EPS = 1e-12


def rope_matrices(coords: torch.Tensor, head_dim: int, axes: int = ROPE_AXES,
                  theta: float = ROPE_THETA) -> torch.Tensor:
    """RoPE for integer grid coordinates, with ``axes`` axes sharing ``head_dim / 2`` pairs."""
    per_axis = head_dim // 2 // axes
    freqs = matrix_rope([2 * per_axis] * axes, theta)(coords[None].float())
    spare = head_dim // 2 - axes * per_axis
    if spare:  # pairs a head has left over after the axes stay unrotated, as trained
        identity = torch.eye(2, dtype=freqs.dtype, device=coords.device).expand(
            1, coords.shape[0], spare, 2, 2)
        freqs = torch.cat([freqs, identity], dim=-3)
    return freqs


class TimestepEmbedder(nn.Module):
    """Sinusoidal features of the flow timestep, then a two-layer MLP."""

    def __init__(self, hidden_size: int, features: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(QuantizedLinear(features, hidden_size), nn.SiLU(),
                                 QuantizedLinear(hidden_size, hidden_size))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(timestep, self.mlp[0].in_features,
                                          time_factor=1.0))


class SelfAttention(nn.Module):
    """Attention within the latent tokens, rotated by their grid position."""

    def __init__(self, params: Pixal3DStageParams) -> None:
        super().__init__()
        width = params.hidden_size
        self.heads = params.heads
        self.to_qkv = QuantizedLinear(width, 3 * width)
        self.qk_norm = QKNorm(params.head_dim, eps=HEAD_NORM_EPS,
                              norm=partial(RMSNorm, heads=params.heads))
        self.to_out = QuantizedLinear(width, width)

    def forward(self, feats: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        q, k, v = _split(self.to_qkv(feats), 3, self.heads)
        q, k = self.qk_norm(q, k)
        q, k = apply_rope(_heads(q), _heads(k), rope)
        return attention(q, k, _heads(v))


class CrossAttention(nn.Module):
    """Attention from the latent tokens to the image tokens."""

    def __init__(self, params: Pixal3DStageParams) -> None:
        super().__init__()
        width = params.hidden_size
        self.heads = params.heads
        self.to_q = QuantizedLinear(width, width)
        self.to_kv = QuantizedLinear(params.cond_channels, 2 * width)
        self.qk_norm = QKNorm(params.head_dim, eps=HEAD_NORM_EPS,
                              norm=partial(RMSNorm, heads=params.heads))
        self.to_out = QuantizedLinear(width, width)

    def forward(self, feats: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = _split(self.to_q(feats), 1, self.heads)[0]
        k, v = _split(self.to_kv(context), 2, self.heads)
        q, k = self.qk_norm(q, k)
        return attention(_heads(q), _heads(k), _heads(v))


class ProjectAttention(nn.Module):
    """Image tokens cross-attended plus the image features projected onto each latent token."""

    def __init__(self, params: Pixal3DStageParams) -> None:
        super().__init__()
        self.cross_attn_block = CrossAttention(params)
        self.proj_linear = QuantizedLinear(params.proj_channels, params.hidden_size)

    def forward(self, feats: torch.Tensor, context: torch.Tensor,
                projection: torch.Tensor) -> torch.Tensor:
        return self.cross_attn_block(feats, context) + self.proj_linear(projection)


class BlockFeedForward(nn.Module):
    """Two projections around a GELU, filed as ``mlp.mlp.0`` and ``mlp.mlp.2``."""

    def __init__(self, params: Pixal3DStageParams) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            QuantizedLinear(params.hidden_size, params.mlp_dim),
            nn.GELU(),
            QuantizedLinear(params.mlp_dim, params.hidden_size),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(feats)


class StageBlock(nn.Module):
    """Self-attention, image attention, feed-forward; the first and last AdaLN-modulated."""

    def __init__(self, params: Pixal3DStageParams) -> None:
        super().__init__()
        width = params.hidden_size
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.norm3 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(params)
        self.cross_attn = ProjectAttention(params)
        self.mlp = BlockFeedForward(params)
        self.modulation = nn.Parameter(torch.zeros(ADALN_CHUNKS * width))

   @torch.compile(fullgraph=True)
    def forward(self, feats: torch.Tensor, rope: torch.Tensor, timestep: torch.Tensor,
                context: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            part.unsqueeze(1) for part in (self.modulation + timestep).chunk(ADALN_CHUNKS, dim=1))
        feats = feats + self.self_attn(
            self.norm1(feats) * (1 + scale_msa) + shift_msa, rope) * gate_msa
        feats = feats + self.cross_attn(self.norm2(feats), context, projection)
        feats = feats + self.mlp(
            self.norm3(feats) * (1 + scale_mlp) + shift_mlp) * gate_mlp
        return feats


class Pixal3DStage(nn.Module):
    """One stage of the cascade; ``concat`` carries another latent in alongside."""

    def __init__(self, params: Pixal3DStageParams) -> None:
        super().__init__()
        if not params.shared_modulation:
            raise ValueError("this file modulates every block separately, which is not shipped")
        width = params.hidden_size
        self.params = params
        self.t_embedder = TimestepEmbedder(width, params.timestep_features)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),
                                              QuantizedLinear(width, ADALN_CHUNKS * width))
        self.input_layer = nn.Linear(params.in_channels, width)
        self.blocks = nn.ModuleList(StageBlock(params) for _ in range(params.depth))
        self.out_layer = nn.Linear(width, params.out_channels)

    def forward(self, feats: torch.Tensor, coords: torch.Tensor, timestep: torch.Tensor,
                context: torch.Tensor, projection: torch.Tensor,
                concat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if concat is not None:
            feats = torch.cat([feats, concat], dim=-1)
        feats = self.input_layer(feats)
        rope = rope_matrices(coords, self.params.head_dim)
        modulated = self.adaLN_modulation(self.t_embedder(timestep))
        for block in self.blocks:
            feats = block(feats, rope, modulated, context, projection)
        feats = F.layer_norm(feats, feats.shape[-1:])
        return self.out_layer(feats)


def load_pixal3d_stage(checkpoint: Union[str, Header], stage: str, device,
                       dtype: Optional[torch.dtype] = None) -> Pixal3DStage:
    """Build on meta from the shared file's geometry, then load one stage out of it."""
    model = build_dit(lambda: Pixal3DStage(stage_params(as_header(checkpoint), stage)),
                      checkpoint, device=device, dtype=None,
                      state_map=take_prefix(stage_prefix(stage)), key_map=pixal3d_key_map)
    return model.to(dtype=dtype)


def _split(feats: torch.Tensor, parts: int, heads: int) -> list[torch.Tensor]:
    """``[B, N, parts * heads * dim]`` into ``parts`` of ``[B, N, heads, dim]``."""
    batch, tokens, width = feats.shape
    dim = width // parts // heads
    return [part for part in feats.view(batch, tokens, parts, heads, dim).unbind(dim=2)]


def _heads(tokens: torch.Tensor) -> torch.Tensor:
    """``[B, N, heads, dim]`` into ``[B, heads, N, dim]``, the order the kernels read."""
    return tokens.transpose(1, 2)


__all__ = [
    "CrossAttention",
    "BlockFeedForward",
    "Pixal3DStage",
    "ProjectAttention",
    "SelfAttention",
    "StageBlock",
    "TimestepEmbedder",
    "load_pixal3d_stage",
    "rope_matrices",
]
