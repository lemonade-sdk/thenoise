from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from thenoise.dit.feed_forward import FeedForward
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import attention
from thenoise.utils.fourier import fourier_dim, fourier_features
from thenoise.utils.loader import load_dit
from thenoise.utils.qk_norm import QKNorm

from .keys import (
    CODEC_PREFIX,
    Hunyuan3DCodecParams,
    codec_key_map,
    take_prefix,
)

COORDS = 3


def _heads(fused: torch.Tensor, num_heads: int, parts: int) -> Tuple[torch.Tensor, ...]:
    """``[B, N, parts*heads*d]`` -> ``parts`` ``[B, heads, N, d]``, parts inside each head."""
    batches, tokens, total = fused.shape
    per = total // (num_heads * parts)
    grouped = fused.view(batches, tokens, num_heads, parts, per)
    return tuple(head.transpose(1, 2) for head in torch.unbind(grouped, dim=3))


class CodecSelfAttention(nn.Module):
    """Fused-QKV self-attention, with a LayerNorm on the query and key heads."""

    def __init__(self, width: int, num_heads: int, qkv_bias: bool, qk_norm: bool,
                 eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.c_qkv = QuantizedLinear(width, width * 3, bias=qkv_bias)
        self.c_proj = QuantizedLinear(width, width)
        self.attention = (QKNorm(width // num_heads, eps=eps, norm=nn.LayerNorm)
                          if qk_norm else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = _heads(self.c_qkv(x), self.num_heads, 3)
        if self.attention is not None:
            q, k = self.attention(q, k)
        return self.c_proj(attention(q, k=k, v=v))


class CodecCrossAttention(nn.Module):
    """Cross-attention over the latent tokens; K and V are one fused projection."""

    def __init__(self, width: int, num_heads: int, qkv_bias: bool, qk_norm: bool,
                 eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.c_q = QuantizedLinear(width, width, bias=qkv_bias)
        self.c_kv = QuantizedLinear(width, width * 2, bias=qkv_bias)
        self.c_proj = QuantizedLinear(width, width)
        self.attention = (QKNorm(width // num_heads, eps=eps, norm=nn.LayerNorm)
                          if qk_norm else None)

    def forward(self, x: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        q = _heads(self.c_q(x), self.num_heads, 1)[0]
        k, v = _heads(self.c_kv(data), self.num_heads, 2)
        if self.attention is not None:
            q, k = self.attention(q, k)
        return self.c_proj(attention(q, k=k, v=v))


class SelfBlock(nn.Module):
    """A pre-norm self-attention block over latent tokens."""

    def __init__(self, params: Hunyuan3DCodecParams):
        super().__init__()
        self.ln_1 = nn.LayerNorm(params.width, eps=params.norm_eps)
        self.attn = CodecSelfAttention(params.width, params.num_heads, params.qkv_bias,
                                       params.qk_norm, params.norm_eps)
        self.ln_2 = nn.LayerNorm(params.width, eps=params.norm_eps)
        self.mlp = FeedForward(params.width, mult=params.mlp_ratio, approximate="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class CrossBlock(nn.Module):
    """A pre-norm cross-attention block: queries attend to the latent tokens."""

    def __init__(self, width: int, num_heads: int, qkv_bias: bool, qk_norm: bool,
                 mlp_ratio: int, eps: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, eps=eps)
        self.ln_2 = nn.LayerNorm(width, eps=eps)
        self.attn = CodecCrossAttention(width, num_heads, qkv_bias, qk_norm, eps)
        self.ln_3 = nn.LayerNorm(width, eps=eps)
        self.mlp = FeedForward(width, mult=mlp_ratio, approximate="none")

    def forward(self, x: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        return x + self.mlp(self.ln_3(x))


class GeoDecoder(nn.Module):
    """Queried points -> one SDF logit each, by cross-attending to the latent tokens."""

    def __init__(self, params: Hunyuan3DCodecParams):
        super().__init__()
        self.num_freqs = params.num_freqs
        self.query_proj = QuantizedLinear(fourier_dim(COORDS, params.num_freqs), params.width)
        self.cross_attn_decoder = CrossBlock(params.width, params.num_heads,
                                             params.qkv_bias, params.qk_norm,
                                             params.geo_mlp_ratio, params.norm_eps)
        self.ln_post = (nn.LayerNorm(params.width, eps=params.norm_eps)
                        if params.geo_ln_post else None)
        self.output_proj = QuantizedLinear(params.width, 1)

    def forward(self, latents: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """``latents`` ``[B, N, width]``, ``queries`` ``[B, M, 3]`` -> ``[B, M, 1]``."""
        x = self.query_proj(fourier_features(queries, self.num_freqs).to(latents.dtype))
        x = self.cross_attn_decoder(x, latents)
        if self.ln_post is not None:
            x = self.ln_post(x)
        return self.output_proj(x)


class CodecTransformer(nn.Module):
    """The latent tokens, refined by ``layers`` self-attention blocks."""

    def __init__(self, params: Hunyuan3DCodecParams):
        super().__init__()
        self.resblocks = nn.ModuleList(SelfBlock(params) for _ in range(params.layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class Hunyuan3DShapeCodec(nn.Module):
    """Latent tokens to an SDF at any point; nothing in generation encodes, so there is no encode half."""

    def __init__(self, params: Hunyuan3DCodecParams):
        super().__init__()
        self.params = params
        self.scale_factor = params.scale_factor
        self.post_kl = QuantizedLinear(params.embed_dim, params.width)
        self.transformer = CodecTransformer(params)
        self.geo_decoder = GeoDecoder(params)

    def tokens(self, latents: torch.Tensor) -> torch.Tensor:
        """``[B, N, embed_dim]`` -> ``[B, N, width]``, shared by every query chunk because the tokens are unpositioned."""
        return self.transformer(self.post_kl(latents))

    def sdf(self, tokens: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """``tokens`` ``[B, N, width]``, ``points`` ``[B, M, 3]`` -> ``[B, M]`` SDF."""
        return self.geo_decoder(tokens, points).squeeze(-1)

    def forward(self, latents: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """``latents`` ``[B, N, embed_dim]``, ``points`` ``[B, M, 3]`` -> ``[B, M]`` SDF."""
        return self.sdf(self.tokens(latents), points)


def load_hunyuan3d_shape_codec(codec_path: str, params: Hunyuan3DCodecParams, device,
                               dtype: torch.dtype) -> Hunyuan3DShapeCodec:
    """Build on meta from the header's geometry, then load the codec's decode half."""
    with init_empty_weights():
        model = Hunyuan3DShapeCodec(params)
    load_dit(model, codec_path, device=device, dtype=None,
             drop_keys=(CODEC_PREFIX + "encoder.", CODEC_PREFIX + "pre_kl."),
             state_map=take_prefix(CODEC_PREFIX), key_map=codec_key_map)
    model.to(dtype=dtype)
    return model


__all__ = [
    "CodecCrossAttention",
    "CodecSelfAttention",
    "CodecTransformer",
    "CrossBlock",
    "GeoDecoder",
    "Hunyuan3DShapeCodec",
    "SelfBlock",
    "load_hunyuan3d_shape_codec",
]
