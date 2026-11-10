"""Wan 2.2 TI2V-5B transformer using TheNoise attention and RoPE."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from thenoise.utils.attention import attention
from thenoise.utils.rope import apply_rope


@dataclass
class WanNativeOutput:
    sample: Tensor


def _timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Wan's cos/sin embedding (Diffusers Timesteps, flip_sin_to_cos=True)."""
    half = dim // 2
    scale = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angle = t.reshape(-1, 1).float() * scale[None]
    embedding = torch.cat((torch.cos(angle), torch.sin(angle)), dim=-1)
    return F.pad(embedding, (0, dim % 2))


class WanNativeRoPE(nn.Module):
    """Repackage checkpoint frequency tables into TheNoise's adjacent-pair RoPE.

    This preserves Diffusers' FP64-generated frequency phases already stored in
    FP32 buffers; recreating the frequencies in FP32 causes avoidable drift.
    """

    def __init__(self, weights: nn.Module):
        super().__init__()
        self.patch_size = tuple(weights.patch_size)
        self.t_dim = int(weights.t_dim)
        self.h_dim = int(weights.h_dim)
        self.w_dim = int(weights.w_dim)
        self.register_buffer("freqs_cos", weights.freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", weights.freqs_sin, persistent=False)

    def forward(self, latents: Tensor) -> Tensor:
        _, _, t, h, w = latents.shape
        pt, ph, pw = self.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError("Wan native DiT needs latent geometry divisible by patch size")
        t, h, w = t // pt, h // ph, w // pw
        sizes = (self.t_dim, self.h_dim, self.w_dim)
        cos_t, cos_h, cos_w = self.freqs_cos.split(sizes, dim=-1)
        sin_t, sin_h, sin_w = self.freqs_sin.split(sizes, dim=-1)
        if t > cos_t.shape[0] or h > cos_h.shape[0] or w > cos_w.shape[0]:
            raise ValueError("Wan 3D RoPE geometry exceeds checkpoint table")
        cos = torch.cat((
            cos_t[:t, None, None].expand(t, h, w, -1),
            cos_h[None, :h, None].expand(t, h, w, -1),
            cos_w[None, None, :w].expand(t, h, w, -1),
        ), dim=-1).reshape(t * h * w, -1)
        sin = torch.cat((
            sin_t[:t, None, None].expand(t, h, w, -1),
            sin_h[None, :h, None].expand(t, h, w, -1),
            sin_w[None, None, :w].expand(t, h, w, -1),
        ), dim=-1).reshape(t * h * w, -1)
        # TheNoise apply_rope consumes adjacent 2x2 matrices, [B, N, D/2, 2, 2].
        # Diffusers' table duplicates each phase across the two components.
        cs = cos[:, ::2]
        sn = sin[:, 1::2]
        return torch.stack((cs, -sn, sn, cs), dim=-1).reshape(1, t * h * w, -1, 2, 2)


class WanNativeAttention(nn.Module):
    """Wan QK normalization and attention via TheNoise's shared SDPA."""

    def __init__(self, weights: nn.Module):
        super().__init__()
        self.heads = int(weights.heads)
        self.to_q = weights.to_q
        self.to_k = weights.to_k
        self.to_v = weights.to_v
        self.norm_q = weights.norm_q
        self.norm_k = weights.norm_k
        self.to_out = weights.to_out
        if weights.add_k_proj is not None or weights.add_v_proj is not None:
            raise ValueError("Wan TI2V-5B cannot use CLIP-image cross-attention")

    def forward(self, x: Tensor, context: Tensor | None, rope: Tensor | None) -> Tensor:
        source = x if context is None else context
        b, n, _ = x.shape
        q = self.norm_q(self.to_q(x)).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k = self.norm_k(self.to_k(source)).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        v = self.to_v(source).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, k, rope)
        out = attention([q, k, v])  # TheNoise: [B, N, H * D]
        return self.to_out[1](self.to_out[0](out))


class WanNativeBlock(nn.Module):
    def __init__(self, weights: nn.Module):
        super().__init__()
        self.norm1 = weights.norm1
        self.norm2 = weights.norm2
        self.norm3 = weights.norm3
        self.attn1 = WanNativeAttention(weights.attn1)
        self.attn2 = WanNativeAttention(weights.attn2)
        self.ffn_in = weights.ffn.net[0].proj
        self.ffn_out = weights.ffn.net[2]
        self.ffn_dropout = weights.ffn.net[1]
        self.scale_shift_table = weights.scale_shift_table

    def forward(self, x: Tensor, context: Tensor, temb: Tensor, rope: Tensor) -> Tensor:
        # I2V supplies [B, N, 6, D]; T2V supplies [B, 6, D].
        if temb.ndim == 4:
            mod = (self.scale_shift_table.unsqueeze(0) + temb.float()).unbind(dim=2)
        else:
            mod = (self.scale_shift_table + temb.float()).chunk(6, dim=1)
        shift, scale, gate, shift_ff, scale_ff, gate_ff = mod

        y = (self.norm1(x.float()) * (1 + scale) + shift).to(x.dtype)
        x = (x.float() + self.attn1(y, None, rope) * gate).to(x.dtype)
        x = x + self.attn2(self.norm2(x.float()).to(x.dtype), context, None)
        y = (self.norm3(x.float()) * (1 + scale_ff) + shift_ff).to(x.dtype)
        y = self.ffn_out(self.ffn_dropout(F.gelu(self.ffn_in(y), approximate="tanh")))
        return (x.float() + y.float() * gate_ff).to(x.dtype)


class WanNativeCondition(nn.Module):
    def __init__(self, weights: nn.Module):
        super().__init__()
        self.freq_dim = int(weights.timesteps_proj.num_channels)
        self.time_linear_1 = weights.time_embedder.linear_1
        self.time_linear_2 = weights.time_embedder.linear_2
        self.time_proj = weights.time_proj
        self.text_linear_1 = weights.text_embedder.linear_1
        self.text_linear_2 = weights.text_embedder.linear_2
        if weights.image_embedder is not None:
            raise ValueError("Wan TI2V-5B does not use a separate image encoder")

    def forward(self, timestep: Tensor, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tokens = timestep.shape[1] if timestep.ndim == 2 else None
        ts = _timestep_embedding(timestep, self.freq_dim)
        if tokens is not None:
            ts = ts.unflatten(0, (-1, tokens))
        ts = ts.to(dtype=self.time_linear_1.weight.dtype)
        temb = self.time_linear_2(F.silu(self.time_linear_1(ts))).to(dtype=context.dtype)
        mod = self.time_proj(F.silu(temb))
        mod = mod.unflatten(-1, (6, -1))
        ctx = self.text_linear_2(F.gelu(self.text_linear_1(context), approximate="tanh"))
        return temb, mod, ctx


class WanNativeDiT(nn.Module):
    """Wan 2.2 transformer using independently loaded checkpoint weights."""

    def __init__(self, weights: nn.Module):
        super().__init__()
        cfg = weights.config
        if (cfg.in_channels != 48 or cfg.out_channels != 48 or
                getattr(cfg, "image_dim", None) is not None or
                getattr(cfg, "added_kv_proj_dim", None) is not None):
            raise ValueError("native Wan DiT supports only 48-channel TI2V-5B")
        self.config = cfg
        self.rope = WanNativeRoPE(weights.rope)
        self.patch_embedding = weights.patch_embedding
        self.condition_embedder = WanNativeCondition(weights.condition_embedder)
        self.blocks = nn.ModuleList([WanNativeBlock(block) for block in weights.blocks])
        self.norm_out = weights.norm_out
        self.proj_out = weights.proj_out
        self.scale_shift_table = weights.scale_shift_table
        self.eval()

    @classmethod
    def from_weights(cls, weights: nn.Module) -> "WanNativeDiT":
        if getattr(weights, "training", False):
            weights.eval()
        return cls(weights)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embedding.weight.device

    def cache_context(self, *_args, **_kwargs):
        # No cross-step caching: Wan 5B denoising changes every latent token.
        return nullcontext()

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_hidden_states_image: Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict | None = None,
    ):
        if encoder_hidden_states_image is not None:
            raise ValueError("TI2V-5B has no CLIP-image embedding path")
        if attention_kwargs:
            raise ValueError("native Wan DiT does not support attention adapters yet")
        b, c, t, h, w = hidden_states.shape
        pt, ph, pw = tuple(self.config.patch_size)
        if c != 48 or t % pt or h % ph or w % pw:
            raise ValueError("invalid 48-channel Wan latent geometry")

        rope = self.rope(hidden_states)
        x = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2).contiguous()
        temb, modulation, ctx = self.condition_embedder(timestep, encoder_hidden_states)
        if timestep.ndim == 2 and timestep.shape != (b, x.shape[1]):
            raise ValueError("I2V token timesteps must match patchified latent tokens")
        for block in self.blocks:
            x = block(x, ctx, modulation, rope)
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).unbind(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        x = (self.norm_out(x.float()) * (1 + scale) + shift).to(x.dtype)
        x = self.proj_out(x)
        x = x.reshape(b, t // pt, h // ph, w // pw, pt, ph, pw, -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return WanNativeOutput(x) if return_dict else (x,)
