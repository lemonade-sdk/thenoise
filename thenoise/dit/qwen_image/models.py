"""Qwen-Image DiT — dual-stream transformer (parallel image + text joint attention).

Ported from kohya-ss/musubi-tuner's ``qwen_image/qwen_image_model.py`` (itself
Diffusers ``QwenImageTransformer2DModel``), trimmed to inference-only. RoPE uses 3D
image frequencies; ``zero_cond_t`` (edit-2511, flagged by ``__index_timestep_zero__``)
zeroes the timestep on the reference tokens. Weights load via ``load_dit`` (BF16 and
int8_convrot checkpoints).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights

from thenoise.utils.loader import load_dit
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (Diffusers ``get_timestep_embedding``)."""
    half_dim = embedding_dim // 2
    exponent = (
        -math.log(max_period)
        * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        / (half_dim - downscale_freq_shift)
    )
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        self.linear_1 = QuantizedLinear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        out_dim = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = QuantizedLinear(time_embed_dim, out_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: float = 1.0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return _get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps = timestep.to(hidden_states.dtype)
        return self.timestep_embedder(self.time_proj(timesteps))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), eps=self.eps, weight=self.weight)


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, elementwise_affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = QuantizedLinear(embedding_dim, output_dim * 2, bias=True)
        self.norm = nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = QuantizedLinear(dim_in, dim_out, bias=bias)
        self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, bias: bool = True):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        # Dropout is a no-op (p=0) but must stay so ``net.1``/``net.2`` align with the checkpoint.
        self.net = nn.ModuleList(
            [
                GELU(dim, inner_dim, approximate="tanh", bias=bias),
                nn.Dropout(0.0),
                QuantizedLinear(inner_dim, dim_out, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def apply_rotary_emb_qwen(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to ``[B, S, H, D]`` using real ``cos/sin [S, D//2]``.

    Real arithmetic throughout (no ``view_as_complex``, no complex ``freqs``) so the
    ``torch.compile``d block never sees complex tensors -- inductor cannot codegen them.
    """
    cos = cos.to(x.device)
    sin = sin.to(x.device)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    xr = x_real * cos - x_imag * sin
    xi = x_real * sin + x_imag * cos
    return torch.stack([xr, xi], dim=-1).flatten(3).to(x.dtype)


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self._rope_params(pos_index, self.axes_dim[0], self.theta),
                self._rope_params(pos_index, self.axes_dim[1], self.theta),
                self._rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self._rope_params(neg_index, self.axes_dim[0], self.theta),
                self._rope_params(neg_index, self.axes_dim[1], self.theta),
                self._rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}
        self.scale_rope = scale_rope

    def _rope_params(self, index: torch.Tensor, dim: int, theta: float = 10000) -> torch.Tensor:
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(frame * height * width, -1)
        return freqs.clone().contiguous()

    def forward(self, video_fhw, txt_seq_lens, device):
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{frame}_{height}_{width}"
            if rope_key not in self.rope_cache:
                self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
            video_freq = self.rope_cache[rope_key].to(device)
            vid_freqs.append(video_freq)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        if max_vid_index + max_len <= self.pos_freqs.size(0):
            txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        else:
            index = torch.arange(max_vid_index, max_vid_index + max_len, dtype=torch.float32)
            txt_freqs = torch.cat(
                [self._rope_params(index, dim, self.theta) for dim in self.axes_dim],
                dim=1,
            ).to(device)
        vid_freqs = torch.cat(vid_freqs, dim=0)
        # Split the complex ``torch.polar`` freqs into real cos/sin here (outside the
        # compiled block) so the attention kernel never sees complex tensors.
        return (vid_freqs.real, vid_freqs.imag), (txt_freqs.real, txt_freqs.imag)


class Attention(nn.Module):
    """Dual-stream joint attention: image + text QKV, concatenated, one SDPA."""

    def __init__(
        self,
        dim_head: int,
        heads: int,
        out_dim: int,
        added_kv_proj_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.inner_dim = out_dim
        self.inner_kv_dim = out_dim
        self.heads = heads

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_q = QuantizedLinear(out_dim, self.inner_dim, bias=True)
        self.to_k = QuantizedLinear(out_dim, self.inner_kv_dim, bias=True)
        self.to_v = QuantizedLinear(out_dim, self.inner_kv_dim, bias=True)

        self.add_q_proj = QuantizedLinear(added_kv_proj_dim, self.inner_dim, bias=True)
        self.add_k_proj = QuantizedLinear(added_kv_proj_dim, self.inner_kv_dim, bias=True)
        self.add_v_proj = QuantizedLinear(added_kv_proj_dim, self.inner_kv_dim, bias=True)
        self.to_out = nn.ModuleList([QuantizedLinear(self.inner_dim, out_dim, bias=True), nn.Dropout(0.0)])
        self.to_add_out = QuantizedLinear(self.inner_dim, added_kv_proj_dim, bias=True)

        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_query = self.to_q(hidden_states)
        img_key = self.to_k(hidden_states)
        img_value = self.to_v(hidden_states)

        txt_query = self.add_q_proj(encoder_hidden_states)
        txt_key = self.add_k_proj(encoder_hidden_states)
        txt_value = self.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (self.heads, -1))
        img_key = img_key.unflatten(-1, (self.heads, -1))
        img_value = img_value.unflatten(-1, (self.heads, -1))
        txt_query = txt_query.unflatten(-1, (self.heads, -1))
        txt_key = txt_key.unflatten(-1, (self.heads, -1))
        txt_value = txt_value.unflatten(-1, (self.heads, -1))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_cos, img_sin)
            img_key = apply_rotary_emb_qwen(img_key, img_cos, img_sin)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_cos, txt_sin)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_cos, txt_sin)

        seq_img = img_query.shape[1]
        joint_query = torch.cat([img_query, txt_query], dim=1)
        joint_key = torch.cat([img_key, txt_key], dim=1)
        joint_value = torch.cat([img_value, txt_value], dim=1)

        joint_query = joint_query.transpose(1, 2)
        joint_key = joint_key.transpose(1, 2)
        joint_value = joint_value.transpose(1, 2)
        joint_hidden_states = F.scaled_dot_product_attention(
            joint_query, joint_key, joint_value, attn_mask=None, dropout_p=0.0
        )
        joint_hidden_states = joint_hidden_states.transpose(1, 2).flatten(2, 3)

        img_attn_output = joint_hidden_states[:, :seq_img, :]
        txt_attn_output = joint_hidden_states[:, seq_img:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)
        return img_attn_output, txt_attn_output

class QwenImageTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, attention_head_dim: int, eps: float = 1e-5, zero_cond_t: bool = False):
        super().__init__()
        self.zero_cond_t = zero_cond_t
        self.img_mod = nn.Sequential(nn.SiLU(), QuantizedLinear(dim, 6 * dim, bias=True))
        self.txt_mod = nn.Sequential(nn.SiLU(), QuantizedLinear(dim, 6 * dim, bias=True))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim)
        self.attn = Attention(
            dim_head=attention_head_dim,
            heads=heads,
            out_dim=dim,
            added_kv_proj_dim=dim,
            eps=eps,
        )

    def _modulate(self, x, mod_params, timestep_zero_index: Optional[int] = None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if timestep_zero_index is not None:
            actual_batch = shift.size(0) // 2
            shift_base, shift_ext = shift[:actual_batch], shift[actual_batch:]
            scale_base, scale_ext = scale[:actual_batch], scale[actual_batch:]
            gate_base, gate_ext = gate[:actual_batch], gate[actual_batch:]
            x_base = x[:, :timestep_zero_index] * (1 + scale_base.unsqueeze(1)) + shift_base.unsqueeze(1)
            x_ext = x[:, timestep_zero_index:] * (1 + scale_ext.unsqueeze(1)) + shift_ext.unsqueeze(1)
            gate = torch.cat(
                [
                    gate_base.unsqueeze(1).expand(-1, timestep_zero_index, -1),
                    gate_ext.unsqueeze(1).expand(-1, x.size(1) - timestep_zero_index, -1),
                ],
                dim=1,
            )
            return torch.cat([x_base, x_ext], dim=1), gate
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    @torch.compile(fullgraph=True, dynamic=True)
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor],
        timestep_zero_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, timestep_zero_index)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
        del img_mod1, txt_mod1

        img_attn_output, txt_attn_output = self.attn(
            img_modulated, txt_modulated, image_rotary_emb
        )
        del img_modulated, txt_modulated

        hidden_states = torch.addcmul(hidden_states, img_gate1, img_attn_output)
        encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate1, txt_attn_output)

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, timestep_zero_index)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = torch.addcmul(hidden_states, img_gate2, img_mlp_output)

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, txt_mlp_output)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.img_in = QuantizedLinear(in_channels, self.inner_dim)
        self.txt_in = QuantizedLinear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    eps=1e-5,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = QuantizedLinear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.zero_cond_t = zero_cond_t

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            if img_shapes is None:
                raise ValueError("`img_shapes` must be provided when `zero_cond_t=True`.")
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            sample = img_shapes[0]
            if isinstance(sample, (tuple, list)) and len(sample) == 3 and all(isinstance(x, (int,)) for x in sample):
                base_len = int(sample[0] * sample[1] * sample[2])
            else:
                base = sample[0]
                base_len = int(base[0] * base[1] * base[2])
            timestep_zero_index = base_len
        else:
            timestep_zero_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                timestep_zero_index=timestep_zero_index,
            )

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


def create_model(
    zero_cond_t: bool,
    dtype: Optional[torch.dtype] = None,
    num_layers: int = 60,
) -> QwenImageTransformer2DModel:
    model = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=num_layers,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=3584,
        axes_dims_rope=(16, 56, 56),
        zero_cond_t=zero_cond_t,
    )
    if dtype is not None:
        model.to(dtype)
    return model


def load_qwen_image_dit(
    dit_path: str,
    device: str,
    zero_cond_t: bool,
    dtype: torch.dtype,
    num_layers: int = 60,
) -> QwenImageTransformer2DModel:
    """Load the Qwen-Image DiT via the central quant-aware loader.

    ``init_empty_weights`` (not ``torch.device("meta")``) keeps ``pos_freqs``/
    ``neg_freqs`` as real plain tensors, since ``model.to(device)`` won't move them.
    """
    with init_empty_weights():
        model = create_model(zero_cond_t=zero_cond_t, num_layers=num_layers)
    load_dit(model, dit_path, device=device, dtype=dtype, drop_keys=("__index_timestep_zero__",))
    logger.info("Loaded Qwen-Image DiT from %s", dit_path)
    return model


__all__ = ["QwenImageTransformer2DModel", "load_qwen_image_dit", "create_model"]
