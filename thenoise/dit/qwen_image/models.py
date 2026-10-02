"""Qwen-Image DiT — dual-stream transformer (parallel image + text joint attention).

Ported from kohya-ss/musubi-tuner's ``qwen_image/qwen_image_model.py`` (itself
Diffusers ``QwenImageTransformer2DModel``), trimmed to inference-only. Weights load
via ``load_dit`` (BF16 and int8_convrot checkpoints).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from thenoise.utils.loader import load_dit
from thenoise.dit.kvcache import KVBuffers, KVCache, attend, cache_mode
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.dynamo import mark_token_axis
from thenoise.utils.positions import broadcast_positions, grid_positions
from thenoise.utils.rope import RopeCache, apply_rope, matrix_rope
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.setup_logging import setup_logging
from thenoise.utils.timestep import timestep_embedding

setup_logging()
import logging

logger = logging.getLogger(__name__)


def build_video_positions(img_shapes, device):
    """Build ``[1, total_tokens, 3]`` (t,h,w) positions for the image+ref stream.

    ``img_shapes`` are ``(frame, height, width)``. The h/w axes use the centered
    convention ``pos = r - ceil(size / 2)``; shape ``i`` spans t in ``[i, i + frame)``.
    """
    parts = []
    for i, (frame, height, width) in enumerate(img_shapes):
        parts.append(
            grid_positions(
                [frame, height, width],
                start=[i, 0, 0],
                centered=[False, True, True],
                dtype=torch.float32,
                device=device,
            )
        )
    return torch.cat(parts, dim=0).unsqueeze(0)


def build_txt_positions(max_vid_index, txt_len, device):
    """Build ``[1, txt_len, 3]`` positions for the text stream.

    Text advances a single index ``max_vid_index + j`` across all three axes.
    """
    return broadcast_positions(
        txt_len, 3, offset=max_vid_index, dtype=torch.float32, device=device
    ).unsqueeze(0)


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


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps = timestep.to(hidden_states.dtype)
        return self.timestep_embedder(timestep_embedding(timesteps, 256))


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
        # Dropout stays (p=0) so ``net.1``/``net.2`` line up with the checkpoint keys.
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
        img_pe: torch.Tensor,
        txt_pe: torch.Tensor,
        bufs: Optional[KVBuffers] = None,
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

        img_query = img_query.transpose(1, 2)
        img_key = img_key.transpose(1, 2)
        img_value = img_value.transpose(1, 2)
        txt_query = txt_query.transpose(1, 2)
        txt_key = txt_key.transpose(1, 2)
        txt_value = txt_value.transpose(1, 2)

        img_query, img_key = apply_rope(img_query, img_key, img_pe)
        txt_query, txt_key = apply_rope(txt_query, txt_key, txt_pe)

        # Joint sequence order is text first, then image (its trailing slice being
        # the reference tokens): every step rewrites the leading text + target prefix
        # of its K/V buffers while the references stay frozen in the tail.
        txt_len = txt_query.shape[2]
        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key = torch.cat([txt_key, img_key], dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)

        joint_hidden_states = attend(joint_query, joint_key, joint_value, bufs)  # [B, L, H*D]

        img_attn_output = joint_hidden_states[:, txt_len:, :]
        txt_attn_output = joint_hidden_states[:, :txt_len, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)
        return img_attn_output, txt_attn_output

class QwenImageTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, attention_head_dim: int, eps: float = 1e-5):
        super().__init__()
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

    def _modulate(self, x, mod_params, token_mask: Optional[torch.Tensor] = None):
        """AdaLN modulation of ``x``; ``token_mask`` picks the per-token modulation row.

        ``mod_params`` carries the ``[t; 0]`` conditioning rows and ``token_mask`` is
        the ``[1, L, 1]`` leading-prefix mask of the target tokens: ``True`` takes the
        t row, the trailing reference tokens the t=0 row. A mask rather than an
        ``expand`` + ``cat`` of the two row groups: a concatenation whose split point
        is the (dynamic) token count is what Inductor cannot tile in a compiled block.
        """
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if token_mask is None:
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)
        half = shift.size(0) // 2
        shift = torch.where(token_mask, shift[:half].unsqueeze(1), shift[half:].unsqueeze(1))
        scale = torch.where(token_mask, scale[:half].unsqueeze(1), scale[half:].unsqueeze(1))
        gate = torch.where(token_mask, gate[:half].unsqueeze(1), gate[half:].unsqueeze(1))
        return x * (1 + scale) + shift, gate

    @torch.compile(fullgraph=False)
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        img_pe: torch.Tensor,
        txt_pe: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        bufs: Optional[KVBuffers] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        if token_mask is not None:
            # ``temb`` carries the t and t=0 rows; the text stream uses the t row.
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, token_mask)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
        del img_mod1, txt_mod1

        img_attn_output, txt_attn_output = self.attn(
            img_modulated, txt_modulated, img_pe, txt_pe, bufs
        )
        del img_modulated, txt_modulated

        hidden_states = torch.addcmul(hidden_states, img_gate1, img_attn_output)
        encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate1, txt_attn_output)

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, token_mask)
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
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.patch_size = patch_size

        self.pe_embedder = RopeCache(matrix_rope(list(axes_dims_rope), 10000))
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
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = QuantizedLinear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor = None,
        img_pe: torch.Tensor = None,
        txt_pe: torch.Tensor = None,
        ref_tokens: Optional[torch.Tensor] = None,
        ref_pe: Optional[torch.Tensor] = None,
        kv: Optional[KVCache] = None,
        timestep_zero_index: Optional[int] = None,
    ) -> torch.Tensor:
        """One DiT forward, returning the velocity of the *target* tokens only.

        ``ref_tokens`` are the reference (edit) latents, packed into image tokens and
        appended after the target ones, with their positions inside ``ref_pe``.
        ``timestep_zero_index`` is the target-token count at which the image stream
        switches from the t row to the t=0 row; with it, ``temb`` carries both rows.
        """
        num_img_tokens = hidden_states.shape[1]
        ref_len = 0 if ref_tokens is None else ref_tokens.shape[1]
        mode = cache_mode(kv, ref_tokens is not None)

        if ref_tokens is not None:
            hidden_states = torch.cat([hidden_states, ref_tokens], dim=1)
            if ref_pe is not None:
                img_pe = torch.cat([img_pe, ref_pe], dim=1)

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        # Timestep-zero conditioning only applies while reference tokens are in the
        # sequence.
        zero_cond_t = timestep_zero_index is not None and ref_len > 0
        if zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)

        # Per-token conditioning row, as a mask over the image stream so the blocks
        # never have to concatenate at a dynamic token count (see ``_modulate``).
        token_mask = None
        if zero_cond_t:
            token_mask = (
                torch.arange(hidden_states.shape[1], device=hidden_states.device)
                < timestep_zero_index
            )[None, :, None]
            mark_token_axis(token_mask)

        # A block attends over text + target + references while filling; the
        # references are simply absent from the sequence once cached.
        kv_len = encoder_hidden_states.shape[1] + hidden_states.shape[1]
        kv_shape = (hidden_states.shape[0], self.num_heads, kv_len, self.head_dim)

        for i, block in enumerate(self.transformer_blocks):
            bufs = (None if kv is None else
                    kv.buffers(i, mode, kv_shape, hidden_states.dtype, hidden_states.device))
            mark_token_axis(hidden_states, encoder_hidden_states, img_pe, txt_pe)
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                img_pe=img_pe,
                txt_pe=txt_pe,
                token_mask=token_mask,
                bufs=bufs,
            )

        if mode == "fill" and kv is not None:
            kv.set_filled()

        if zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        if ref_len:
            # The reference tokens carry no prediction: drop them before the head.
            hidden_states = hidden_states[:, :num_img_tokens]

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


def create_model(
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
    )
    if dtype is not None:
        model.to(dtype)
    return model


def load_qwen_image_dit(
    dit_path: str,
    device: str,
    dtype: torch.dtype,
    num_layers: int = 60,
) -> QwenImageTransformer2DModel:
    """Load the Qwen-Image DiT via the central quant-aware loader."""
    with init_empty_weights():
        model = create_model(num_layers=num_layers)
    load_dit(model, dit_path, device=device, dtype=dtype)
    logger.info("Loaded Qwen-Image DiT from %s", dit_path)
    return model


__all__ = ["QwenImageTransformer2DModel", "load_qwen_image_dit", "create_model", "build_video_positions", "build_txt_positions"]
