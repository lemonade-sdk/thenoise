"""Wan 2.2 TI2V-5B checkpoint loader for Comfy-Org single-file weights."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
from torch import nn
from safetensors import safe_open

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.checkpoint_weights import load_mapped_weights
from thenoise.utils.model_dir import find_weight_file
from thenoise.utils.wan_weights import wan_dit_key
from .model import WanNativeDiT


_TI2V_CONFIG = dict(
    in_channels=48, out_channels=48, num_attention_heads=24,
    attention_head_dim=128, num_layers=30, ffn_dim=14336,
    freq_dim=256, text_dim=4096, patch_size=(1, 2, 2),
    cross_attn_norm=True, qk_norm="rms_norm_across_heads",
    image_dim=None, added_kv_proj_dim=None,
    rope_max_seq_len=1024, eps=1e-6,
)


class _WanWeightsAttention(nn.Module):
    def __init__(self, width: int, heads: int, eps: float):
        super().__init__()
        self.heads = heads
        self.to_q = QuantizedLinear(width, width)
        self.to_k = QuantizedLinear(width, width)
        self.to_v = QuantizedLinear(width, width)
        self.to_out = nn.ModuleList([QuantizedLinear(width, width), nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(width, eps=eps)
        self.norm_k = nn.RMSNorm(width, eps=eps)
        self.add_k_proj = None
        self.add_v_proj = None


class _WanWeightsGELU(nn.Module):
    def __init__(self, width: int, ffn_dim: int):
        super().__init__()
        self.proj = QuantizedLinear(width, ffn_dim)


class _WanWeightsBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_dim: int, eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, eps=eps, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(width, eps=eps, elementwise_affine=False)
        self.attn1 = _WanWeightsAttention(width, heads, eps)
        self.attn2 = _WanWeightsAttention(width, heads, eps)
        self.ffn = nn.Module()
        self.ffn.net = nn.ModuleList([
            _WanWeightsGELU(width, ffn_dim), nn.Dropout(0.0),
            QuantizedLinear(ffn_dim, width),
        ])
        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, width))


class _WanWeightsRoPE(nn.Module):
    def __init__(self, head_dim: int, patch_size: tuple[int, ...]):
        super().__init__()
        self.patch_size = patch_size
        self.h_dim = self.w_dim = 2 * (head_dim // 6)
        self.t_dim = head_dim - self.h_dim - self.w_dim

class _WanWeightsCondition(nn.Module):
    def __init__(self, width: int, text_dim: int, freq_dim: int):
        super().__init__()
        self.timesteps_proj = SimpleNamespace(num_channels=freq_dim)
        self.time_embedder = nn.Module()
        self.time_embedder.linear_1 = QuantizedLinear(freq_dim, width)
        self.time_embedder.linear_2 = QuantizedLinear(width, width)
        self.time_proj = QuantizedLinear(width, width * 6)
        self.text_embedder = nn.Module()
        self.text_embedder.linear_1 = QuantizedLinear(text_dim, width)
        self.text_embedder.linear_2 = QuantizedLinear(width, width)
        self.image_embedder = None


class _WanWeightModel(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        dim = config.num_attention_heads * config.attention_head_dim
        self.rope = _WanWeightsRoPE(config.attention_head_dim, config.patch_size)
        self.patch_embedding = nn.Conv3d(48, dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.condition_embedder = _WanWeightsCondition(dim, config.text_dim, config.freq_dim)
        self.blocks = nn.ModuleList([
            _WanWeightsBlock(dim, config.num_attention_heads, config.ffn_dim, config.eps)
            for _ in range(config.num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim, eps=config.eps, elementwise_affine=False)
        self.proj_out = QuantizedLinear(dim, 48 * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, dim))


def _fp32_key(key: str) -> bool:
    parts = key.split(".")
    return "time_embedder" in parts or "patch_embedding" in parts or bool(
        set(parts) & {"scale_shift_table", "norm1", "norm2", "norm3", "rope"}
    )



def load_wan_native_transformer(
    model_dir: str | Path, *, dtype: torch.dtype = torch.bfloat16,
    check_cancel: Callable[[], None] | None = None,
    _test_config: dict | None = None,
) -> WanNativeDiT:
    """Read the ComfyUI WanModel layout without constructing a Diffusers module."""
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Wan TI2V-5B uses FP16/BF16/FP32 execution")
    config = SimpleNamespace(**(_TI2V_CONFIG | (_test_config or {})))
    with torch.device("meta"):
        weights = _WanWeightModel(config)
    checkpoint = find_weight_file(
        model_dir, "diffusion_models",
        ("wan2.2_ti2v_5B_fp16.safetensors", "wan2.2_ti2v_5B_bf16.safetensors"),
    )
    if check_cancel:
        check_cancel()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
        load_mapped_weights(
            file, weights, wan_dit_key,
            dtype_for=lambda key: torch.float32 if _fp32_key(key) else dtype,
            excluded=frozenset(), check_cancel=check_cancel, label="Wan transformer",
        )
    return WanNativeDiT.from_weights(weights.eval())
