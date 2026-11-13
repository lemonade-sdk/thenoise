"""Wan 2.2 Comfy-Org 5B / 14B transformer checkpoint loaders."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
from torch import nn
from safetensors import safe_open

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.checkpoint_weights import load_mapped_weights, validated_key_map
from thenoise.utils.loader import _build_fp8_qt
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
_T2V_14B_CONFIG = _TI2V_CONFIG | dict(
    in_channels=16, out_channels=16, num_attention_heads=40,
    num_layers=40, ffn_dim=13824,
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
        self.patch_embedding = nn.Conv3d(config.in_channels, dim, kernel_size=config.patch_size,
                                         stride=config.patch_size)
        self.condition_embedder = _WanWeightsCondition(dim, config.text_dim, config.freq_dim)
        self.blocks = nn.ModuleList([
            _WanWeightsBlock(dim, config.num_attention_heads, config.ffn_dim, config.eps)
            for _ in range(config.num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim, eps=config.eps, elementwise_affine=False)
        self.proj_out = QuantizedLinear(dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, dim))


def _fp32_key(key: str) -> bool:
    parts = key.split(".")
    return "time_embedder" in parts or "patch_embedding" in parts or bool(
        set(parts) & {"scale_shift_table", "norm1", "norm2", "norm3", "rope"}
    )



def _load_14b_weights(source, weights, dtype, check_cancel):
    """Strict, tensor-at-a-time loader for FP16 and Comfy FP8-scaled experts."""
    scale_suffixes = (".weight_scale", ".scale_weight")
    keys = set(source.keys())
    scales = {}
    for raw in keys:
        suffix = next((s for s in scale_suffixes if raw.endswith(s)), None)
        if suffix:
            stem = raw[:-len(suffix)]
            if stem in scales:
                raise ValueError(f"duplicate FP8 scale for {stem}")
            scales[stem] = raw
    input_scales = {k[:-len(".scale_input")]: k for k in keys if k.endswith(".scale_input")}
    markers = {k for k in keys if k.removeprefix("model.diffusion_model.")
               .removeprefix("diffusion_model.") == "scaled_fp8"}
    if len(markers) > 1:
        raise ValueError("Wan 14B duplicate scaled_fp8 markers")
    full_precision_mm = False
    if markers:
        marker = source.get_tensor(next(iter(markers)))
        # ComfyUI's legacy marker uses two elements for full-precision matmul.
        # Every other element count (including an empty marker) selects normal FP8.
        full_precision_mm = marker.numel() == 2
    ignored = (set(scales.values()) | set(input_scales.values()) | markers |
               {k for k in keys if k.endswith(".comfy_quant")})
    expected = {k: tuple(v.shape) for k, v in weights.state_dict().items()}
    mapped = validated_key_map(keys - ignored, wan_dit_key, set(expected))
    # Check all tensors before materializing weights (no partial silent load).
    for raw, key in mapped.items():
        entry = source.get_slice(raw)
        if tuple(entry.get_shape()) != expected[key]:
            raise ValueError(f"Wan 14B shape mismatch: {raw} -> {key}")
        kind = entry.get_dtype()
        if kind in ("F8_E4M3", "F8_E5M2"):
            if not raw.endswith(".weight") or raw[:-7] not in scales:
                raise ValueError(f"Wan 14B FP8 weight without scale: {raw}")
            if not isinstance(weights.get_submodule(key[:-7]), QuantizedLinear):
                raise ValueError(f"Wan 14B FP8 non-linear unsupported: {raw}")
        elif kind not in ("F16", "BF16", "F32"):
            raise ValueError(f"Wan 14B unsupported tensor dtype: {raw}: {kind}")
    for stem in scales:
        if stem + ".weight" not in mapped:
            raise ValueError(f"Wan 14B orphan FP8 scale: {stem}")

    # Legacy Comfy FP8 checkpoints also store calibrated input scales.
    # Their keys are metadata, but the values affect activation quantization.
    for stem, raw in input_scales.items():
        weight_raw = stem + ".weight"
        if weight_raw not in mapped or stem not in scales:
            raise ValueError(f"Wan 14B orphan FP8 input scale: {raw}")
        if source.get_slice(weight_raw).get_dtype() not in ("F8_E4M3", "F8_E5M2"):
            raise ValueError(f"Wan 14B input scale on non-FP8 weight: {raw}")
        scale = source.get_tensor(raw).float()
        if scale.numel() != 1 or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"Wan 14B invalid FP8 input scale: {raw}")
        if not full_precision_mm:
            module = weights.get_submodule(mapped[weight_raw][:-7])
            module.register_buffer("input_scale", scale.reshape(()), persistent=False)

    for i, (raw, key) in enumerate(sorted(mapped.items())):
        if check_cancel is not None and i % 8 == 0:
            check_cancel()
        tensor = source.get_tensor(raw)
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale = source.get_tensor(scales[raw[:-7]])
            weights.get_submodule(key[:-7]).load_quantized(_build_fp8_qt(tensor, scale, {}))
        else:
            weights.load_state_dict(
                {key: tensor.to(dtype=torch.float32 if _fp32_key(key) else dtype)},
                strict=False, assign=True,
            )
    if any(p.is_meta for p in weights.parameters()):
        raise RuntimeError("Wan 14B checkpoint contains unloaded meta parameters")


def load_wan_native_transformer(
    model_dir: str | Path, *, dtype: torch.dtype = torch.bfloat16,
    check_cancel: Callable[[], None] | None = None,
    _test_config: dict | None = None,
    expert: str | None = None,
) -> WanNativeDiT:
    """Read native ComfyUI Wan weights; 14B expects a matching pair of experts."""
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Wan uses FP16/BF16/FP32 execution")
    if expert not in (None, "high", "low"):
        raise ValueError("Wan expert must be 'high' or 'low'")
    config = SimpleNamespace(**((_T2V_14B_CONFIG if expert else _TI2V_CONFIG)
                                | (_test_config or {})))
    with torch.device("meta"):
        weights = _WanWeightModel(config)
    if expert:
        root = Path(model_dir)
        if root.is_file():
            if "high_noise_14B" not in root.name:
                raise ValueError("--dit must name the Wan 14B high-noise checkpoint")
            checkpoint = root if expert == "high" else root.with_name(
                root.name.replace("high_noise_14B", "low_noise_14B"))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing Wan 14B {expert}-noise checkpoint: {checkpoint}")
        else:
            names = tuple(f"wan2.2_t2v_{expert}_noise_14B_{kind}.safetensors"
                          for kind in ("fp8_scaled", "fp16"))
            checkpoint = find_weight_file(root, "diffusion_models", names)
    else:
        checkpoint = find_weight_file(
            model_dir, "diffusion_models",
            ("wan2.2_ti2v_5B_fp16.safetensors", "wan2.2_ti2v_5B_bf16.safetensors"),
        )
    if check_cancel:
        check_cancel()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
        if expert:
            _load_14b_weights(file, weights, dtype, check_cancel)
        else:
            load_mapped_weights(
                file, weights, wan_dit_key,
                dtype_for=lambda key: torch.float32 if _fp32_key(key) else dtype,
                excluded=frozenset(), check_cancel=check_cancel, label="Wan transformer",
            )
    return WanNativeDiT.from_weights(weights.eval())
