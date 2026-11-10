"""Sharded Wan 2.2 TI2V-5B transformer checkpoint loader."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
from torch import nn
from safetensors import safe_open

from .native_dit import WanNativeDiT


# Released Wan-AI/Wan2.2-TI2V-5B-Diffusers transformer/config.json.
# Check this *before* allocating model weights; A14B and Wan2.1 are different.
_TI2V_CONFIG = {
    "in_channels": 48,
    "out_channels": 48,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "num_layers": 30,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "patch_size": (1, 2, 2),
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "image_dim": None,
    "added_kv_proj_dim": None,
}


def _config(path: Path, *, test_size: bool) -> SimpleNamespace:
    source = path / "transformer" / "config.json"
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"missing/invalid local Wan transformer config: {source}") from exc
    if config.get("_class_name") != "WanTransformer3DModel":
        raise ValueError("not a WanTransformer3DModel checkpoint")
    config["patch_size"] = tuple(config.get("patch_size", ()))
    if not test_size:
        wrong = [k for k, v in _TI2V_CONFIG.items() if config.get(k) != v]
        if wrong:
            raise ValueError(f"not Wan 2.2 TI2V-5B (config mismatches: {wrong})")
    if (config.get("in_channels") != 48 or config.get("out_channels") != 48
            or config.get("image_dim") is not None
            or config.get("added_kv_proj_dim") is not None
            or config.get("patch_size") != (1, 2, 2)
            or not config.get("cross_attn_norm")):
        raise ValueError("unsupported Wan video transformer variant")
    for key in ("num_attention_heads", "attention_head_dim", "num_layers", "ffn_dim",
                "freq_dim", "text_dim", "rope_max_seq_len"):
        value = config.get(key)
        if type(value) is not int or value <= 0 or value > 65536:
            raise ValueError(f"invalid Wan transformer config: {key}")
    # Wan splits a head into two spatial axes of 2 * floor(D/6) each;
    # temporal RoPE gets the remainder. TI2V-5B uses D=128, not /6.
    if config["attention_head_dim"] < 6 or config["attention_head_dim"] % 2:
        raise ValueError("Wan head dimension must be an even integer >= 6")
    config["eps"] = float(config.get("eps", 1e-6))
    if not 0 < config["eps"] < 0.1:
        raise ValueError("invalid Wan norm epsilon")
    return SimpleNamespace(**config)


class _WanWeightsAttention(nn.Module):
    def __init__(self, width: int, heads: int, eps: float):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.to_out = nn.ModuleList([nn.Linear(width, width), nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(width, eps=eps)
        self.norm_k = nn.RMSNorm(width, eps=eps)
        self.add_k_proj = None
        self.add_v_proj = None


class _WanWeightsGELU(nn.Module):
    def __init__(self, width: int, ffn_dim: int):
        super().__init__()
        self.proj = nn.Linear(width, ffn_dim)


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
            _WanWeightsGELU(width, ffn_dim), nn.Dropout(0.0), nn.Linear(ffn_dim, width)
        ])
        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, width))


class _WanWeightsRoPE(nn.Module):
    def __init__(self, head_dim: int, patch_size: tuple[int, ...]):
        super().__init__()
        self.patch_size = patch_size
        self.h_dim = self.w_dim = 2 * (head_dim // 6)
        self.t_dim = head_dim - self.h_dim - self.w_dim
        # Filled after meta construction. The official tables are nonpersistent.
        self.register_buffer("freqs_cos", torch.empty(0), persistent=False)
        self.register_buffer("freqs_sin", torch.empty(0), persistent=False)


def _rope_tables(rope: _WanWeightsRoPE, max_seq_len: int) -> None:
    # Match Diffusers' CPU FP64 phase generation, followed by FP32 buffers.
    # Construct outside the meta-device context (tables are small).
    cos_parts, sin_parts = [], []
    for dim in (rope.t_dim, rope.h_dim, rope.w_dim):
        frequency = 1.0 / (10000.0 ** (
            torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim
        ))
        phases = torch.outer(torch.arange(max_seq_len, dtype=torch.float64), frequency)
        cos_parts.append(phases.cos().repeat_interleave(2, dim=-1).float())
        sin_parts.append(phases.sin().repeat_interleave(2, dim=-1).float())
    rope.freqs_cos = torch.cat(cos_parts, dim=-1)
    rope.freqs_sin = torch.cat(sin_parts, dim=-1)


class _WanWeightsCondition(nn.Module):
    def __init__(self, width: int, text_dim: int, freq_dim: int):
        super().__init__()
        self.timesteps_proj = SimpleNamespace(num_channels=freq_dim)
        self.time_embedder = nn.Module()
        self.time_embedder.linear_1 = nn.Linear(freq_dim, width)
        self.time_embedder.linear_2 = nn.Linear(width, width)
        self.time_proj = nn.Linear(width, width * 6)
        self.text_embedder = nn.Module()
        self.text_embedder.linear_1 = nn.Linear(text_dim, width)
        self.text_embedder.linear_2 = nn.Linear(width, width)
        self.image_embedder = None


class _WanWeightModel(nn.Module):
    """Checkpoint-shaped modules only: no Diffusers dependency or forward."""

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
        self.proj_out = nn.Linear(dim, 48 * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, dim))


def _fp32_key(key: str) -> bool:
    # Mirror WanTransformer3DModel._keep_in_fp32_modules at load time. Casting
    # all weights with model.to(bfloat16) would lose these checkpoint values.
    return ("time_embedder" in key.split(".") or
            any(part in {"scale_shift_table", "norm1", "norm2", "norm3", "rope"}
                for part in key.split(".")))


def _weight_files(directory: Path) -> dict[str, list[str]]:
    idx = directory / "diffusion_pytorch_model.safetensors.index.json"
    single = directory / "diffusion_pytorch_model.safetensors"
    if idx.is_file():
        data = json.loads(idx.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("invalid Wan safetensors weight index")
        files: dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            if (not isinstance(key, str) or not isinstance(filename, str)
                    or filename != Path(filename).name
                    or not filename.startswith("diffusion_pytorch_model-")
                    or not filename.endswith(".safetensors")):
                raise ValueError("invalid Wan safetensors index entry")
            files.setdefault(filename, []).append(key)
        return files
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {single.name: list(f.keys())}
    raise FileNotFoundError(f"Wan transformer safetensors/index not found in {directory}")


def _load_weight_shards(
    weights: _WanWeightModel, directory: Path, dtype: torch.dtype,
    check_cancel: Callable[[], None] | None,
) -> None:
    plan = _weight_files(directory)
    expected_shapes = {name: tuple(t.shape) for name, t in weights.state_dict().items()}
    declared = {key for keys in plan.values() for key in keys}
    if declared != expected_shapes.keys():
        missing = sorted(expected_shapes.keys() - declared)
        extra = sorted(declared - expected_shapes.keys())
        raise ValueError(f"Wan checkpoint key mismatch: missing={missing[:6]}, extra={extra[:6]}")

    for filename, keys in plan.items():
        if check_cancel is not None:
            check_cancel()
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing Wan transformer shard: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            if set(shard.keys()) != set(keys):
                raise ValueError(f"Wan shard {filename} keys differ from index")
            state = {}
            for key in keys:
                if tuple(shard.get_slice(key).get_shape()) != expected_shapes[key]:
                    raise ValueError(f"Wan tensor shape mismatch: {key}")
                tensor = shard.get_tensor(key)
                if not tensor.is_floating_point():
                    raise ValueError(f"unsupported Wan tensor dtype for {key}: {tensor.dtype}")
                target = torch.float32 if _fp32_key(key) else dtype
                state[key] = tensor.to(dtype=target)
            # Assign CPU tensors directly to meta parameters, one shard at a
            # time. No 20GB float32 state_dict and no randomly initialized 5B.
            weights.load_state_dict(state, strict=False, assign=True)
            del state
    if any(param.is_meta for param in weights.parameters()):
        raise RuntimeError("Wan native transformer has unloaded meta parameters")


def load_wan_native_transformer(
    model_dir: str | Path, *, dtype: torch.dtype = torch.bfloat16,
    _test_size: bool = False,
    check_cancel: Callable[[], None] | None = None,
) -> WanNativeDiT:
    """Load the local TI2V-5B DiT from safetensors shards."""
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Wan native DiT loader supports BF16 or FP32 only")
    directory = Path(model_dir).expanduser().resolve() / "transformer"
    config = _config(directory.parent, test_size=_test_size)
    with torch.device("meta"):
        weights = _WanWeightModel(config)
    _rope_tables(weights.rope, config.rope_max_seq_len)
    _load_weight_shards(weights, directory, dtype, check_cancel)
    return WanNativeDiT.from_weights(weights.eval())
