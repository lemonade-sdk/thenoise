"""Qwen-Image 2.1 DiT loading: architecture from the header, weights via ``load_dit``.

At load time the attention QK norms (``attn.norm_q`` / ``attn.norm_k``) are mapped
onto the shared ``QKNorm`` layout, and ``txt_in.text_norm`` — a *zero-centred*
RMSNorm — has the 1 added back so the plain ``RMSNorm`` runs it.
"""
from __future__ import annotations

import logging
from typing import Tuple, Union

import torch
from accelerate import init_empty_weights

from thenoise.dit.qwen_image21.models import QwenImage21Params, QwenImage21Transformer2DModel
from thenoise.utils.loader import load_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key

logger = logging.getLogger(__name__)

#: Keys that identify a Qwen-Image 2.1 DiT regardless of repackaging wrappers.
_SIGNATURE_KEYS = (
    "txt_in.text_norm.weight",
    "modulation.1.weight",
    "transformer_blocks.0.attn.norm_q.weight",
    "img_in.weight",
    "proj_out.weight",
)


def is_qwen_image21_key(keys) -> bool:
    """True if these (unwrapped) tensor names are a Qwen-Image 2.1 DiT."""
    names = {unwrap_key(k) for k in keys}
    return all(k in names for k in _SIGNATURE_KEYS)


def _header_shapes(path: str) -> dict[str, Tuple[int, ...]]:
    """Unwrapped checkpoint key -> tensor shape (header only, no tensors read)."""
    with MemoryEfficientSafeOpen(path) as f:
        return {unwrap_key(k): tuple(f.header[k]["shape"]) for k in f.keys()}


def detect_params(dit_path: str) -> QwenImage21Params:
    """Read the architecture knobs out of a Qwen-Image 2.1 checkpoint header."""
    shapes = _header_shapes(dit_path)
    missing = [k for k in _SIGNATURE_KEYS if k not in shapes]
    if missing:
        raise ValueError(f"{dit_path} is not a Qwen-Image 2.1 DiT (missing {missing})")

    img_in = shapes["img_in.weight"]
    head_dim = shapes["transformer_blocks.0.attn.norm_q.weight"][0]
    inner_dim = img_in[0]
    if inner_dim % head_dim:
        raise ValueError(f"img_in width {inner_dim} is not a multiple of head dim {head_dim}")
    num_layers = 1 + max(
        int(k.split(".")[1]) for k in shapes
        if k.startswith("transformer_blocks.") and k.split(".")[1].isdigit()
    )

    # A split MLP names its projections ``proj``/``gate_layer`` (``fused_mlp=False``).
    gate_up = shapes.get("transformer_blocks.0.img_mlp.gate_up.weight")
    if gate_up is not None:
        mlp_ratio, fused_mlp = gate_up[0] // 2 // inner_dim, True
    else:
        proj = shapes.get("transformer_blocks.0.img_mlp.proj.weight")
        if proj is None:
            raise ValueError(
                f"{dit_path} has neither img_mlp.gate_up nor img_mlp.proj weights"
            )
        mlp_ratio, fused_mlp = proj[0] // inner_dim, False

    params = QwenImage21Params(
        in_channels=img_in[1],
        out_channels=shapes["proj_out.weight"][0],
        num_layers=num_layers,
        attention_head_dim=head_dim,
        num_attention_heads=inner_dim // head_dim,
        context_in_dim=shapes["txt_in.text_norm.weight"][0],
        mlp_ratio=mlp_ratio,
        fused_mlp=fused_mlp,
    )
    logger.info(
        "Qwen-Image 2.1 DiT: %d layers, %dx%d, context %d, mlp_ratio %d (%s MLP)",
        params.num_layers, params.num_attention_heads, params.attention_head_dim,
        params.context_in_dim, params.mlp_ratio, "fused" if fused_mlp else "split",
    )
    return params


def _key_map(key: str) -> str:
    return qk_norm_key_map(key, "norm_q", "norm_k")


def _value_map(key: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Un-centre the zero-centred RMSNorm: the checkpoint stores ``scale - 1``."""
    if key == "txt_in.text_norm.weight":
        return key, tensor + 1.0
    return key, tensor


def _fuse_qkv_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Stack split checkpoint Q/K/V weights and their quantization metadata."""
    fused = dict(state)
    for q in (k for k in state if k.endswith(".attn.to_q.weight")):
        prefix = q[: -len("to_q.weight")]
        for suffix in ("weight", "weight_scale", "comfy_quant"):
            keys = [f"{prefix}to_{p}.{suffix}" for p in "qkv"]
            present = [k for k in keys if k in fused]
            if not present:
                continue
            if len(present) != 3:
                raise ValueError(f"incomplete QKV {suffix}: {prefix}")
            values = [fused.pop(k) for k in keys]
            if suffix == "comfy_quant" or (suffix == "weight_scale" and
                                            all(v.numel() == 1 for v in values)):
                if not all(torch.equal(values[0], v) for v in values[1:]):
                    raise ValueError(f"incompatible QKV {suffix}: {prefix}")
                merged = values[0]
            else:
                if suffix == "weight_scale" and any(v.numel() == 1 for v in values):
                    raise ValueError(f"mixed QKV scale granularity: {prefix}")
                if any(v.dtype != values[0].dtype for v in values[1:]):
                    raise ValueError(f"mixed QKV {suffix} dtypes: {prefix}")
                merged = torch.cat(values, dim=0)
            fused[f"{prefix}qkv.{suffix}"] = merged
    return fused


def load_qwen_image21_dit(
    dit_path: str,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> QwenImage21Transformer2DModel:
    """Build the DiT on meta from the checkpoint's own geometry, then load its weights."""
    params = detect_params(dit_path)
    logger.info("Loading Qwen-Image 2.1 DiT from %s", dit_path)
    with init_empty_weights():
        dit = QwenImage21Transformer2DModel(params)
    load_dit(
        dit,
        dit_path,
        device=device,
        dtype=dtype,
        key_map=_key_map,
        value_map=_value_map,
        state_map=_fuse_qkv_state,
    )
    # A baked INT8 LoRA must undo by reloading the original split checkpoint keys.
    restore = getattr(dit, "_quantized_restore_map", None)
    if restore is not None:
        for i in range(params.num_layers):
            prefix = f"transformer_blocks.{i}.attn."
            keys = [prefix + f"to_{p}" for p in "qkv"]
            if all(k in restore for k in keys):
                restore[prefix + "qkv"] = tuple(restore.pop(k) for k in keys)
    return dit


__all__ = [
    "detect_params",
    "is_qwen_image21_key",
    "load_qwen_image21_dit",
]
