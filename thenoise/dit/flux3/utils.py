"""Flux.3 DiT loading: geometry out of the header, and the two key folds.

The architecture is read from the safetensors header (width, head count, the two
block depths) plus two exports the released checkpoints disagree on:

  * ``flow_map`` — ``ref_time_in`` is present exactly on the step-distilled flow maps;
  * ``fused_qkv`` — some exports store ``qkv`` fused, others the three projections
    apart. This module always builds the fused one (one GEMM, and the LoRA path
    already fuses ``q_proj``/``k_proj``/``v_proj`` onto it), so the split form is
    folded at load time.
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Tuple

import torch
from accelerate import init_empty_weights

from thenoise.dit.flux3.models import Flux3, Flux3Params
from thenoise.utils.loader import load_dit
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key

logger = logging.getLogger(__name__)

#: Tensor names a Flux.3 DiT file must carry, whatever wrapper it was repackaged in.
_SIGNATURE_KEYS = (
    "content_mode_blocks.image.0.attn_out.weight",
    "single_stream_modulations.txt.lin.weight",
    "emb_in.image.weight",
    "txt_in.weight",
    "final_layer.image.linear.weight",
)

# The QK-norm scale names the head dim; exports disagree on the leaf name, which is
# exactly what ``_flux3_key_map`` papers over.
_HEAD_DIM_KEY = "content_mode_blocks.image.0.norm.query_norm"

#: The checkpoint's QK-norm parameters are named ``scale``; the shared ``RMSNorm``
#: names them ``weight`` (same rename the Flux.2 loader applies).
_NORM_SCALE_SUFFIXES = (".norm.key_norm.scale", ".norm.query_norm.scale")

_SCALE = ".weight_scale"
_MARKER = ".comfy_quant"


def _flux3_key_map(key: str) -> str:
    for suffix in _NORM_SCALE_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(".scale")] + ".weight"
    return key


def _header(path: str) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, str]]:
    """Unwrapped checkpoint key -> (shape, header dtype), reading the header only."""
    with MemoryEfficientSafeOpen(path) as f:
        shapes = {unwrap_key(k): tuple(f.header[k]["shape"]) for k in f.keys()}
        dtypes = {unwrap_key(k): f.header[k]["dtype"] for k in f.keys()}
    return shapes, dtypes


def _block_count(shapes: Mapping[str, Tuple[int, ...]], prefix: str) -> int:
    """Highest ``<prefix><i>.`` block index the key-set contains, plus one (0 if none)."""
    indices = []
    for key in shapes:
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].split(".")
        if parts and parts[0].isdigit():
            indices.append(int(parts[0]))
    return max(indices) + 1 if indices else 0


def _head_dim(shapes: Mapping[str, Tuple[int, ...]]) -> int:
    """Head dim, read off the QK-norm scale under either leaf name."""
    for leaf in ("scale", "weight"):
        found = shapes.get(f"{_HEAD_DIM_KEY}.{leaf}")
        if found is not None:
            return found[0]
    raise ValueError(f"no {_HEAD_DIM_KEY}.<scale|weight> in the checkpoint header")


def detect_params(dit_path: str) -> Flux3Params:
    """Read the Flux.3 architecture knobs out of a checkpoint header."""
    shapes, _ = _header(dit_path)
    missing = [k for k in _SIGNATURE_KEYS if k not in shapes]
    if missing:
        raise ValueError(f"{dit_path} is not a Flux.3 DiT (missing {missing})")

    hidden = shapes["content_mode_blocks.image.0.attn_out.weight"][0]
    head_dim = _head_dim(shapes)
    if hidden % head_dim:
        raise ValueError(f"hidden_size {hidden} is not a multiple of head dim {head_dim}")
    if hidden != shapes["emb_in.image.weight"][0]:
        raise ValueError(
            f"attn_out width {hidden} does not match emb_in width "
            f"{shapes['emb_in.image.weight'][0]}"
        )
    if head_dim % 4:
        raise ValueError(f"Flux.3 splits the head dim into 4 RoPE axes, got {head_dim}")

    depth = _block_count(shapes, "txt_mode_blocks.")
    depth_single = _block_count(shapes, "single_blocks.")
    if not depth or not depth_single:
        raise ValueError(f"{dit_path} has no txt_mode_blocks / single_blocks")

    params = Flux3Params(
        in_channels=shapes["emb_in.image.weight"][1],
        context_in_dim=shapes["txt_in.weight"][1],
        hidden_size=hidden,
        num_heads=hidden // head_dim,
        depth=depth,
        depth_single_blocks=depth_single,
        axes_dim=[head_dim // 4] * 4,
        flow_map="ref_time_in.in_layer.weight" in shapes,
        fused_qkv="single_blocks.0.qkv.weight" in shapes,
    )
    logger.info(
        "Flux.3 DiT: %d early + %d joint blocks, %dx%d, in %d, context %d%s%s",
        params.depth, params.depth_single_blocks, params.num_heads, head_dim,
        params.in_channels, params.context_in_dim,
        ", flow map" if params.flow_map else "",
        "" if params.fused_qkv else ", split q/k/v",
    )
    return params


# ------------------------------------------------------------------ q/k/v -> qkv fold


def fuse_qkv_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Stack a split ``q_proj``/``k_proj``/``v_proj`` export into this module's ``qkv``.

    The fused matrix is the three stacked in that row order, so their per-row
    ``.weight_scale`` siblings stack the same way; a per-tensor scale is shared by the
    whole matrix and is kept once. The ``.comfy_quant`` profile moves with the weight.
    """
    prefixes = sorted(k[: -len(".q_proj.weight")] for k in state if k.endswith(".q_proj.weight"))
    if not prefixes:
        return state

    fused = dict(state)
    for prefix in prefixes:
        parts = [f"{prefix}.{p}" for p in ("q_proj", "k_proj", "v_proj")]
        for suffix in (".weight", _SCALE, _MARKER):
            keys = [p + suffix for p in parts]
            present = [k for k in keys if k in fused]
            if not present:
                continue
            out = f"{prefix}.qkv{suffix}"
            if suffix == _MARKER:
                # One profile moves with the weight; agreeing parts make that legal.
                if len({bytes(fused[k].tolist()) for k in present}) > 1:
                    raise ValueError(
                        f"cannot fuse q/k/v stored with different quantization "
                        f"profiles: {prefix}"
                    )
                fused[out] = fused[present[0]]
            elif len(present) != len(keys):
                raise ValueError(f"{prefix} has a partial {suffix[1:]} set: {keys}")
            elif suffix == ".weight":
                fused[out] = torch.cat([fused[k] for k in keys], dim=0)
            else:
                fused[out] = _stack_scales([fused[k] for k in keys])
            for key in keys:
                fused.pop(key, None)
    return fused


def _stack_scales(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-row scales in the fused row order, or keep one shared per-tensor scale."""
    if all(t.numel() == 1 for t in tensors):
        return tensors[0]
    if any(t.numel() == 1 for t in tensors):
        raise ValueError(
            "cannot fuse q/k/v with mixed per-tensor and per-row weight scales: "
            f"{[tuple(t.shape) for t in tensors]}"
        )
    return torch.cat(tensors, dim=0)


def load_flux3_dit(
    dit_path: str,
    params: Flux3Params,
    device: str | torch.device,
    dtype: Optional[torch.dtype],
) -> Flux3:
    """Build the DiT on meta from the checkpoint's own geometry, then load its weights."""
    with init_empty_weights():
        model = Flux3(params)
    return load_dit(
        model,
        dit_path,
        device=device,
        dtype=dtype,
        key_map=_flux3_key_map,
        state_map=None if params.fused_qkv else fuse_qkv_state,
    )


__all__ = [
    "detect_params",
    "fuse_qkv_state",
    "load_flux3_dit",
]
