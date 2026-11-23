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
from typing import Dict, Optional, Union

import torch

from thenoise.dit.flux3.models import Flux3, Flux3Params
from thenoise.utils.loader import build_dit
from thenoise.utils.safetensors import (
    Header,
    Signature,
    as_header,
    require_divisible,
)

logger = logging.getLogger(__name__)

#: The QK-norm scale names the head dim; exports disagree on the leaf name, which is
#: exactly what ``_flux3_key_map`` papers over.
_HEAD_DIM_KEY = "content_mode_blocks.image.0.norm.query_norm"

#: The checkpoint's QK-norm parameters are named ``scale``; the shared ``RMSNorm``
#: names them ``weight`` (same rename the Flux.2 loader applies).
_NORM_SCALE_SUFFIXES = (".norm.key_norm.scale", ".norm.query_norm.scale")

_SCALE = ".weight_scale"
_MARKER = ".comfy_quant"

FLUX3_SIGNATURE = Signature(
    "Flux.3 DiT",
    keys=(
        "content_mode_blocks.image.0.attn_out.weight",
        "single_stream_modulations.txt.lin.weight",
        "emb_in.image.weight",
        "txt_in.weight",
        "final_layer.image.linear.weight",
    ),
    #: The two block depths, which is what the block count is taken over.
    prefixes=("txt_mode_blocks.", "single_blocks."),
)


def _flux3_key_map(key: str) -> str:
    for suffix in _NORM_SCALE_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(".scale")] + ".weight"
    return key


def _head_dim(header: Header) -> int:
    """Head dim, read off the QK-norm scale under either leaf name."""
    for leaf in ("scale", "weight"):
        found = header.shapes.get(f"{_HEAD_DIM_KEY}.{leaf}")
        if found is not None:
            return found[0]
    raise ValueError(f"{header.path}: no {_HEAD_DIM_KEY}.<scale|weight> in the header")


def detect_params(checkpoint: Union[str, Header]) -> Flux3Params:
    """Read the Flux.3 architecture knobs out of a checkpoint header."""
    header = as_header(checkpoint)
    header.require(FLUX3_SIGNATURE)

    hidden = header.shape("content_mode_blocks.image.0.attn_out.weight")[0]
    head_dim = _head_dim(header)
    require_divisible(hidden, head_dim, "hidden_size", "head dim")
    emb_in = header.shape("emb_in.image.weight")
    if hidden != emb_in[0]:
        raise ValueError(
            f"{header.path}: attn_out width {hidden} does not match emb_in width {emb_in[0]}"
        )
    if head_dim % 4:
        raise ValueError(
            f"{header.path}: Flux.3 splits the head dim into 4 RoPE axes, got {head_dim}"
        )

    params = Flux3Params(
        in_channels=emb_in[1],
        context_in_dim=header.shape("txt_in.weight")[1],
        hidden_size=hidden,
        num_heads=hidden // head_dim,
        depth=header.block_count("txt_mode_blocks."),
        depth_single_blocks=header.block_count("single_blocks."),
        axes_dim=[head_dim // 4] * 4,
        flow_map="ref_time_in.in_layer.weight" in header,
        fused_qkv="single_blocks.0.qkv.weight" in header,
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
    checkpoint: Union[str, Header],
    params: Flux3Params,
    device: str | torch.device,
    dtype: Optional[torch.dtype],
) -> Flux3:
    """Build the DiT on meta from the checkpoint's own geometry, then load its weights."""
    return build_dit(
        lambda: Flux3(params),
        checkpoint,
        device=device,
        dtype=dtype,
        key_map=_flux3_key_map,
        state_map=None if params.fused_qkv else fuse_qkv_state,
        relay_stores=True,
    )


__all__ = [
    "FLUX3_SIGNATURE",
    "detect_params",
    "fuse_qkv_state",
    "load_flux3_dit",
]
