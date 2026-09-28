"""Checkpoint-name helpers shared by the Lumina/S3-DiT loaders.

The family stores the same modules under two generations of names:

  * **Legacy Lumina** (the Ming-Image bf16 release, and ComfyUI's
    ``all_x_embedder["2-1"]`` / ``all_final_layer["2-1"]`` dict-of-patch-config
    layout): separate ``attention.to_q/to_k/to_v``, ``attention.to_out.0``,
    ``attention.norm_q/norm_k``.
  * **Fused** (the Z-Image release, and every int8-convrot export): one
    ``attention.qkv`` (q, k, v as row blocks) and ``attention.out``, with the QK
    norms stored as ``q_norm``/``k_norm`` (int8) or ``norm_q``/``norm_k`` (legacy).

This repo's module tree is the fused layout everywhere (one GEMM, quant-friendly,
and what ``FUSE_QKV`` covers for LoRAs), so a legacy checkpoint needs a key rename
AND a fold that concatenates the three projections into one weight. The fold
cannot be a ``load_dit`` ``value_map`` (which is per-tensor, and the fused weight
is three tensors), so it is a state-dict-level function passed as
``load_dit(..., state_map=...)``.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from thenoise.utils.qk_norm import qk_norm_key_map

#: The three attention projections a legacy checkpoint stores separately, in the
#: fused matrix's row order (``lumina.models.Attention`` splits ``qkv`` back in
#: exactly this order).
QKV_PARTS: Tuple[str, ...] = ("to_q", "to_k", "to_v")

#: The legacy dict-of-patch-config names (``all_x_embedder["2-1"]``) collapse to
#: plain modules: a single-patch model has exactly one entry each, so the "which
#: patch config" part of the name carries no information.
_MODULE_RENAMES: Tuple[Tuple[str, str], ...] = (
    ("all_x_embedder.2-1.", "x_embedder."),
    ("all_final_layer.2-1.", "final_layer."),
    # ``to_out`` is stored as an nn.Sequential, the module is a plain projection.
    ("attention.to_out.0.", "attention.out."),
)


def lumina_key_map(key: str) -> str:
    """Rename a legacy Lumina checkpoint key onto this repo's fused module tree.

    Names only (values untouched) — pair it with :func:`fuse_qkv` for a checkpoint
    that still stores the three attention projections apart. The QK norms are
    mapped for BOTH legacy spellings, because the int8 export names them
    ``q_norm``/``k_norm`` and the legacy bf16 one ``norm_q``/``norm_k``.
    """
    for old, new in _MODULE_RENAMES:
        if old in key:
            key = key.replace(old, new)
    key = qk_norm_key_map(key, q_legacy="norm_q", k_legacy="norm_k")
    return qk_norm_key_map(key)


def fuse_qkv(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Stack per-projection ``to_q/to_k/to_v`` weights into one ``qkv`` weight.

    Concatenated on dim 0 in ``q, k, v`` order, i.e. the row order the fused module
    is built for. Keys that are not one of the three parts pass through untouched,
    so a checkpoint that already ships a fused ``qkv`` comes back unchanged — the
    fold is safe to apply unconditionally, and an incomplete trio is left alone for
    the loader's strict check to report rather than half-fusing silently.
    """
    grouped: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    for key, tensor in state_dict.items():
        head, part, attr = _split_projection(key)
        if part in QKV_PARTS:
            grouped.setdefault(head, {}).setdefault(part, {})[attr] = tensor

    if not any(len(parts) == len(QKV_PARTS) for parts in grouped.values()):
        return state_dict

    fused = dict(state_dict)
    for head, parts in grouped.items():
        if len(parts) != len(QKV_PARTS):
            continue
        for part in QKV_PARTS:
            for attr in parts[part]:
                fused.pop(_proj_key(head, part, attr), None)
        # The projections are bias-free in this family; kept for completeness, in
        # the same row order as the weight.
        if all("bias" in parts[p] for p in QKV_PARTS):
            fused[_proj_key(head, "qkv", "bias")] = torch.cat(
                [parts[p]["bias"] for p in QKV_PARTS], dim=0
            )
        fused[_proj_key(head, "qkv", "weight")] = torch.cat(
            [parts[p]["weight"] for p in QKV_PARTS], dim=0
        )
    return fused


def _split_projection(key: str) -> tuple[str, str, str]:
    """``(<head>, <part>, <attr>)`` for ``<head>.<part>.<attr>``, else ``("", "", "")``."""
    parts = key.split(".")
    if len(parts) < 3:
        return "", "", ""
    head, part, attr = ".".join(parts[:-2]), parts[-2], parts[-1]
    if attr not in ("weight", "bias"):
        return "", "", ""
    return head, part, attr


def _proj_key(head: str, part: str, attr: str) -> str:
    return f"{head}.{part}.{attr}" if head else f"{part}.{attr}"


def lumina_state_map(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename every key with :func:`lumina_key_map`, then :func:`fuse_qkv`."""
    return fuse_qkv({lumina_key_map(k): v for k, v in state_dict.items()})


#: A state-dict-level transform (``load_dit``'s ``state_map`` contract).
StateMap = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


__all__ = [
    "QKV_PARTS",
    "StateMap",
    "fuse_qkv",
    "lumina_key_map",
    "lumina_state_map",
]
