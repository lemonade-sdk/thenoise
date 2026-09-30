"""Checkpoint-name helpers shared by the Lumina/S3-DiT loaders and detectors.

The family stores the same modules under two generations of names: legacy Lumina
(separate ``to_q/to_k/to_v``, ``to_out.0``, ``norm_q/norm_k``) and the fused layout
(``qkv``, ``out``, ``q_norm``/``k_norm``). This repo's tree is fused everywhere, so a
legacy checkpoint needs a rename AND a fold that concatenates the three projections."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import torch

from thenoise.utils.qk_norm import qk_norm_key_map

#: The three attention projections a legacy checkpoint stores separately, in the
#: fused matrix's row order (``Attention`` splits ``qkv`` back in this order).
QKV_PARTS: Tuple[str, ...] = ("to_q", "to_k", "to_v")

#: Legacy dict-of-patch-config names (``all_x_embedder["2-1"]``) collapse to plain
#: modules: a single-patch model has exactly one entry each.
_MODULE_RENAMES: Tuple[Tuple[str, str], ...] = (
    ("all_x_embedder.2-1.", "x_embedder."),
    ("all_final_layer.2-1.", "final_layer."),
    # ``to_out`` is stored as an nn.Sequential, the module is a plain projection.
    ("attention.to_out.0.", "attention.out."),
)


def is_s3dit(keys: Iterable[str]) -> bool:
    """True for any Lumina/S3-DiT file: caption embedder + context refiner + patch
    embedder, under either naming generation. ``keys`` must be wrapper-prefix free.
    """
    keys = list(keys)
    has_cap = any(key.startswith("cap_embedder.") for key in keys)
    has_context = any(key.startswith("context_refiner.") for key in keys)
    has_patch_embed = any(
        key.startswith("x_embedder.") or key.startswith("all_x_embedder.") for key in keys
    )
    return has_cap and has_context and has_patch_embed


def has_learned_pad_tokens(keys: Iterable[str]) -> bool:
    """True when the file ships Z-Image's learned alignment-pad tokens — the only
    separator of the two family members, since Ming-Image zero-fills and masks pads.
    """
    return any(
        key.startswith("x_pad_token") or key.startswith("cap_pad_token") for key in keys
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
    "has_learned_pad_tokens",
    "is_s3dit",
    "lumina_key_map",
    "lumina_state_map",
]
