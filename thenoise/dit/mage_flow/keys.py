"""Checkpoint-name helpers for the Qwen-Image family: the block layout its members
share, and the depth that separates them.

Detection only ever sees names, so each detector is a predicate on the family plus
its own depth window. ``keys`` arguments are expected wrapper-prefix free.
"""
from __future__ import annotations

from typing import Iterable

#: Layer count of the released Mage-Flow DiTs.
MAGE_LAYERS = 12

#: Module prefixes every member of the family carries. ``add_q_proj`` is what rules out
#: the single-stream members that share the top-level projections.
_FAMILY_PREFIXES: tuple[str, ...] = (
    "img_in.",
    "txt_in.",
    "time_text_embed.",
    "transformer_blocks.0.attn.add_q_proj.",
    "transformer_blocks.0.img_mlp.net.2.",
    "norm_out.linear.",
    "proj_out.",
)


def is_qwen_image_family(keys: Iterable[str]) -> bool:
    """True for any DiT with the shared block layout; depth says which member."""
    names = list(keys)
    return all(
        any(key.startswith(prefix) for key in names) for prefix in _FAMILY_PREFIXES
    )


def dit_block_count(keys: Iterable[str]) -> int:
    """Number of ``transformer_blocks.<i>.`` blocks a key-set contains (0 if none)."""
    indices = []
    for key in keys:
        if not key.startswith("transformer_blocks."):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return max(indices) + 1 if indices else 0


__all__ = ["MAGE_LAYERS", "dit_block_count", "is_qwen_image_family"]
