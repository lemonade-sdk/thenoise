"""Checkpoint-name helpers for the Qwen-Image *family* — the block layout Qwen-Image
and Mage-Flow share, and the one thing that tells its members apart.

The two DiTs are the same dual-stream block: identical tensor names from ``img_in``
down to ``proj_out``, right down to the ``add_*_proj`` text stream and the
``img_mlp.net.2`` output. Detection matches on names only (that is all a
``safe_open`` handle exposes and all the catalog's ``resolve()`` iterates), so the
separator has to be the *depth*: Qwen-Image ships 60 blocks, Mage-Flow 12. Each
detector is therefore a positive predicate on the family plus its own depth window,
with no cross-model coupling and no dependence on catalog order.

``keys`` arguments are expected wrapper-prefix free (see
``thenoise.models.base.normalize_keys``), like the rest of the family helpers.
"""
from __future__ import annotations

from typing import Iterable

#: Layer count of the released Mage-Flow DiTs. The separator of the family: a
#: Qwen-Image checkpoint has five times as many blocks and never this few.
MAGE_LAYERS = 12

#: Module prefixes every member of the family carries. The three top-level
#: projections + the time/text embedding, one dual-stream block (its ``add_q_proj``
#: is what rules out the single-stream Qwen-Image 2.1, which shares the same three
#: top-level prefixes) and the output head.
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
    """True for any Qwen-Image-family DiT (Qwen-Image or Mage-Flow).

    Depth says *which* member; this only says the block layout is the shared one.
    """
    names = list(keys)
    return all(
        any(key.startswith(prefix) for key in names) for prefix in _FAMILY_PREFIXES
    )


def dit_block_count(keys: Iterable[str]) -> int:
    """Number of ``transformer_blocks.<i>.`` blocks a key-set contains (0 if none).

    The header lists every tensor name, so the block count is the highest index
    present plus one — no shapes, no tensors, and nothing to reconcile with a
    checkpoint that ships a subset of its layers under another prefix.
    """
    indices = []
    for key in keys:
        if not key.startswith("transformer_blocks."):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return max(indices) + 1 if indices else 0


__all__ = ["MAGE_LAYERS", "dit_block_count", "is_qwen_image_family"]
