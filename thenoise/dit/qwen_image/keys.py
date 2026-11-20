"""The checkpoint names the Qwen-Image family shares.

Names alone cannot tell the members apart, so each member's ``Signature`` is this layout
plus its own depth condition in :meth:`thenoise.models.base.DiffusionModel.detect`.
Mage-Flow, a member, states its own in ``thenoise.dit.mage_flow.keys``.
"""
from __future__ import annotations

from thenoise.utils.safetensors import Signature

#: Module prefixes every member of the family carries. ``add_q_proj`` is what rules out
#: the single-stream members that share the top-level projections.
FAMILY_PREFIXES: tuple[str, ...] = (
    "img_in.",
    "txt_in.",
    "time_text_embed.",
    "transformer_blocks.0.attn.add_q_proj.",
    "transformer_blocks.0.img_mlp.net.2.",
    "norm_out.linear.",
    "proj_out.",
)

#: Any DiT with the shared block layout; depth says which member.
QWEN_IMAGE_FAMILY_SIGNATURE = Signature("Qwen-Image family DiT", prefixes=FAMILY_PREFIXES)

__all__ = ["FAMILY_PREFIXES", "QWEN_IMAGE_FAMILY_SIGNATURE"]
