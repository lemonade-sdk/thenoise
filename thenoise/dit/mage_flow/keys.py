"""Checkpoint names for Mage-Flow: the 12-layer member of the Qwen-Image family.

The layout it shares with the family is stated once in
``thenoise.dit.qwen_image.keys``; the depth is what separates the members, so this file
adds it to the shared layout rather than repeating the names.
"""
from __future__ import annotations

from thenoise.dit.qwen_image.keys import FAMILY_PREFIXES, QWEN_IMAGE_FAMILY_SIGNATURE
from thenoise.utils.safetensors import Signature

#: Layer count of the released Mage-Flow DiTs.
MAGE_LAYERS = 12

#: The 12-layer member. The keys are the ones its geometry is read from, so the loader
#: cannot be handed a file the catalog claimed and it cannot build.
MAGE_FLOW_SIGNATURE = Signature(
    "Mage-Flow DiT",
    keys=(
        "img_in.weight",
        "txt_in.weight",
        "txt_norm.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
    ),
    prefixes=FAMILY_PREFIXES,
)

__all__ = ["MAGE_FLOW_SIGNATURE", "MAGE_LAYERS", "QWEN_IMAGE_FAMILY_SIGNATURE"]  # re-exported: the family layout is imported from here by both members
