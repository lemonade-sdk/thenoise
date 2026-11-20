"""The Mage-Flow DiT architecture, read out of the safetensors header.

Reading the geometry from the file lets a different-depth or different-width export
load without touching this package, and tells the dense AdaLN heads from the low-rank
modulation variant. Which file is Mage-Flow at all is
:data:`thenoise.dit.mage_flow.keys.MAGE_FLOW_SIGNATURE`, and the catalog and this read agree on it.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

from thenoise.dit.mage_flow.keys import MAGE_FLOW_SIGNATURE
from thenoise.dit.mage_flow.models import MageFlowParams
from thenoise.utils.safetensors import Header, as_header, require_divisible

logger = logging.getLogger(__name__)


def detect_params(checkpoint: Union[str, Header]) -> MageFlowParams:
    """Read the architecture knobs out of a Mage-Flow checkpoint header."""
    header = as_header(checkpoint)
    header.require(MAGE_FLOW_SIGNATURE)

    inner_dim = header.shape("img_in.weight")[0]
    head_dim = header.shape("transformer_blocks.0.attn.norm_q.weight")[0]
    require_divisible(inner_dim, head_dim, "img_in width", "head dim")
    num_layers = header.block_count("transformer_blocks.")

    params = MageFlowParams(
        in_channels=header.shape("img_in.weight")[1],
        out_channels=header.shape("proj_out.weight")[0],
        num_layers=num_layers,
        num_heads=inner_dim // head_dim,
        head_dim=head_dim,
        context_dim=header.shape("txt_in.weight")[1],
        modulation_rank=_modulation_rank(header),
    )
    logger.info(
        "Mage-Flow DiT: %d layers, %dx%d, in/out %d/%d, context %d%s",
        params.num_layers, params.num_heads, params.head_dim,
        params.in_channels, params.out_channels, params.context_dim,
        "" if params.modulation_rank is None else f", modulation rank {params.modulation_rank}",
    )
    return params


def _modulation_rank(header: Header) -> Optional[int]:
    """The low-rank export's shared modulation width, or ``None`` for the dense heads.

    The variant is the presence of ``modulation_down``; its width must agree with the
    per-block heads it feeds.
    """
    down = header.shapes.get("modulation_down.weight")
    if down is None:
        return None
    rank = down[0]
    heads = header.shape("transformer_blocks.0.img_mod.1.weight")
    if heads[1] != rank:
        raise ValueError(
            f"{header.path}: low-rank modulation width {rank} does not match the "
            f"modulation heads (input {heads[1]})"
        )
    return rank


__all__ = ["detect_params"]
