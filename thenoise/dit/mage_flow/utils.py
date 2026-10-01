"""The Mage-Flow DiT architecture, read out of the safetensors header.

``MageFlowParams``' defaults are the released checkpoint's; reading the geometry from
the file is what lets a different-depth or different-width export load without touching
this package. Detection by *name* lives in :mod:`thenoise.dit.mage_flow.keys`; this is
the shape pass, run once at load.
"""
from __future__ import annotations

import logging
from typing import Tuple

from thenoise.dit.mage_flow.keys import dit_block_count
from thenoise.dit.mage_flow.models import MageFlowParams
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key

logger = logging.getLogger(__name__)

#: Tensor names a Mage-Flow DiT file must carry, whatever wrapper it was repackaged in.
_SIGNATURE_KEYS = (
    "img_in.weight",
    "txt_in.weight",
    "txt_norm.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "transformer_blocks.0.attn.norm_q.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
)


def _header_shapes(path: str) -> dict[str, Tuple[int, ...]]:
    """Unwrapped checkpoint key -> tensor shape (header only, no tensors read)."""
    with MemoryEfficientSafeOpen(path) as f:
        return {unwrap_key(k): tuple(f.header[k]["shape"]) for k in f.keys()}


def detect_params(dit_path: str) -> MageFlowParams:
    """Read the architecture knobs out of a Mage-Flow checkpoint header."""
    shapes = _header_shapes(dit_path)
    missing = [k for k in _SIGNATURE_KEYS if k not in shapes]
    if missing:
        raise ValueError(f"{dit_path} is not a Mage-Flow DiT (missing {missing})")

    img_in = shapes["img_in.weight"]
    inner_dim = img_in[0]
    head_dim = shapes["transformer_blocks.0.attn.norm_q.weight"][0]
    if inner_dim % head_dim:
        raise ValueError(f"img_in width {inner_dim} is not a multiple of head dim {head_dim}")
    num_layers = dit_block_count(shapes)
    if not num_layers:
        raise ValueError(f"{dit_path} has no transformer_blocks")

    params = MageFlowParams(
        in_channels=img_in[1],
        out_channels=shapes["proj_out.weight"][0],
        num_layers=num_layers,
        num_heads=inner_dim // head_dim,
        head_dim=head_dim,
        context_dim=shapes["txt_in.weight"][1],
    )
    logger.info(
        "Mage-Flow DiT: %d layers, %dx%d, in/out %d/%d, context %d",
        params.num_layers, params.num_heads, params.head_dim,
        params.in_channels, params.out_channels, params.context_dim,
    )
    return params


__all__ = ["detect_params"]
