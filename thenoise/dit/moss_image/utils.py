"""MOSS-Image DiT loading: geometry out of the safetensors header.

Reading the knobs off the file lets an export of a different size load without
touching this package, and turns a wrong checkpoint into a clear message rather than
a shape error halfway through the load.

Both released checkpoints store the block projections **fused**
(``layers.<i>.proj.qkv``), which is also this module's layout; an export that keeps
``q``/``k``/``v`` apart is refused rather than half-loaded.
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights

from thenoise.dit.moss_image.models import MossImage, MossImageParams
from thenoise.utils.loader import load_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key

logger = logging.getLogger(__name__)

#: Names a MOSS-Image DiT must carry, whatever wrapper it was repackaged in.
_SIGNATURE_KEYS = (
    "video_embedder.weight",
    "text_embedder.proj.weight",
    "shared_block_modulation.weight",
    "video_head_modulation.weight",
    "layers.0.proj.ffn_up_gate.weight",
    "final_linear_video.weight",
)

#: The giveaway of a split-qkv export, which this loader refuses.
_SPLIT_QKV_KEY = "layers.0.proj.q.weight"

#: ``rope.bands`` is a derived constant the exports carry; the tables are built here.
_DROP_KEYS = ("rope.",)


def _header_shapes(path: str) -> Dict[str, Tuple[int, ...]]:
    """Unwrapped checkpoint key -> shape, without reading a single tensor."""
    with MemoryEfficientSafeOpen(path) as f:
        return {unwrap_key(k): tuple(f.header[k]["shape"]) for k in f.keys()}


def _block_count(shapes: Mapping[str, Tuple[int, ...]]) -> int:
    """How many ``layers.<i>.`` blocks the key-set spans (0 if none)."""
    indices = []
    for key in shapes:
        if not key.startswith("layers."):
            continue
        parts = key[len("layers."):].split(".")
        if parts and parts[0].isdigit():
            indices.append(int(parts[0]))
    return max(indices) + 1 if indices else 0


def detect_params(dit_path: str) -> MossImageParams:
    """Read the MOSS-Image architecture knobs out of a checkpoint header."""
    shapes = _header_shapes(dit_path)
    missing = [k for k in _SIGNATURE_KEYS if k not in shapes]
    if missing:
        raise ValueError(f"{dit_path} is not a MOSS-Image DiT (missing {missing})")
    if _SPLIT_QKV_KEY in shapes:
        raise ValueError(
            f"{dit_path} stores the block projections apart (q/k/v); this loader "
            "reads the fused layout"
        )

    dim = shapes["shared_block_modulation.weight"][0] // 6
    ffn_dim = shapes["layers.0.proj.ffn_up_gate.weight"][0] // 2
    head_dim = shapes["layers.0.q_norm.weight"][0]
    if dim % head_dim:
        raise ValueError(f"dim {dim} is not a multiple of head dim {head_dim}")
    if shapes["video_head_modulation.weight"][0] != 2 * dim:
        raise ValueError(
            f"video_head_modulation width {shapes['video_head_modulation.weight'][0]} "
            f"is not 2 x dim ({2 * dim})"
        )
    num_layers = _block_count(shapes)
    if not num_layers:
        raise ValueError(f"{dit_path} has no layers.<i> blocks")

    params = MossImageParams(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=dim // head_dim,
        num_layers=num_layers,
        text_dim=shapes["text_embedder.proj.weight"][1],
        latent_channels=shapes["video_embedder.weight"][1],
        freq_dim=shapes["time_embedding.0.weight"][1],
    )
    logger.info(
        "MOSS-Image DiT: %d joint blocks, dim %d / ffn %d, %dx%d heads, latent %d, text %d",
        params.num_layers, params.dim, params.ffn_dim,
        params.num_heads, head_dim, params.latent_channels, params.text_dim,
    )
    return params


def load_moss_image_dit(
    dit_path: str,
    params: MossImageParams,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype],
) -> MossImage:
    """Build the DiT from the checkpoint's own geometry, then load its weights.

    The QK-norm leaves are the only naming difference to bridge: the checkpoint calls
    them ``q_norm``/``k_norm``.
    """
    with init_empty_weights():
        model = MossImage(params)
    load_dit(
        model,
        dit_path,
        device=device,
        dtype=dtype,
        key_map=qk_norm_key_map,
        drop_keys=_DROP_KEYS,
    )
    logger.info("Loaded MOSS-Image DiT from %s", dit_path)
    return model


__all__ = ["detect_params", "load_moss_image_dit"]
