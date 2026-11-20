"""MOSS-Image DiT loading: geometry out of the safetensors header.

Reading the knobs off the file lets an export of a different size load without
touching this package, and turns a wrong checkpoint into a clear message rather than
a shape error halfway through the load. :data:`MOSS_SIGNATURE` is what the catalog and
this read agree makes a file MOSS-Image.

Both released checkpoints store the block projections **fused**
(``layers.<i>.proj.qkv``), which is also this module's layout; an export that keeps
``q``/``k``/``v`` apart is refused rather than half-loaded.
"""
from __future__ import annotations

import logging
from typing import Union

import torch

from thenoise.dit.moss_image.models import MossImage, MossImageParams
from thenoise.utils.loader import build_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import (
    Header,
    Signature,
    as_header,
    require_divisible,
)

logger = logging.getLogger(__name__)

#: ``rope.bands`` is a derived constant the exports carry; the tables are built here.
_DROP_KEYS = ("rope.",)

#: The giveaway of a split-qkv export, which this loader refuses.
_SPLIT_QKV_KEY = "layers.0.proj.q.weight"

MOSS_SIGNATURE = Signature(
    "MOSS-Image DiT",
    keys=(
        "video_embedder.weight",
        "text_embedder.proj.weight",
        "shared_block_modulation.weight",
        "video_head_modulation.weight",
        "layers.0.proj.ffn_up_gate.weight",
        "layers.0.q_norm.weight",
        "time_embedding.0.weight",
        "final_linear_video.weight",
    ),
)


def detect_params(checkpoint: Union[str, Header]) -> MossImageParams:
    """Read the MOSS-Image architecture knobs out of a checkpoint header."""
    header = as_header(checkpoint)
    header.require(MOSS_SIGNATURE)
    if _SPLIT_QKV_KEY in header:
        raise ValueError(
            f"{header.path} stores the block projections apart (q/k/v); this loader "
            "reads the fused layout"
        )

    dim = header.shape("shared_block_modulation.weight")[0] // 6
    head_dim = header.shape("layers.0.q_norm.weight")[0]
    require_divisible(dim, head_dim, "dim", "head dim")
    head = header.shape("video_head_modulation.weight")[0]
    if head != 2 * dim:
        raise ValueError(
            f"{header.path}: video_head_modulation width {head} is not 2 x dim ({2 * dim})"
        )
    num_layers = header.block_count("layers.")

    params = MossImageParams(
        dim=dim,
        ffn_dim=header.shape("layers.0.proj.ffn_up_gate.weight")[0] // 2,
        num_heads=dim // head_dim,
        num_layers=num_layers,
        text_dim=header.shape("text_embedder.proj.weight")[1],
        latent_channels=header.shape("video_embedder.weight")[1],
        freq_dim=header.shape("time_embedding.0.weight")[1],
    )
    logger.info(
        "MOSS-Image DiT: %d joint blocks, dim %d / ffn %d, %dx%d heads, latent %d, text %d",
        params.num_layers, params.dim, params.ffn_dim,
        params.num_heads, head_dim, params.latent_channels, params.text_dim,
    )
    return params


def load_moss_image_dit(
    checkpoint: Union[str, Header],
    params: MossImageParams,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> MossImage:
    """Build the DiT from the checkpoint's own geometry, then load its weights.

    The QK-norm leaves are the only naming difference to bridge: the checkpoint calls
    them ``q_norm``/``k_norm``.
    """
    return build_dit(
        lambda: MossImage(params),
        checkpoint,
        device=device,
        dtype=dtype,
        key_map=qk_norm_key_map,
        drop_keys=_DROP_KEYS,
    )


__all__ = ["MOSS_SIGNATURE", "detect_params", "load_moss_image_dit"]
