"""The H3-Image geometry and loader: the architecture, the PDD head bank and its
sigma grid, all read out of the checkpoint.

The head count is a shape and the sigma grid is only in the file's ``config`` metadata,
and the two have to agree; everything else the model needs to be built is a shape too.
"""
from __future__ import annotations

import logging
import math
from typing import Union

import torch

from thenoise.dit.h3_image.layout import PATCH_SIZE
from thenoise.dit.h3_image.models import H3ImageDiT, H3ImageParams
from thenoise.utils.loader import build_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import (
    Header,
    Signature,
    as_header,
    require_divisible,
)

logger = logging.getLogger(__name__)

H3_IMAGE_SIGNATURE = Signature(
    "H3-Image DiT",
    keys=(
        "video_patch_proj.weight",
        "condition_proj.weight",
        "final_layer.video_out.weight",
        "adaln_t_table",
        "rope.inv_freq",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.q_norm.weight",
        "blocks.0.mlp.fc2.weight",
    ),
    #: The packed sibling this adapter cannot run shares every other name and carries an
    #: audio stream no part of this model reads.
    absent=("audio_patch_proj.weight",),
)


def detect_params(checkpoint: Union[str, Header]) -> H3ImageParams:
    """Read the architecture, the head-bank rows and its sigma grid out of a file."""
    header = as_header(checkpoint)
    header.require(H3_IMAGE_SIGNATURE)

    video_out = header.shape("final_layer.video_out.weight")
    patch_proj = header.shape("video_patch_proj.weight")
    adaln_t_table = header.shape("adaln_t_table")
    qkv = header.shape("blocks.0.attn.qkv_proj.weight")
    head_dim = header.shape("blocks.0.attn.q_norm.weight")[0]

    require_divisible(video_out[0], patch_proj[1], "head bank width", "one head's width")
    pdd_steps = video_out[0] // patch_proj[1]
    pdd_sigmas = tuple(
        header.metadata_json("config").get("transformer", {}).get("pdd_sigmas", (1.0, 0.0)))
    if len(pdd_sigmas) != pdd_steps + 1:
        raise ValueError(
            f"{header.path}: the head bank is {pdd_steps} rows but the distilled sigma "
            f"grid has {len(pdd_sigmas)} points ({pdd_sigmas}); expected grid length "
            f"{pdd_steps + 1}"
        )

    num_heads = qkv[0] // (3 * head_dim)
    params = H3ImageParams(
        hidden_size=patch_proj[0],
        num_layers=header.block_count("blocks."),
        token_refiner_layers=header.block_count("token_refiner.blocks."),
        num_heads=num_heads,
        head_dim=head_dim,
        ffn_hidden_size=header.shape("blocks.0.mlp.fc2.weight")[1],
        latents_dim=patch_proj[1] // math.prod(PATCH_SIZE),
        text_dim=header.shape("condition_proj.weight")[1],
        rope_inv_freq=header.shape("rope.inv_freq")[0],
        adaln_grid=adaln_t_table[0],
        adaln_dim=adaln_t_table[1],
        pdd_steps=pdd_steps,
        pdd_sigmas=pdd_sigmas,
    )
    logger.info(
        "H3-Image DiT: %d layers, %dx%d heads, %d-step PDD head bank",
        params.num_layers, params.num_heads, params.head_dim, params.pdd_steps,
    )
    return params


def load_h3_image_dit(
    checkpoint: Union[str, Header],
    params: H3ImageParams,
    device,
    dtype: torch.dtype,
) -> H3ImageDiT:
    """Build the DiT on meta from the checkpoint's own geometry, then load its weights.

    The file is read with its own dtypes (``dtype=None``) and only then moved to
    ``dtype``: that move is where the FP32 islands pin themselves and quantized weights
    stay quantized, where casting the state dict would round the modulation factors.
    ``attn.q_norm`` / ``attn.k_norm`` are remapped by ``qk_norm_key_map``.
    """
    model = build_dit(
        lambda: H3ImageDiT(params, compute_dtype=dtype),
        checkpoint,
        device=device,
        dtype=None,
        key_map=qk_norm_key_map,
    )
    model.to(dtype=dtype)
    return model


__all__ = ["H3_IMAGE_SIGNATURE", "detect_params", "load_h3_image_dit"]
