"""Which H3 a checkpoint is, and the architecture it states.

Both variants are the same trunk under the same names, so one geometry read answers both
and takes the ``Signature`` the calling adapter declares: a file cannot be measured by a
loader that would then refuse to build it.

One difference is not expressible as a name test. The unpruned checkpoints embed the
timestep with a sinusoidal MLP and modulate with matrices as wide as that embedding,
where this trunk reads a curve stored in the file and projects onto it — about 26 GB of
the bf16 file, and the reason only the pruned release loads here. That is said out loud,
because "missing 'adaln_t_table'" would not tell anyone which file to download.
"""
from __future__ import annotations

import logging
import math
from typing import Union

import torch

from thenoise.dit.h3.image import H3ImageDiT
from thenoise.dit.h3.layout import PATCH_SIZE
from thenoise.dit.h3.trunk import H3Params, RELAY_STORES
from thenoise.utils.loader import build_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import (
    Header,
    Signature,
    as_header,
    require_divisible,
)

logger = logging.getLogger(__name__)

#: The trunk, the modulation curve and the video head: names every H3 checkpoint carries.
_CORE_KEYS = (
    "video_patch_proj.weight",
    "condition_proj.weight",
    "final_layer.video_out.weight",
    "adaln_t_table",
    "rope.inv_freq",
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.attn.q_norm.weight",
    "blocks.0.mlp.fc2.weight",
)

#: The audio stream's two names: the projection in, and the head out. A file carrying
#: either is the audiovisual model, which the image loader must refuse rather than
#: half-load — the trunk would run without ever touching the second stream.
_AUDIO_KEYS = ("audio_patch_proj.weight", "final_layer.audio_out.weight")

H3_IMAGE_SIGNATURE = Signature(
    "H3-Image DiT",
    keys=_CORE_KEYS,
    absent=_AUDIO_KEYS,
)

H3_AV_SIGNATURE = Signature("H3 video DiT", keys=(*_CORE_KEYS, *_AUDIO_KEYS))

#: What an unpruned checkpoint embeds its timestep with, instead of the curve.
_UNPRUNED = "time_embedder.proj_in.weight"

#: The layers the model builds in FP32 whatever the run's dtype, because rounding them is
#: a quality regression rather than a rounding error. A file that stores one of them
#: quantized cannot fill it: the packed layouts live in ``QuantizedLinear``, and these are
#: ``Fp32Linear`` and a curve buffer. A lower floating dtype is fine — half to full is
#: exact — which is what the release does with the adaLN matrices.
_FP32_ISLANDS = ("video_patch_proj.weight", "adaln_t_table",
                 "final_layer.video_out.weight", *_AUDIO_KEYS)
_STORES_F32 = ("F32", "F64", "F16", "BF16")


def detect_params(checkpoint: Union[str, Header], signature: Signature) -> H3Params:
    """Read the architecture, the head-bank rows and its sigma grid out of a file.

    The head count is a shape and the sigma grid is only in the file's ``config``
    metadata, and the two have to agree; everything else the model needs to be built is a
    shape too. A file with no metadata — every pruned video checkpoint — therefore reads
    as one head over the plain two-point grid, which is the flow schedule's own endpoints.
    """
    header = as_header(checkpoint)
    if _UNPRUNED in header.shapes:
        raise ValueError(
            f"{header.path} is an unpruned H3 checkpoint: it carries "
            f"'{_UNPRUNED}' and adaLN matrices wider than the trunk, where this model "
            f"reads the 'adaln_t_table' curve. Only the pruned release can be loaded."
        )
    header.require(signature)
    _require_storable_islands(header)

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

    audio = _audio_dim(header)
    num_heads = qkv[0] // (3 * head_dim)
    params = H3Params(
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
        audio_dim=audio,
    )
    logger.info(
        "H3 DiT: %d layers, %dx%d heads, %d-step head bank%s",
        params.num_layers, params.num_heads, params.head_dim, params.pdd_steps,
        f", audio {audio}ch" if audio else "",
    )
    return params


def _require_storable_islands(header: Header) -> None:
    """Refuse a file that stores an FP32 island in a packed layout.

    The pruned ``w6a8`` release does exactly this to ``final_layer.video_out``: 6-bit
    codes plus group scales where the model wants a calibrated FP32 head. Reading it as
    though it were full precision would widen nothing and quantize the output head twice
    over, so it is named instead.
    """
    for name in _FP32_ISLANDS:
        dtype = header.dtypes.get(name)
        if dtype is not None and dtype not in _STORES_F32:
            raise ValueError(
                f"{header.path}: '{name}' is stored {dtype}, but this model keeps it in "
                f"FP32. Pick a variant that leaves the output head unquantized "
                f"(pruned_bf16, pruned_int8_convrot or pruned_fp8_scaled)."
            )


def _audio_dim(header: Header):
    """The audio stream's row width, or ``None`` for a file without one.

    The stream's two names state it twice — as the channels its projection takes and as
    the channels its head emits — and a file that disagrees between them describes a
    model this trunk cannot wire.
    """
    patch = header.shapes.get("audio_patch_proj.weight")
    if patch is None:
        return None
    out = header.shape("final_layer.audio_out.weight")[0]
    if out != patch[1]:
        raise ValueError(
            f"{header.path}: 'audio_patch_proj.weight' takes {patch[1]} channels but "
            f"'final_layer.audio_out.weight' emits {out}; the audio stream's two widths "
            f"have to be one width"
        )
    return patch[1]


def load_h3_image_dit(
    checkpoint: Union[str, Header],
    params: H3Params,
    device,
    dtype: torch.dtype,
) -> H3ImageDiT:
    """Build the image DiT on meta from the checkpoint's own geometry, then load its weights.

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
        relay_stores=RELAY_STORES,
    )
    model.to(dtype=dtype)
    return model


__all__ = [
    "H3_AV_SIGNATURE",
    "H3_IMAGE_SIGNATURE",
    "RELAY_STORES",
    "detect_params",
    "load_h3_image_dit",
]
