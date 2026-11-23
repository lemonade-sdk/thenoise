"""Qwen-Image 2.1 DiT loading: architecture from the header, weights via ``build_dit``.

At load time the attention QK norms (``attn.norm_q`` / ``attn.norm_k``) are mapped
onto the shared ``QKNorm`` layout, and ``txt_in.text_norm`` — a *zero-centred*
RMSNorm — has the 1 added back so the plain ``RMSNorm`` runs it.
"""
from __future__ import annotations

import logging
from typing import Union

import torch

from thenoise.dit.qwen_image21.models import (
    QwenImage21Params,
    QwenImage21Transformer2DModel,
)
from thenoise.utils.loader import build_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import (
    Header,
    Signature,
    as_header,
    require_divisible,
)

logger = logging.getLogger(__name__)

QWEN_IMAGE21_SIGNATURE = Signature(
    "Qwen-Image 2.1 DiT",
    #: Its per-run ``modulation`` and zero-centred ``txt_in.text_norm``, over the
    #: family's projections.
    keys=(
        "txt_in.text_norm.weight",
        "modulation.1.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "img_in.weight",
        "proj_out.weight",
    ),
)


def detect_params(checkpoint: Union[str, Header]) -> QwenImage21Params:
    """Read the architecture knobs out of a Qwen-Image 2.1 checkpoint header."""
    header = as_header(checkpoint)
    header.require(QWEN_IMAGE21_SIGNATURE)

    img_in = header.shape("img_in.weight")
    inner_dim = img_in[0]
    head_dim = header.shape("transformer_blocks.0.attn.norm_q.weight")[0]
    require_divisible(inner_dim, head_dim, "img_in width", "head dim")

    # A split MLP names its projections ``proj``/``gate_layer`` (``fused_mlp=False``).
    gate_up = header.shapes.get("transformer_blocks.0.img_mlp.gate_up.weight")
    if gate_up is not None:
        mlp_ratio, fused_mlp = gate_up[0] // 2 // inner_dim, True
    else:
        proj = header.shapes.get("transformer_blocks.0.img_mlp.proj.weight")
        if proj is None:
            raise ValueError(
                f"{header.path} has neither img_mlp.gate_up nor img_mlp.proj weights"
            )
        mlp_ratio, fused_mlp = proj[0] // inner_dim, False

    params = QwenImage21Params(
        in_channels=img_in[1],
        out_channels=header.shape("proj_out.weight")[0],
        num_layers=header.block_count("transformer_blocks."),
        attention_head_dim=head_dim,
        num_attention_heads=inner_dim // head_dim,
        context_in_dim=header.shape("txt_in.text_norm.weight")[0],
        mlp_ratio=mlp_ratio,
        fused_mlp=fused_mlp,
    )
    logger.info(
        "Qwen-Image 2.1 DiT: %d layers, %dx%d, context %d, mlp_ratio %d (%s MLP)",
        params.num_layers, params.num_attention_heads, params.attention_head_dim,
        params.context_in_dim, params.mlp_ratio, "fused" if fused_mlp else "split",
    )
    return params


def _key_map(key: str) -> str:
    return qk_norm_key_map(key, "norm_q", "norm_k")


def _value_map(key: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Un-centre the zero-centred RMSNorm: the checkpoint stores ``scale - 1``."""
    if key == "txt_in.text_norm.weight":
        return key, tensor + 1.0
    return key, tensor


def load_qwen_image21_dit(
    checkpoint: Union[str, Header],
    params: QwenImage21Params,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> QwenImage21Transformer2DModel:
    """Build the DiT on meta from the checkpoint's own geometry, then load its weights."""
    return build_dit(
        lambda: QwenImage21Transformer2DModel(params),
        checkpoint,
        device=device,
        dtype=dtype,
        key_map=_key_map,
        value_map=_value_map,
        relay_stores=True,
    )


__all__ = [
    "QWEN_IMAGE21_SIGNATURE",
    "detect_params",
    "load_qwen_image21_dit",
]
