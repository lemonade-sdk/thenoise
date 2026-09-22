"""Qwen-Image 2.1 DiT loading: architecture from the header, weights via ``load_dit``.

Two reconciliations happen at load time, both so the model can be built out of the
repo's shared modules instead of checkpoint-shaped ones: the attention QK norms
(``attn.norm_q`` / ``attn.norm_k``) land on the shared ``QKNorm`` layout, and
``txt_in.text_norm`` — a *zero-centred* RMSNorm whose stored weight is ``scale - 1``
— has the 1 added back so the plain ``RMSNorm`` runs it.
"""
from __future__ import annotations

import logging
from typing import Tuple, Union

import torch
from accelerate import init_empty_weights

from thenoise.dit.qwen_image21.models import QwenImage21Params, QwenImage21Transformer2DModel
from thenoise.utils.loader import load_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key

logger = logging.getLogger(__name__)

#: Keys that identify a Qwen-Image 2.1 DiT regardless of repackaging wrappers.
_SIGNATURE_KEYS = (
    "txt_in.text_norm.weight",
    "modulation.1.weight",
    "transformer_blocks.0.attn.norm_q.weight",
    "img_in.weight",
    "proj_out.weight",
)


def is_qwen_image21_key(keys) -> bool:
    """True if these tensor names are a Qwen-Image 2.1 DiT.

    The shared ``modulation`` plus the zero-centred ``txt_in.text_norm`` are unique
    to this architecture. Names are unwrapped, so a repackaged checkpoint matches
    identically.
    """
    names = {unwrap_key(k) for k in keys}
    return all(k in names for k in _SIGNATURE_KEYS)


def _header_shapes(path: str) -> dict[str, Tuple[int, ...]]:
    """Unwrapped checkpoint key -> tensor shape (header only, no tensors read)."""
    with MemoryEfficientSafeOpen(path) as f:
        return {unwrap_key(k): tuple(f.header[k]["shape"]) for k in f.keys()}


def detect_params(dit_path: str) -> QwenImage21Params:
    """Read the architecture knobs out of a Qwen-Image 2.1 checkpoint header."""
    shapes = _header_shapes(dit_path)
    missing = [k for k in _SIGNATURE_KEYS if k not in shapes]
    if missing:
        raise ValueError(f"{dit_path} is not a Qwen-Image 2.1 DiT (missing {missing})")

    img_in = shapes["img_in.weight"]
    head_dim = shapes["transformer_blocks.0.attn.norm_q.weight"][0]
    inner_dim = img_in[0]
    if inner_dim % head_dim:
        raise ValueError(f"img_in width {inner_dim} is not a multiple of head dim {head_dim}")
    num_layers = 1 + max(
        int(k.split(".")[1]) for k in shapes
        if k.startswith("transformer_blocks.") and k.split(".")[1].isdigit()
    )

    # The released checkpoints fuse SwiGLU's gate and up projections into one
    # ``gate_up`` matrix; a split (diffusers-style) file names them ``proj``/
    # ``gate_layer`` and has to be built with ``fused_mlp=False``.
    gate_up = shapes.get("transformer_blocks.0.img_mlp.gate_up.weight")
    if gate_up is not None:
        mlp_ratio, fused_mlp = gate_up[0] // 2 // inner_dim, True
    else:
        proj = shapes.get("transformer_blocks.0.img_mlp.proj.weight")
        if proj is None:
            raise ValueError(
                f"{dit_path} has neither img_mlp.gate_up nor img_mlp.proj weights"
            )
        mlp_ratio, fused_mlp = proj[0] // inner_dim, False

    params = QwenImage21Params(
        in_channels=img_in[1],
        out_channels=shapes["proj_out.weight"][0],
        num_layers=num_layers,
        attention_head_dim=head_dim,
        num_attention_heads=inner_dim // head_dim,
        context_in_dim=shapes["txt_in.text_norm.weight"][0],
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
    dit_path: str,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> QwenImage21Transformer2DModel:
    """Build the DiT on meta from the checkpoint's own geometry, then load its weights."""
    params = detect_params(dit_path)
    logger.info("Loading Qwen-Image 2.1 DiT from %s", dit_path)
    with init_empty_weights():
        dit = QwenImage21Transformer2DModel(params)
    load_dit(
        dit,
        dit_path,
        device=device,
        dtype=dtype,
        # The reference writes the timestep-zero marker; this model has no other
        # reference method, so it is metadata rather than a weight.
        drop_keys=("__index_timestep_zero__",),
        key_map=_key_map,
        value_map=_value_map,
    )
    return dit


__all__ = [
    "detect_params",
    "is_qwen_image21_key",
    "load_qwen_image21_dit",
]
