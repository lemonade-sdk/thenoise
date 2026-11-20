"""Fibo 2.0 DiT loading: geometry out of the safetensors header.

Reading the knobs off the file lets an export of a different depth or width load
without touching this package, and turns a wrong checkpoint into a clear message
instead of a shape error halfway through the load. Which blocks cross-attend to text
is read off the presence of their gate, so a release that injects somewhere else needs
no change here.

Both released checkpoints store ``qkv``, ``to_kv`` and ``w13`` **fused**, which is also
this module's layout; the only bridge needed is the QK-norm naming, and the geometry is
read under the names the model uses (:meth:`Header.relabel`).
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from thenoise.dit.fibo2.models import Fibo2Params, Fibo2Transformer2DModel
from thenoise.utils.loader import build_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import (
    Header,
    Signature,
    as_header,
    require_divisible,
)

logger = logging.getLogger(__name__)

#: The two SwiGLU spellings: fused (the release) or apart.
_FFN_KEYS = ("layers.0.feed_forward.w13.weight", "layers.0.feed_forward.w1.weight")

FIBO2_SIGNATURE = Signature(
    "Fibo 2.0 DiT",
    #: The Perceiver and the single-stream blocks together are the whole fingerprint.
    keys=(
        "perceiver.latent_queries",
        "layers.0.attention.qkv.weight",
        "layers.0.adaLN_modulation.0.weight",
        "t_embedder.mlp.0.weight",
        "final_layer.linear.weight",
        "perceiver.layers.0.cross_attention.to_kv.weight",
    ),
)


def fibo2_key_map(key: str) -> str:
    """Bridge the checkpoint's two QK-norm spellings onto the shared ``QKNorm``.

    The rotated self-attention stores ``q_norm``/``k_norm``, the unrotated
    cross-attention ``norm_q``/``norm_k``.
    """
    key = qk_norm_key_map(key)
    return qk_norm_key_map(key, q_legacy="norm_q", k_legacy="norm_k")


def detect_params(checkpoint: Union[str, Header]) -> Fibo2Params:
    """Read the Fibo 2.0 architecture knobs out of a checkpoint header."""
    header = as_header(checkpoint).relabel(fibo2_key_map)
    header.require(FIBO2_SIGNATURE)

    max_gist_tokens, dim = header.shape("perceiver.latent_queries")
    head_dim = header.shape("layers.0.attention.qk_norm.query_norm.weight")[0]
    require_divisible(dim, head_dim, "dim", "head dim")

    fused = _FFN_KEYS[0] in header
    if not fused and _FFN_KEYS[1] not in header:
        raise ValueError(
            f"{header.path} has neither the fused ({_FFN_KEYS[0]}) nor the split "
            f"({_FFN_KEYS[1]}) SwiGLU layout"
        )
    num_layers = header.block_count("layers.")

    t_embed = header.shape("t_embedder.mlp.0.weight")
    params = Fibo2Params(
        dim=dim,
        ffn_dim=header.shape(_FFN_KEYS[0] if fused else _FFN_KEYS[1])[0] // (2 if fused else 1),
        num_heads=dim // head_dim,
        num_layers=num_layers,
        num_refiner_layers=header.block_count("noise_refiner."),
        perceiver_layers=header.block_count("perceiver.layers."),
        injection_layers=tuple(
            i for i in range(num_layers) if f"layers.{i}.cross_attn_gate" in header
        ),
        latent_channels=header.shape("final_layer.linear.weight")[0],
        cap_dim=header.shape("perceiver.layers.0.cross_attention.to_kv.weight")[1],
        max_gist_tokens=max_gist_tokens,
        adaln_dim=header.shape("layers.0.adaLN_modulation.0.weight")[1],
        t_embed_dim=t_embed[0],
        freq_dim=t_embed[1],
        fused_ffn=fused,
    )
    logger.info(
        "Fibo 2.0 DiT: %d blocks (%d refiner), dim %d / ffn %d, %dx%d heads, "
        "latent %d, %d injections over %d gist queries",
        params.num_layers, params.num_refiner_layers, params.dim, params.ffn_dim,
        params.num_heads, head_dim, params.latent_channels,
        len(params.injection_layers), params.max_gist_tokens,
    )
    return params


def load_fibo2_dit(
    checkpoint: Union[str, Header],
    params: Fibo2Params,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype],
) -> Fibo2Transformer2DModel:
    """Build the DiT from the checkpoint's own geometry, then load its weights."""
    return build_dit(
        lambda: Fibo2Transformer2DModel(params),
        checkpoint,
        device=device,
        dtype=dtype,
        key_map=fibo2_key_map,
    )


__all__ = ["FIBO2_SIGNATURE", "detect_params", "fibo2_key_map", "load_fibo2_dit"]
