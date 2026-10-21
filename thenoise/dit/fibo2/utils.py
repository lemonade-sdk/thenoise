"""Fibo 2.0 DiT loading: geometry out of the safetensors header.

Reading the knobs off the file lets an export of a different depth or width load
without touching this package, and turns a wrong checkpoint into a clear message
instead of a shape error halfway through the load. Which blocks cross-attend to text
is read off the presence of their gate, so a release that injects somewhere else needs
no change here.

Both released checkpoints store ``qkv``, ``to_kv`` and ``w13`` **fused**, which is also
this module's layout; the only bridge needed is the QK-norm naming.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights

from thenoise.dit.fibo2.models import Fibo2Params, Fibo2Transformer2DModel
from thenoise.utils.loader import load_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key

logger = logging.getLogger(__name__)

#: Names a Fibo 2.0 DiT must carry, whatever wrapper it was repackaged in. The Perceiver
#: and the single-stream blocks together are the whole fingerprint.
_SIGNATURE_KEYS = (
    "perceiver.latent_queries",
    "layers.0.attention.qkv.weight",
    "layers.0.adaLN_modulation.0.weight",
    "t_embedder.mlp.0.weight",
    "final_layer.linear.weight",
)

#: The two SwiGLU spellings: fused (the release) or apart.
_FFN_KEYS = ("layers.0.feed_forward.w13.weight", "layers.0.feed_forward.w1.weight")


def is_fibo2(keys: Iterable[str]) -> bool:
    """True for a Fibo 2.0 key-set, under either SwiGLU spelling.

    ``keys`` must be wrapper-prefix free (see ``normalize_keys``).
    """
    keys = set(keys)
    return all(key in keys for key in _SIGNATURE_KEYS) and any(
        key in keys for key in _FFN_KEYS
    )


def fibo2_key_map(key: str) -> str:
    """Bridge the checkpoint's two QK-norm spellings onto the shared ``QKNorm``.

    The rotated self-attention stores ``q_norm``/``k_norm``, the unrotated
    cross-attention ``norm_q``/``norm_k``.
    """
    key = qk_norm_key_map(key)
    return qk_norm_key_map(key, q_legacy="norm_q", k_legacy="norm_k")


def _header_shapes(path: str) -> Dict[str, Tuple[int, ...]]:
    """Checkpoint key -> shape, in this module's naming, without reading a tensor.

    Mapped, not merely unwrapped: the geometry below is read under the names the model
    uses, so a file that spells its QK norms the older way is read like any other.
    """
    with MemoryEfficientSafeOpen(path) as f:
        return {
            fibo2_key_map(unwrap_key(k)): tuple(f.header[k]["shape"]) for k in f.keys()
        }


def _block_count(shapes: Mapping[str, Tuple[int, ...]], prefix: str) -> int:
    """How many ``<prefix><i>.`` blocks the key-set spans (0 if none)."""
    indices = []
    for key in shapes:
        if not key.startswith(prefix):
            continue
        head = key[len(prefix):].split(".")[0]
        if head.isdigit():
            indices.append(int(head))
    return max(indices) + 1 if indices else 0


def detect_params(dit_path: str) -> Fibo2Params:
    """Read the Fibo 2.0 architecture knobs out of a checkpoint header."""
    shapes = _header_shapes(dit_path)
    if not is_fibo2(shapes):
        missing = [k for k in _SIGNATURE_KEYS + _FFN_KEYS if k not in shapes]
        raise ValueError(f"{dit_path} is not a Fibo 2.0 DiT (missing {missing})")

    max_gist_tokens, dim = shapes["perceiver.latent_queries"]
    head_dim = shapes["layers.0.attention.qk_norm.query_norm.weight"][0]
    if dim % head_dim:
        raise ValueError(f"dim {dim} is not a multiple of head dim {head_dim}")

    num_layers = _block_count(shapes, "layers.")
    fused = _FFN_KEYS[0] in shapes
    w_key = _FFN_KEYS[0] if fused else _FFN_KEYS[1]
    t_embed = shapes["t_embedder.mlp.0.weight"]

    params = Fibo2Params(
        dim=dim,
        ffn_dim=shapes[w_key][0] // (2 if fused else 1),
        num_heads=dim // head_dim,
        num_layers=num_layers,
        num_refiner_layers=_block_count(shapes, "noise_refiner."),
        perceiver_layers=_block_count(shapes, "perceiver.layers."),
        injection_layers=tuple(
            i for i in range(num_layers) if f"layers.{i}.cross_attn_gate" in shapes
        ),
        latent_channels=shapes["final_layer.linear.weight"][0],
        cap_dim=shapes["perceiver.layers.0.cross_attention.to_kv.weight"][1],
        max_gist_tokens=max_gist_tokens,
        adaln_dim=shapes["layers.0.adaLN_modulation.0.weight"][1],
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
    dit_path: str,
    params: Fibo2Params,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype],
) -> Fibo2Transformer2DModel:
    """Build the DiT from the checkpoint's own geometry, then load its weights."""
    with init_empty_weights():
        model = Fibo2Transformer2DModel(params)
    load_dit(model, dit_path, device=device, dtype=dtype, key_map=fibo2_key_map)
    logger.info("Loaded Fibo 2.0 DiT from %s", dit_path)
    return model


__all__ = ["detect_params", "fibo2_key_map", "is_fibo2", "load_fibo2_dit"]
