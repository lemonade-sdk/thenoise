"""INT8 checkpoint loading helpers, shared across models.

The INT8 ConvRot checkpoints produced by ComfyUI's exporter store each quantized
linear as three tensors: an int8 ``.weight``, a per-row F32 ``.weight_scale``,
and a small U8 ``.comfy_quant`` JSON marker that records the quantization
profile (``convrot`` flag and ``convrot_groupsize``). The group size is baked
into the stored (rotated) weights and per-row scales, so inference MUST rotate
activations with the same group size (and only when the layer was actually
rotated), or the dequantized GEMM is garbage. Layers kept in full precision
stay as plain BF16 ``.weight``/``.bias``.

These helpers are model-agnostic: any module exposing a ``load_int8(qweight,
scale, ...)`` method (e.g. ``thenoise.dit.quantized.QuantizedLinear``) is
switched to INT8 at load time; every other layer is assigned normally.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import torch

from thenoise.utils.safetensors import MemoryEfficientSafeOpen, WRAP_PREFIXES
from thenoise.utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Suffix used by ComfyUI's INT8 exporter for the per-row weight scale.
_WEIGHT_SCALE_SUFFIX = ".weight_scale"
# U8 JSON marker ComfyUI stores per quantized layer, recording its ConvRot profile.
_COMFY_QUANT_SUFFIX = ".comfy_quant"


def is_int8_checkpoint(dit_path: str) -> bool:
    """Return True if ``dit_path`` is an INT8 checkpoint.

    Detects the presence of any ``.weight_scale`` key in the safetensors header
    (after stripping generic repackaging wrapper prefixes). Reads the header
    only — no tensors are loaded.
    """
    with MemoryEfficientSafeOpen(dit_path) as f:
        for key in f.keys():
            for prefix in WRAP_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if key.endswith(_WEIGHT_SCALE_SUFFIX):
                return True
    return False


def _parse_comfy_quant(tensor: torch.Tensor) -> dict:
    """Decode a per-layer ``comfy_quant`` marker tensor into its JSON dict.

    ComfyUI's INT8 exporter stores the marker as a small U8 tensor containing
    the JSON payload, e.g. ``{"convrot": true, "convrot_groupsize": 256,
    "per_row": true}`` (or ``{"convrot": false, "per_row": true}`` when a
    layer's ``in_features`` is not divisible by the group size and rotation was
    skipped).
    """
    try:
        data = json.loads(tensor.detach().cpu().numpy().tobytes().decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError):
        logger.warning(
            "Could not parse a comfy_quant marker; using default INT8 profile "
            "(convrot=True, group size 256)"
        )
        return {}
    return data if isinstance(data, dict) else {}


def load_int8_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
) -> None:
    """Populate ``model`` from an INT8+ConvRot state dict.

    Quantized linear weights (int8 ``.weight`` paired with a per-row
    ``.weight_scale``) land on modules that implement ``load_int8``; every other
    leaf parameter (kept in full precision — ``weight``/``bias``/embedding tokens
    etc.) is replaced with the loaded tensor, cast to ``dtype`` when given (the
    INT8 kernels emit BF16, so full-precision params must match).

    The per-layer ``.comfy_quant`` JSON marker is decoded and passed along, so
    each module rotates activations with the exact group size it was quantized
    at, and only when the layer was actually ConvRot-rotated (``convrot``).

    ``state_dict`` must already have generic wrapper prefixes stripped (see
    ``thenoise.utils.safetensors.load_dit_safetensors``).
    """
    # Scales are collected before weights are processed: an int8 ``.weight``
    # needs its per-row ``.weight_scale``, and dict order is not guaranteed to
    # place the scale before the weight it belongs to.
    scales: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(_WEIGHT_SCALE_SUFFIX):
            scales[key[: -len(_WEIGHT_SCALE_SUFFIX)]] = tensor

    # ``comfy_quant`` markers record each quantized layer's ConvRot profile;
    # decode them up front so int8 weights can pick up their marker by path.
    markers: dict[str, dict] = {}
    for key, tensor in state_dict.items():
        if key.endswith(_COMFY_QUANT_SUFFIX):
            markers[key[: -len(_COMFY_QUANT_SUFFIX)]] = _parse_comfy_quant(tensor)

    for key, tensor in state_dict.items():
        if key.endswith(_WEIGHT_SCALE_SUFFIX) or key.endswith(_COMFY_QUANT_SUFFIX):
            continue
        module_path, _, attr = key.rpartition(".")
        module = _submodule(model, module_path, key)
        if attr == "weight" and tensor.dtype == torch.int8:
            marker = markers.get(module_path, {})
            _switch_to_int8(
                module,
                tensor,
                scales.pop(module_path, None),
                key,
                convrot=bool(marker.get("convrot", True)),
                convrot_groupsize=marker.get("convrot_groupsize"),
            )
        elif isinstance(getattr(module, attr, None), torch.nn.Parameter):
            # BF16/full-precision leaf parameter (weight, bias, pad tokens, ...):
            # replace the (meta, init-time) parameter rather than ``param.data =
            # ...``, because set_data rejects meta params and dtype mismatches.
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            setattr(module, attr, torch.nn.Parameter(tensor))
        else:
            raise RuntimeError(f"unexpected key in INT8 checkpoint: {key!r}")

    if scales:
        raise RuntimeError(
            f"orphan {_WEIGHT_SCALE_SUFFIX} keys in INT8 checkpoint: {list(scales)[:5]}"
        )


def _submodule(model: torch.nn.Module, module_path: str, key: str) -> torch.nn.Module:
    """Resolve a checkpoint key's module path, with a clear error on mismatch."""
    try:
        return model.get_submodule(module_path)
    except AttributeError as e:
        raise RuntimeError(
            f"INT8 checkpoint key {key!r} does not match the model structure: {e}"
        ) from e


def build_int8_restore_map(
    path: str,
    int8_key_map: Optional[callable] = None,
) -> dict[str, str]:
    """Map INT8 module paths (post key-map) to their raw checkpoint weight keys.

    Captured once at load time (when wrapper-prefix stripping and ``int8_key_map``
    are already resolved) so a later LoRA undo can reload the original INT8
    weights from disk by raw key — no re-deriving the mapping logic. Only reads
    the safetensors header; no tensors are materialized.

    Returns ``{module_path: raw_weight_key}`` for every quantized linear weight
    in the file (an I8 tensor with a sibling ``.weight_scale``).
    """
    restore: dict[str, str] = {}
    with MemoryEfficientSafeOpen(path) as f:
        for raw_key in f.keys():
            if f.header[raw_key]["dtype"] != "I8":
                continue
            if not raw_key.endswith(".weight"):
                continue
            if raw_key[: -len(".weight")] + _WEIGHT_SCALE_SUFFIX not in f.header:
                continue  # not a quantized linear weight
            stripped = raw_key
            for prefix in WRAP_PREFIXES:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break
            mapped = int8_key_map(stripped) if int8_key_map is not None else stripped
            if not mapped.endswith(".weight"):
                continue
            restore[mapped[: -len(".weight")]] = raw_key
    return restore


def restore_int8_layer(
    module: torch.nn.Module,
    path: str,
    raw_key: str,
) -> None:
    """Restore an INT8 layer's ``qweight``/``scale`` from a checkpoint by raw key.

    Reads only the exact tensors the layer needs (the int8 ``weight`` and its
    per-row ``.weight_scale``) straight from the file by key, avoiding a full
    reload. Used by LoRA undo to return baked LoRAs to the original weights.
    """
    with MemoryEfficientSafeOpen(path) as f:
        qweight = f.get_tensor(raw_key, device=module.qweight.device, dtype=torch.int8)
        scale = f.get_tensor(
            raw_key[: -len(".weight")] + _WEIGHT_SCALE_SUFFIX,
            device=module.scale.device,
        )
    module.qweight = qweight
    module.scale = scale


def _switch_to_int8(
    module: torch.nn.Module,
    qweight: torch.Tensor,
    scale,
    key: str,
    convrot: bool = True,
    convrot_groupsize: Optional[int] = None,
) -> None:
    if scale is None:
        raise RuntimeError(f"INT8 weight {key!r} is missing its {_WEIGHT_SCALE_SUFFIX}")
    if not hasattr(module, "load_int8"):
        raise RuntimeError(
            f"INT8 weight {key!r} landed on {type(module).__name__}, "
            "which has no load_int8(); it must be a QuantizedLinear"
        )
    module.load_int8(qweight, scale, convrot=convrot, convrot_groupsize=convrot_groupsize)


__all__ = [
    "is_int8_checkpoint",
    "load_int8_state_dict",
    "build_int8_restore_map",
    "restore_int8_layer",
]
