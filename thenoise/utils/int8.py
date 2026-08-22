"""INT8 (INT8+ConvRot) checkpoint loading helpers, shared across models.

The INT8 ConvRot checkpoints produced by ComfyUI's exporter store each quantized
linear as three tensors: an int8 ``.weight``, a per-row F32 ``.weight_scale``,
and a small U8 ``.comfy_quant`` metadata marker (not needed at inference). Layers
kept in full precision stay as plain BF16 ``.weight``/``.bias``.

These helpers are model-agnostic: any module exposing a ``load_int8(qweight,
scale)`` method (e.g. ``thenoise.dit.quantized.QuantizedLinear``) is switched to
INT8 at load time; every other layer is assigned normally.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from thenoise.utils.safetensors import (
    MemoryEfficientSafeOpen,
    WRAP_PREFIXES,
    load_dit_safetensors,
)
from thenoise.utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Suffix used by ComfyUI's INT8 exporter for the per-row weight scale.
_WEIGHT_SCALE_SUFFIX = ".weight_scale"
# U8 marker ComfyUI stores per quantized layer; not needed at inference.
_COMFY_QUANT_SUFFIX = ".comfy_quant"


def is_int8_checkpoint(dit_path: str) -> bool:
    """Return True if ``dit_path`` is an INT8+ConvRot checkpoint.

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
    INT8 kernels emit BF16, so full-precision params must match). ``comfy_quant``
    metadata markers are ignored.

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

    for key, tensor in state_dict.items():
        if key.endswith(_WEIGHT_SCALE_SUFFIX) or key.endswith(_COMFY_QUANT_SUFFIX):
            continue
        module_path, _, attr = key.rpartition(".")
        module = _submodule(model, module_path, key)
        if attr == "weight" and tensor.dtype == torch.int8:
            _switch_to_int8(module, tensor, scales.pop(module_path, None), key)
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


def _switch_to_int8(module: torch.nn.Module, qweight: torch.Tensor, scale, key: str) -> None:
    if scale is None:
        raise RuntimeError(f"INT8 weight {key!r} is missing its {_WEIGHT_SCALE_SUFFIX}")
    if not hasattr(module, "load_int8"):
        raise RuntimeError(
            f"INT8 weight {key!r} landed on {type(module).__name__}, "
            "which has no load_int8(); it must be a QuantizedLinear"
        )
    module.load_int8(qweight, scale)


def load_int8_if_present(
    model: torch.nn.Module,
    dit_path: str,
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    key_map: Optional[callable] = None,
) -> bool:
    """Load an INT8+ConvRot checkpoint into ``model`` if ``dit_path`` is one.

    Centralizes the INT8 detection + loading shared by every model adapter: if
    the checkpoint is INT8 it is fully loaded (native int8 weights/scales, full-
    precision params cast to ``dtype``) and True is returned; otherwise the
    caller loads the BF16 checkpoint as usual and False is returned. Handles the
    final device move of buffers/parameters.

    ``key_map`` (optional) transforms checkpoint keys before loading; it is used
    to reconcile exporter-specific renames (e.g. ComfyUI's INT8 exporter stores
    norm ``scale`` params under ``weight``).
    """
    if not is_int8_checkpoint(dit_path):
        return False
    device = torch.device(device)
    sd = load_dit_safetensors(dit_path, device=device, disable_mmap=True, dtype=None)
    if key_map is not None:
        sd = {key_map(k): v for k, v in sd.items()}
    load_int8_state_dict(model, sd, dtype=dtype)
    model.to(device)
    logger.info("Loaded INT8+ConvRot checkpoint from %s", dit_path)
    return True


__all__ = ["is_int8_checkpoint", "load_int8_state_dict", "load_int8_if_present"]
