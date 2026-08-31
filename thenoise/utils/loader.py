"""Central quant-aware DiT loader.

Every DiT adapter loads its weights through a single entry point, ``load_dit``,
which automatically selects the correct quantization:

* **Quantized** checkpoint (INT8 or FP8, detected via ``is_quantized_checkpoint``):
  weights land via ``load_quantized_state_dict`` — quantized linears on modules
  exposing ``load_quantized`` (``QuantizedLinear``) receive a reconstructed
  ``comfy_kitchen.QuantizedTensor``, and full-precision params are cast to
  ``dtype``.
* **BF16 / plain** checkpoint: weights land via ``load_state_dict(assign=True)``,
  with the loaded tensors cast to ``dtype``.

The quantized loading helpers below are model-agnostic: any module exposing a
``load_quantized(QuantizedTensor)`` method (e.g. ``thenoise.dit.quantized.
QuantizedLinear``) is switched to its quantized layout at load time; every other
layer is assigned normally. The loader detects the layout from the stored weight
dtype (int8 vs FP8 E4M3/E5M2) and reconstructs a ``QuantizedTensor`` carrying the
decoded ``comfy_quant`` marker profile, so inference rotates activations with the
exact group size a layer was quantized at — and only when it was actually
ConvRot-rotated.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from typing import Callable, Optional, Union

import torch

from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout

from thenoise.utils.safetensors import (
    MemoryEfficientSafeOpen,
    WRAP_PREFIXES,
    load_dit_safetensors,
)
from thenoise.utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Suffix used by ComfyUI's quantized exporter for the weight scale.
_WEIGHT_SCALE_SUFFIX = ".weight_scale"
# U8 JSON marker ComfyUI stores per quantized layer, recording its profile.
_COMFY_QUANT_SUFFIX = ".comfy_quant"
# Storage dtypes that mark a tensor as a quantized linear weight.
_QUANT_DTYPES = [torch.int8]
for _name in ("float8_e4m3fn", "float8_e5m2"):
    if hasattr(torch, _name):
        _QUANT_DTYPES.append(getattr(torch, _name))
_QUANT_DTYPES = tuple(_QUANT_DTYPES)


def load_dit(
    model: torch.nn.Module,
    path: str,
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    drop_keys: Optional[tuple[str, ...]] = None,
    expected_missing: tuple[str, ...] = (),
    key_map: Optional[Callable[[str], str]] = None,
) -> torch.nn.Module:
    """Load a DiT checkpoint into ``model``, selecting quantized vs BF16 automatically.

    Args:
        model: the (meta-constructed) DiT to populate.
        path: safetensors checkpoint path.
        device: final device for all parameters/buffers.
        dtype: dtype for full-precision (BF16-path and non-quantized) params.
        drop_keys: optional tuple of key prefixes to drop from the checkpoint
            before loading (e.g. Krea2's unused ``last.down``/``last.up``).
            Applied on BOTH the quantized and BF16 paths.
        expected_missing: optional substrings allowed to be absent from the
            checkpoint (model-internal buffers, e.g. Anima's RoPE buffers). If
            non-empty the load is non-strict and only these are tolerated;
            otherwise loading is strict.
        key_map: optional transform applied to checkpoint keys on the
            quantized path only (e.g. Flux Klein's ComfyUI norm ``weight`` ->
            ``scale`` rename). BF16 checkpoints carry the canonical names already.

    Returns:
        ``model`` (loaded in place, moved to ``device``).
    """
    device = torch.device(device)

    # Load the state dict once (stripping generic wrapper prefixes inside
    # ``load_dit_safetensors``) and apply ``drop_keys`` before branching, so the
    # quantized and BF16 paths see the same prepared dict.
    sd = load_dit_safetensors(path, device=device, disable_mmap=True, dtype=None)
    if drop_keys:
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_keys)}

    if is_quantized_checkpoint(path):
        if key_map is not None:
            sd = {key_map(k): v for k, v in sd.items()}
        # The quantization profile (ConvRot group size / rotation flag for INT8,
        # stored format for FP8) is baked into each layer's weights and scales at
        # export time and must match at inference; ``load_quantized_state_dict``
        # reads it from each layer's ``comfy_quant`` marker in the state dict
        # and reconstructs the ``QuantizedTensor`` layout accordingly.
        load_quantized_state_dict(model, sd, dtype=dtype)
        # Record each quantized layer's raw checkpoint key so a later LoRA undo
        # can reload the original weights from disk by key.
        model._quantized_restore_map = build_quantized_restore_map(path, key_map)
        logger.info("Loaded quantized checkpoint from %s", path)
    else:
        if dtype is not None:
            sd = {k: v.to(dtype=dtype) for k, v in sd.items()}
        _load_bf16(model, sd, expected_missing, path)

    # Move the whole model onto the device, including buffers not present in the
    # checkpoint (e.g. RoPE position-embedding buffers that were created on meta
    # and are not saved in the file).
    model.to(device)
    return model


def _load_bf16(
    model: torch.nn.Module,
    sd: dict[str, torch.Tensor],
    expected_missing: tuple[str, ...],
    path: str,
) -> None:
    """Load a plain/BF16 state dict, strict unless ``expected_missing`` is set."""
    if expected_missing:
        info = model.load_state_dict(sd, strict=False, assign=True)
        missing = [
            k for k in info.missing_keys
            if not any(buf in k for buf in expected_missing)
        ]
        if missing or info.unexpected_keys:
            raise RuntimeError(
                f"checkpoint {path!r} did not match the model: "
                f"missing={missing[:10]}, "
                f"unexpected={info.unexpected_keys[:10]}"
            )
    else:
        model.load_state_dict(sd, strict=True, assign=True)
    logger.info("Loaded BF16 checkpoint from %s", path)


# --------------------------------------------------------------------------- quantized checkpoint helpers


def is_quantized_checkpoint(dit_path: str) -> bool:
    """Return True if ``dit_path`` is a quantized (INT8 or FP8) checkpoint.

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

    ComfyUI's exporter stores the marker as a small U8 tensor containing the
    JSON payload, e.g. ``{"convrot": true, "convrot_groupsize": 256,
    "per_row": true}`` for an INT8 layer (or ``{"convrot": false,
    "per_row": true}`` when ``in_features`` was not divisible by the group size
    and rotation was skipped), and ``{"format": "float8_e4m3fn",
    "full_precision_matrix_mult": true}`` for an FP8 layer.
    """
    try:
        data = json.loads(tensor.detach().cpu().numpy().tobytes().decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError):
        logger.warning(
            "Could not parse a comfy_quant marker; using default quantized profile"
        )
        return {}
    return data if isinstance(data, dict) else {}


def load_quantized_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
) -> None:
    """Populate ``model`` from a quantized (INT8/FP8) state dict.

    Quantized linear weights (a low-bit ``.weight`` paired with a
    ``.weight_scale``) land on modules that implement ``load_quantized``; every
    other leaf parameter (kept in full precision — ``weight``/``bias``/embedding
    tokens etc.) is replaced with the loaded tensor, cast to ``dtype`` when given
    (quantized kernels emit BF16, so full-precision params must match).

    The per-layer ``.comfy_quant`` JSON marker is decoded and carried into the
    ``QuantizedTensor``, so each module rotates activations with the exact group
    size it was quantized at, and only when the layer was actually ConvRot-
    rotated (``convrot``).

    ``state_dict`` must already have generic wrapper prefixes stripped (see
    ``thenoise.utils.safetensors.load_dit_safetensors``).
    """
    # Scales are collected before weights are processed: a low-bit ``.weight``
    # needs its ``.weight_scale``, and dict order is not guaranteed to place the
    # scale before the weight it belongs to.
    scales: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(_WEIGHT_SCALE_SUFFIX):
            scales[key[: -len(_WEIGHT_SCALE_SUFFIX)]] = tensor

    # ``comfy_quant`` markers record each quantized layer's profile; decode them
    # up front so weights can pick up their marker by path.
    markers: dict[str, dict] = {}
    for key, tensor in state_dict.items():
        if key.endswith(_COMFY_QUANT_SUFFIX):
            markers[key[: -len(_COMFY_QUANT_SUFFIX)]] = _parse_comfy_quant(tensor)

    for key, tensor in state_dict.items():
        if key.endswith(_WEIGHT_SCALE_SUFFIX) or key.endswith(_COMFY_QUANT_SUFFIX):
            continue
        module_path, _, attr = key.rpartition(".")
        module = _submodule(model, module_path, key)
        if attr == "weight" and tensor.dtype in _QUANT_DTYPES:
            _switch_to_quantized(
                module,
                _build_quantized_tensor(tensor, scales.pop(module_path, None), markers.get(module_path, {}), key),
                key,
            )
        elif isinstance(getattr(module, attr, None), torch.nn.Parameter):
            # BF16/full-precision leaf parameter (weight, bias, pad tokens, ...):
            # replace the (meta, init-time) parameter rather than ``param.data =
            # ...``, because set_data rejects meta params and dtype mismatches.
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            setattr(module, attr, torch.nn.Parameter(tensor))
        else:
            raise RuntimeError(f"unexpected key in quantized checkpoint: {key!r}")

    if scales:
        raise RuntimeError(
            f"orphan {_WEIGHT_SCALE_SUFFIX} keys in quantized checkpoint: {list(scales)[:5]}"
        )


def _build_quantized_tensor(
    qweight: torch.Tensor,
    scale,
    marker: dict,
    key: str,
) -> QuantizedTensor:
    """Wrap a stored low-bit ``weight`` + ``scale`` into a ``QuantizedTensor``.

    The layout is chosen from the stored weight dtype (int8 vs FP8 E4M3/E5M2);
    the ``comfy_quant`` marker supplies the profile (ConvRot flag/group size for
    INT8). Quantized kernels emit BF16, so ``orig_dtype`` is set to bf16.
    """
    if scale is None:
        raise RuntimeError(f"quantized weight {key!r} is missing its {_WEIGHT_SCALE_SUFFIX}")
    shape = tuple(qweight.shape)
    if qweight.dtype == torch.int8:
        params = TensorWiseINT8Layout.Params(
            scale=scale,
            orig_dtype=torch.bfloat16,
            orig_shape=shape,
            is_weight=True,
            convrot=bool(marker.get("convrot", True)),
            convrot_groupsize=marker.get("convrot_groupsize", 256),
        )
        return QuantizedTensor(qweight, "TensorWiseINT8Layout", params)
    if qweight.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        params = TensorCoreFP8Layout.Params(
            scale=scale,
            orig_dtype=torch.bfloat16,
            orig_shape=shape,
        )
        return QuantizedTensor(qweight, "TensorCoreFP8Layout", params)
    raise RuntimeError(
        f"unsupported quantized dtype for {key!r}: {qweight.dtype} "
        f"(expected one of {_QUANT_DTYPES})"
    )


def _switch_to_quantized(module: torch.nn.Module, qt: QuantizedTensor, key: str) -> None:
    if not hasattr(module, "load_quantized"):
        raise RuntimeError(
            f"quantized weight {key!r} landed on {type(module).__name__}, "
            "which has no load_quantized(); it must be a QuantizedLinear"
        )
    module.load_quantized(qt)


def _submodule(model: torch.nn.Module, module_path: str, key: str) -> torch.nn.Module:
    """Resolve a checkpoint key's module path, with a clear error on mismatch."""
    try:
        return model.get_submodule(module_path)
    except AttributeError as e:
        raise RuntimeError(
            f"quantized checkpoint key {key!r} does not match the model structure: {e}"
        ) from e


def build_quantized_restore_map(
    path: str,
    key_map: Optional[Callable[[str], str]] = None,
) -> dict[str, str]:
    """Map quantized module paths (post key-map) to their raw checkpoint weight keys.

    Captured once at load time (when wrapper-prefix stripping and ``key_map``
    are already resolved) so a later LoRA undo can reload the original quantized
    weights from disk by raw key — no re-deriving the mapping logic. Only reads
    the safetensors header; no tensors are materialized.

    Returns ``{module_path: raw_weight_key}`` for every quantized linear weight
    in the file (a low-bit ``.weight`` with a sibling ``.weight_scale``).
    """
    restore: dict[str, str] = {}
    with MemoryEfficientSafeOpen(path) as f:
        for raw_key in f.keys():
            if f.header[raw_key]["dtype"] not in ("I8", "F8_E4M3", "F8_E5M2"):
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
            mapped = key_map(stripped) if key_map is not None else stripped
            if not mapped.endswith(".weight"):
                continue
            restore[mapped[: -len(".weight")]] = raw_key
    return restore


def restore_quantized_layer(
    module: torch.nn.Module,
    path: str,
    raw_key: str,
) -> None:
    """Restore a quantized layer's weight from a checkpoint by raw key.

    Reads only the exact tensors the layer needs (the low-bit ``weight`` and its
    ``.weight_scale``) straight from the file by key, avoiding a full reload.
    Rebuilds a ``QuantizedTensor`` in the layer's existing layout profile and
    swaps it in. Used by LoRA undo to return baked LoRAs to the original weights.
    """
    with MemoryEfficientSafeOpen(path) as f:
        qdata = f.get_tensor(
            raw_key,
            device=module.weight.device,
            dtype=module.weight.storage_dtype,
        )
        scale = f.get_tensor(
            raw_key[: -len(".weight")] + _WEIGHT_SCALE_SUFFIX,
            device=module.weight.params.scale.device,
        )
    qt = module.weight._copy_with(
        qdata=qdata,
        params=dataclasses.replace(module.weight.params, scale=scale),
        clone_params=False,
    )
    module.load_quantized(qt)


__all__ = [
    "load_dit",
    "is_quantized_checkpoint",
    "load_quantized_state_dict",
    "build_quantized_restore_map",
    "restore_quantized_layer",
]
