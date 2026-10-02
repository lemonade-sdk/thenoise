"""Central quant-aware DiT loader.

Every DiT adapter loads its weights through ``load_dit``, which selects the
quantization automatically. Checkpoint metadata (``RUNTIME_STATE_SUFFIXES``,
``CHECKPOINT_MARKER_KEYS``) is dropped for every model.
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

from thenoise.utils.checkpoint import CHECKPOINT_MARKER_KEYS
from thenoise.utils.safetensors import (
    MemoryEfficientSafeOpen,
    WRAP_PREFIXES,
    load_dit_safetensors,
    load_safetensors,
    unwrap_key,
)
from thenoise.utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ComfyUI's quantized exporter: per-weight scale, and a U8 JSON marker recording
# the layer's quantization profile.
_WEIGHT_SCALE_SUFFIX = ".weight_scale"
_COMFY_QUANT_SUFFIX = ".comfy_quant"

RUNTIME_STATE_SUFFIXES: tuple[str, ...] = (".comfy_attention.config",)


def drop_runtime_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy without the keys that are not module state."""
    return {k: v for k, v in state_dict.items() if not k.endswith(RUNTIME_STATE_SUFFIXES)}


def drop_checkpoint_markers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy without the checkpoint-level markers."""
    return {k: v for k, v in state_dict.items() if unwrap_key(k) not in CHECKPOINT_MARKER_KEYS}


def _build_int8_qt(qweight: torch.Tensor, scale: torch.Tensor, marker: dict) -> QuantizedTensor:
    """Reconstruct a TensorWiseINT8Layout weight from stored int8 + scale."""
    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qweight.shape),
        is_weight=True,
        convrot=bool(marker.get("convrot", False)),
        convrot_groupsize=marker.get("convrot_groupsize", 256),
    )
    return QuantizedTensor(qweight, "TensorWiseINT8Layout", params)


def _build_fp8_qt(qweight: torch.Tensor, scale: torch.Tensor, marker: dict) -> QuantizedTensor:
    """Reconstruct a TensorCoreFP8Layout weight from stored FP8 + per-tensor scale.

    The same layout covers E4M3 and E5M2; the stored ``qweight`` dtype selects the
    variant.
    """
    params = TensorCoreFP8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qweight.shape),
    )
    return QuantizedTensor(qweight, "TensorCoreFP8Layout", params)


# Storage dtype -> (safetensors header dtype, layout builder).
_QUANT_FORMATS = {torch.int8: ("I8", _build_int8_qt)}
for _name, _header in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    if hasattr(torch, _name):
        _QUANT_FORMATS[getattr(torch, _name)] = (_header, _build_fp8_qt)
_QUANT_DTYPES = tuple(_QUANT_FORMATS)
_QUANT_HEADER_DTYPES = {header for header, _ in _QUANT_FORMATS.values()}


def load_text_encoder_weights(
    model: torch.nn.Module,
    path: str,
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    key_map: Optional[Callable[[str], str]] = None,
    drop_keys: Optional[tuple[str, ...]] = None,
) -> torch.nn.Module:
    """Load a text-encoder checkpoint (single safetensors file), auto-selecting
    quantized vs BF16. No sharded-file support.

    Each low-bit ``.weight`` + ``.weight_scale`` pair goes to its module's
    ``load_quantized``; every other leaf parameter/buffer is cast to ``dtype``.

    The whole ``lm_head.`` subtree is dropped: encoders remove their ``lm_head``
    module before calling, and no LM head is ever run. ``key_map`` normalizes the
    checkpoint's key layout, and ``drop_keys`` drops key PREFIXES on BOTH paths — a
    multimodal file ships an image tower the text-to-image conditioner does not
    build, and some store their tokenizer as a U8 payload tensor; dropping keeps the
    load strict.
    """
    device = torch.device(device)

    sd = load_safetensors(path, device=device, dtype=None)
    sd = {k: v for k, v in sd.items() if k != "lm_head" and not k.startswith("lm_head.")}
    sd = drop_runtime_state(sd)
    sd = drop_checkpoint_markers(sd)
    if drop_keys:
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_keys)}
    if key_map is not None:
        sd = {key_map(k): v for k, v in sd.items()}

    if is_quantized_checkpoint(path):
        load_quantized_state_dict(model, sd, dtype=dtype)
        logger.info("Loaded quantized text encoder from %s", path)
    else:
        if dtype is not None:
            sd = {k: v.to(dtype=dtype) for k, v in sd.items()}
        info = model.load_state_dict(sd, strict=True, assign=True)
        if info.unexpected_keys or info.missing_keys:
            raise RuntimeError(
                f"text encoder checkpoint {path!r} did not match the model: "
                f"missing={info.missing_keys[:10]}, "
                f"unexpected={info.unexpected_keys[:10]}"
            )
        logger.info("Loaded BF16 text encoder from %s", path)

    model.to(device)
    return model


def load_dit(
    model: torch.nn.Module,
    path: str,
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    drop_keys: Optional[tuple[str, ...]] = None,
    expected_missing: tuple[str, ...] = (),
    key_map: Optional[Callable[[str], str]] = None,
    value_map: Optional[Callable[[str, torch.Tensor], tuple[str, torch.Tensor]]] = None,
    state_map: Optional[Callable[[dict], dict]] = None,
) -> torch.nn.Module:
    """Load a DiT checkpoint into ``model`` (constructed on meta), selecting
    quantized vs BF16 automatically.

    ``expected_missing`` lists substrings allowed to be absent from the checkpoint
    (model-internal buffers); with it non-empty the load is non-strict and only
    those are tolerated. ``key_map`` and ``value_map`` transform checkpoint keys and
    values, in that order.

    ``state_map`` is a whole-state-dict fold, for the case where ONE parameter is
    several checkpoint tensors (e.g. stacking ``to_q/to_k/to_v`` into a fused
    ``qkv``). A fold must keep every ``.weight_scale``/``.comfy_quant`` sibling with
    the weight it belongs to. Caveat: it renames module paths while the LoRA-undo
    restore map is keyed on the checkpoint's own names, so folding a QUANTIZED
    checkpoint breaks baked-LoRA undo on the folded layers.
    """
    device = torch.device(device)

    # Load once and prepare before branching so both paths see the same dict.
    sd = load_dit_safetensors(path, device=device, dtype=None)
    sd = drop_runtime_state(sd)
    sd = drop_checkpoint_markers(sd)
    if drop_keys:
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_keys)}
    if key_map is not None:
        sd = {key_map(k): v for k, v in sd.items()}
    if value_map is not None:
        sd = {nk: nv for nk, nv in (value_map(k, v) for k, v in sd.items())}
    if state_map is not None:
        sd = state_map(sd)

    if is_quantized_checkpoint(path):
        # The quantization profile is baked into the weights and scales at export;
        # the ``comfy_quant`` markers reconstruct the ``QuantizedTensor`` layout.
        load_quantized_state_dict(model, sd, dtype=dtype)
        # Raw checkpoint key per quantized layer, for a later LoRA undo.
        model._quantized_restore_map = build_quantized_restore_map(path, key_map)
        logger.info("Loaded quantized checkpoint from %s", path)
    else:
        if dtype is not None:
            sd = {k: v.to(dtype=dtype) for k, v in sd.items()}
        _load_bf16(model, sd, expected_missing, path)

    # Includes buffers (e.g. RoPE) not present in the checkpoint.
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

    Looks for any ``.weight_scale`` key in the safetensors header (after stripping
    generic repackaging wrapper prefixes); reads the header only.
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

    The marker is a small U8 tensor holding the JSON payload, e.g.
    ``{"convrot": true, "convrot_groupsize": 256, "per_row": true}`` for INT8 or
    ``{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}`` for FP8.
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
    other leaf parameter is replaced with the loaded tensor, cast to ``dtype`` when
    given (quantized kernels emit BF16, so full-precision params must match).

    The per-layer ``.comfy_quant`` JSON marker is carried into the
    ``QuantizedTensor`` so each module rotates activations with the exact group size
    it was quantized at, and only when the layer was actually ConvRot-rotated.

    ``state_dict`` must already have generic wrapper prefixes stripped.
    """
    # A low-bit ``.weight`` needs its ``.weight_scale``, and dict order does not
    # guarantee the scale comes first.
    scales: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(_WEIGHT_SCALE_SUFFIX):
            scales[key[: -len(_WEIGHT_SCALE_SUFFIX)]] = tensor

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
                dtype,
            )
        elif isinstance(getattr(module, attr, None), torch.nn.Parameter):
            # ``set_data`` rejects meta params and dtype mismatches.
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            setattr(module, attr, torch.nn.Parameter(tensor))
        elif isinstance(getattr(module, attr, None), torch.Tensor):
            # Non-leaf buffer (e.g. ``rotary_emb.inv_freq``).
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            setattr(module, attr, tensor)
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

    Dispatches on the stored weight dtype to the registered layout builder (see
    ``_QUANT_FORMATS``), which reads the ``comfy_quant`` marker profile. Quantized
    kernels emit BF16, so ``orig_dtype`` is bf16.
    """
    if scale is None:
        raise RuntimeError(f"quantized weight {key!r} is missing its {_WEIGHT_SCALE_SUFFIX}")
    entry = _QUANT_FORMATS.get(qweight.dtype)
    if entry is None:
        raise RuntimeError(
            f"unsupported quantized dtype for {key!r}: {qweight.dtype} "
            f"(expected one of {_QUANT_DTYPES})"
        )
    return entry[1](qweight, scale, marker)


def _switch_to_quantized(
    module: torch.nn.Module, qt: QuantizedTensor, key: str, dtype: Optional[torch.dtype]
) -> None:
    """Put a quantized weight on ``module``: quantized if it can run one, else dequantized.

    A quantized **embedding** is the one other thing a text-encoder export stores; a
    row gather cannot run on a quantized table, so it is dequantized into the compute
    dtype.
    """
    if hasattr(module, "load_quantized"):
        module.load_quantized(qt)
        return
    if isinstance(module, torch.nn.Embedding):
        weight = qt.dequantize()
        module.weight = torch.nn.Parameter(weight if dtype is None else weight.to(dtype))
        return
    raise RuntimeError(
        f"quantized weight {key!r} landed on {type(module).__name__}, "
        "which has no load_quantized(); it must be a QuantizedLinear"
    )


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

    Captured once at load time (when wrapper-prefix stripping and ``key_map`` are
    already resolved) so a later LoRA undo can reload the original quantized weights
    from disk by raw key. Only reads the safetensors header.
    """
    restore: dict[str, str] = {}
    with MemoryEfficientSafeOpen(path) as f:
        for raw_key in f.keys():
            if f.header[raw_key]["dtype"] not in _QUANT_HEADER_DTYPES:
                continue
            if not raw_key.endswith(".weight"):
                continue
            if raw_key[: -len(".weight")] + _WEIGHT_SCALE_SUFFIX not in f.header:
                continue
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

    Reads only the low-bit ``weight`` and its ``.weight_scale`` straight from the
    file, and rebuilds a ``QuantizedTensor`` in the layer's existing layout profile.
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
    "load_text_encoder_weights",
    "is_quantized_checkpoint",
    "load_quantized_state_dict",
    "build_quantized_restore_map",
    "restore_quantized_layer",
    "RUNTIME_STATE_SUFFIXES",
    "drop_runtime_state",
    "drop_checkpoint_markers",
]
