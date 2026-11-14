"""Central quant-aware checkpoint loaders.

``load_dit`` loads a complete checkpoint for standard image models.
``load_mapped_safetensors`` streams oversized models, VAEs and encoders one
tensor at a time while reusing the same quantized layout builders.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Callable, Collection, Optional, Union

import torch
from safetensors import safe_open

from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout

from thenoise.utils.checkpoint import CHECKPOINT_MARKER_KEYS
from thenoise.utils.checkpoint_weights import load_mapped_weights, validated_key_map
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

# ComfyUI's quantized exporter: per-weight scale, a calibrated per-activation scale
# for FP8 layers, and a U8 JSON marker recording the layer's quantization profile.
# Both spellings of each scale appear in the wild.
_WEIGHT_SCALE_SUFFIXES = (".weight_scale", ".scale_weight")
_INPUT_SCALE_SUFFIXES = (".scale_input", ".input_scale")
_WEIGHT_SCALE_SUFFIX = _WEIGHT_SCALE_SUFFIXES[0]
_COMFY_QUANT_SUFFIX = ".comfy_quant"
_SCALED_FP8_MARKER = "scaled_fp8"

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
# FP8 is the only scheme exported with a calibrated activation scale, because its
# range (not its precision) is what the scale buys.
_FP8_DTYPES = tuple(dtype for dtype in _QUANT_FORMATS if dtype is not torch.int8)
_FP8_HEADER_DTYPES = {_QUANT_FORMATS[dtype][0] for dtype in _FP8_DTYPES}


def _strip_suffix(key: str, suffixes: tuple[str, ...]) -> Optional[str]:
    """The part of ``key`` before one of ``suffixes``, or None if it has none."""
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _is_scaled_fp8_marker(key: str) -> bool:
    """True for the checkpoint-level FP8 payload, wherever it is nested."""
    return key == _SCALED_FP8_MARKER or key.endswith("." + _SCALED_FP8_MARKER)


def _scaled_fp8_disables_activation_scale(payload: torch.Tensor) -> bool:
    """ComfyUI's two-value legacy ``scaled_fp8`` payload marks a checkpoint whose
    matmuls were exported to run in full precision."""
    return payload.numel() == 2


def _is_input_scale_valid(scale: torch.Tensor) -> bool:
    """A calibrated activation scale is one finite positive number."""
    flat = scale.detach().to(torch.float32).reshape(-1)
    return flat.numel() == 1 and bool(torch.isfinite(flat).all()) and bool((flat > 0).all())


def _activation_scale(
    scale: Optional[torch.Tensor], marker: dict, full_precision_mm: bool
) -> Optional[torch.Tensor]:
    """The activation scale to install on a layer, or None.

    A full-precision-matmul export disables it checkpoint-wide or per layer, and a
    stored scale then has to be ignored rather than trusted.
    """
    if scale is None or full_precision_mm or marker.get("full_precision_matrix_mult"):
        return None
    return scale


def load_text_encoder_weights(
    model: torch.nn.Module,
    path: str,
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    key_map: Optional[Callable[[str], str]] = None,
    drop_keys: Optional[tuple[str, ...]] = None,
    check_cancel: Optional[Callable[[], None]] = None,
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

    ``check_cancel`` is called per tensor read: an oversized checkpoint can sit in
    the reader long enough that abandoning a queued job matters.
    """
    device = torch.device(device)

    sd = load_safetensors(path, device=device, dtype=None, check_cancel=check_cancel)
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
    check_cancel: Optional[Callable[[], None]] = None,
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
    sd = load_dit_safetensors(path, device=device, dtype=None, check_cancel=check_cancel)
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


def load_mapped_safetensors(
    model: torch.nn.Module,
    path: str | Path,
    key_map: Callable[[str], str],
    *,
    dtype_for: Callable[[str], torch.dtype],
    excluded: Collection[str] = (),
    check_cancel: Callable[[], None] | None = None,
    label: str = "checkpoint",
    quantized: bool = False,
) -> None:
    """Strict, streaming load for huge checkpoints with model-specific key mapping.

    Unlike ``load_safetensors``/``load_dit`` (which create a whole state dict),
    the safetensors reader exposes one tensor at a time, avoiding an additional
    whole-model state dict. The same float loader is shared by DiTs, VAEs
    and text encoders; quantized metadata uses the existing layout builders.
    """
    # Keep CPU checkpoint weights file-backed; avoid copying them into
    # anonymous RAM through the NumPy reader.
    with safe_open(str(path), framework="pt", device="cpu") as source:
        if quantized:
            _load_mapped_quantized(source, model, key_map, dtype_for, excluded,
                                   check_cancel, label)
        else:
            load_mapped_weights(
                source, model, key_map, dtype_for=dtype_for, excluded=excluded,
                check_cancel=check_cancel, label=label,
            )
    logger.info("Loaded %s checkpoint from %s", label, path)


def _load_mapped_quantized(
    source,
    model: torch.nn.Module,
    key_map: Callable[[str], str],
    dtype_for: Callable[[str], torch.dtype],
    excluded: Collection[str],
    check_cancel: Callable[[], None] | None,
    label: str,
) -> None:
    """Stream FP8/INT8 weights with legacy ComfyUI calibration metadata."""
    # Imports remain local because QuantizedLinear depends on this loader.
    from thenoise.dit.quantized import QuantizedLinear

    keys = set(source.keys())
    scales: dict[str, str] = {}
    input_scales: dict[str, str] = {}
    profiles: dict[str, str] = {}
    ignored = set(excluded)
    for raw in keys - ignored:
        stem = _strip_suffix(raw, _WEIGHT_SCALE_SUFFIXES)
        if stem is not None:
            if stem in scales:
                raise ValueError(f"{label} duplicate weight scale for {stem}")
            scales[stem] = raw
            ignored.add(raw)
            continue
        stem = _strip_suffix(raw, _INPUT_SCALE_SUFFIXES)
        if stem is not None:
            if stem in input_scales:
                raise ValueError(f"{label} duplicate input scale for {stem}")
            input_scales[stem] = raw
            ignored.add(raw)
        elif raw.endswith(_COMFY_QUANT_SUFFIX):
            stem = raw[: -len(_COMFY_QUANT_SUFFIX)]
            profiles[stem] = raw
            ignored.add(raw)

    markers = {k for k in keys - ignored
               if _is_scaled_fp8_marker(unwrap_key(k).removeprefix("diffusion_model."))}
    if len(markers) > 1:
        raise ValueError(f"{label} duplicate scaled_fp8 markers")
    ignored.update(markers)
    full_precision_mm = bool(
        markers and _scaled_fp8_disables_activation_scale(source.get_tensor(next(iter(markers))))
    )

    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    mapped = validated_key_map(keys - ignored, key_map, set(expected))
    quantized_weights: set[str] = set()
    for raw, key in mapped.items():
        entry = source.get_slice(raw)
        if tuple(entry.get_shape()) != expected[key]:
            raise ValueError(f"{label} shape mismatch: {raw} -> {key}")
        kind = entry.get_dtype()
        if kind in _QUANT_HEADER_DTYPES:
            if not raw.endswith(".weight") or raw[:-7] not in scales:
                raise ValueError(f"{label} quantized weight without scale: {raw}")
            if not isinstance(model.get_submodule(key[:-7]), QuantizedLinear):
                raise ValueError(f"{label} quantized non-linear unsupported: {raw}")
            quantized_weights.add(raw)
        elif kind not in ("F16", "BF16", "F32"):
            raise ValueError(f"{label} unsupported tensor dtype: {raw}: {kind}")
    for stem in scales:
        if stem + ".weight" not in quantized_weights:
            raise ValueError(f"{label} orphan weight scale: {stem}")
    for stem in (*input_scales, *profiles):
        if stem + ".weight" not in quantized_weights:
            raise ValueError(f"{label} orphan quantized metadata: {stem}")

    calibrated: dict[str, torch.Tensor] = {}
    for stem, raw in input_scales.items():
        weight_raw = stem + ".weight"
        if source.get_slice(weight_raw).get_dtype() not in _FP8_HEADER_DTYPES:
            raise ValueError(f"{label} FP8 input scale on non-FP8 weight: {raw}")
        scale = source.get_tensor(raw)
        if not _is_input_scale_valid(scale):
            raise ValueError(f"{label} invalid FP8 input scale: {raw}")
        calibrated[stem] = scale.to(torch.float32).reshape(())

    for i, (raw, key) in enumerate(sorted(mapped.items())):
        if check_cancel is not None and i % 8 == 0:
            check_cancel()
        tensor = source.get_tensor(raw)
        if raw in quantized_weights:
            stem = raw[:-7]
            marker = (_parse_comfy_quant(source.get_tensor(profiles[stem]))
                      if stem in profiles else {})
            activation_scale = _activation_scale(
                calibrated.get(stem), marker, full_precision_mm
            )
            model.get_submodule(key[:-7]).load_quantized(
                _build_quantized_tensor(tensor, source.get_tensor(scales[stem]), marker, raw),
                input_scale=activation_scale,
            )
        else:
            model.load_state_dict({key: tensor.to(dtype=dtype_for(key))},
                                  strict=False, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError(f"{label} checkpoint contains unloaded meta parameters")


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

    Looks for any weight-scale key in the safetensors header (after stripping
    generic repackaging wrapper prefixes); reads the header only. Both spellings
    count: ComfyUI's quantizer wrote ``.scale_weight`` before it wrote
    ``.weight_scale``, and missing the older one would read a low-bit file as if it
    were BF16.
    """
    with MemoryEfficientSafeOpen(dit_path) as f:
        for key in f.keys():
            for prefix in WRAP_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if _strip_suffix(key, _WEIGHT_SCALE_SUFFIXES) is not None:
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


def _is_quant_metadata(key: str) -> bool:
    """True for the checkpoint keys that describe a quantized layer instead of being one."""
    return (
        _strip_suffix(key, _WEIGHT_SCALE_SUFFIXES) is not None
        or _strip_suffix(key, _INPUT_SCALE_SUFFIXES) is not None
        or key.endswith(_COMFY_QUANT_SUFFIX)
        or _is_scaled_fp8_marker(key)
    )


def _take_input_scale(
    module_path: str, input_scales: dict[str, torch.Tensor], weight: torch.Tensor, key: str
) -> Optional[torch.Tensor]:
    """Claim and check the calibrated activation scale stored for one quantized layer."""
    scale = input_scales.pop(module_path, None)
    if scale is None:
        return None
    if weight.dtype not in _FP8_DTYPES:
        raise RuntimeError(f"FP8 input scale on non-FP8 weight: {key!r}")
    if not _is_input_scale_valid(scale):
        raise RuntimeError(f"invalid FP8 input scale: {key!r}")
    return scale.detach().to(torch.float32).reshape(())


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

    A calibrated FP8 ``.scale_input`` goes to the same ``load_quantized`` call, so
    the whole-state and streaming loaders honour activation scaling identically.

    The per-layer ``.comfy_quant`` JSON marker is carried into the
    ``QuantizedTensor`` so each module rotates activations with the exact group size
    it was quantized at, and only when the layer was actually ConvRot-rotated.

    ``state_dict`` must already have generic wrapper prefixes stripped.
    """
    # A low-bit ``.weight`` needs its scale siblings, and dict order does not
    # guarantee they come first.
    scales: dict[str, torch.Tensor] = {}
    input_scales: dict[str, torch.Tensor] = {}
    markers: dict[str, dict] = {}
    full_precision_mm = False
    for key, tensor in state_dict.items():
        stem = _strip_suffix(key, _WEIGHT_SCALE_SUFFIXES)
        if stem is not None:
            if stem in scales:
                raise RuntimeError(f"duplicate weight scale for {stem!r}")
            scales[stem] = tensor
        elif (stem := _strip_suffix(key, _INPUT_SCALE_SUFFIXES)) is not None:
            input_scales[stem] = tensor
        elif key.endswith(_COMFY_QUANT_SUFFIX):
            markers[key[: -len(_COMFY_QUANT_SUFFIX)]] = _parse_comfy_quant(tensor)
        elif _is_scaled_fp8_marker(key):
            full_precision_mm = _scaled_fp8_disables_activation_scale(tensor)

    for key, tensor in state_dict.items():
        if _is_quant_metadata(key):
            continue
        module_path, _, attr = key.rpartition(".")
        module = _submodule(model, module_path, key)
        if attr == "weight" and tensor.dtype in _QUANT_DTYPES:
            activation_scale = _activation_scale(
                _take_input_scale(module_path, input_scales, tensor, key),
                markers.get(module_path, {}),
                full_precision_mm,
            )
            _switch_to_quantized(
                module,
                _build_quantized_tensor(tensor, scales.pop(module_path, None), markers.get(module_path, {}), key),
                key,
                dtype,
                activation_scale,
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
    if input_scales:
        raise RuntimeError(
            f"orphan {_INPUT_SCALE_SUFFIXES[0]} keys in quantized checkpoint: "
            f"{list(input_scales)[:5]}"
        )
    # A parameter still on meta is a weight the checkpoint never carried: the
    # state-dict walk above only sees what the file has.
    if any(p.is_meta for p in model.parameters()):
        missing = [n for n, p in model.named_parameters() if p.is_meta]
        raise RuntimeError(f"checkpoint leaves {len(missing)} parameters unloaded: {missing[:5]}")


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
    module: torch.nn.Module,
    qt: QuantizedTensor,
    key: str,
    dtype: Optional[torch.dtype],
    input_scale: Optional[torch.Tensor] = None,
) -> None:
    """Put a quantized weight on ``module``: quantized if it can run one, else dequantized.

    A quantized weight goes to any module speaking the loader's
    ``load_quantized(qt, *, input_scale=None)`` protocol.

    A quantized **embedding** is the one other thing a text-encoder export stores; a
    row gather cannot run on a quantized table, so it is dequantized into the compute
    dtype — and has no GEMM left for an activation scale to calibrate.
    """
    if hasattr(module, "load_quantized"):
        module.load_quantized(qt, input_scale=input_scale)
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
    # The activation scale belongs to the checkpoint's export, not to this layer's
    # history: restoring the weight must not silently drop it.
    module.load_quantized(qt, input_scale=getattr(module, "input_scale", None))


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
