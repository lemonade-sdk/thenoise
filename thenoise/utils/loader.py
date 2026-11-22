"""Central quant-aware checkpoint loaders.

``load_dit`` and ``load_text_encoder_weights`` take a checkpoint's ``Header``, decide
quantized or not from what it says, and assign the file to a module built on meta.
``build_dit`` adds the other half every DiT shares: build the tree on meta from the
geometry the header gave, then load into it.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import torch

from accelerate import init_empty_weights
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout
from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

from thenoise.utils.checkpoint import CHECKPOINT_MARKER_KEYS
from thenoise.utils.safetensors import (
    Header,
    Signature,
    MemoryEfficientSafeOpen,
    as_header,
    load_safetensors,
    COMFY_QUANT_SUFFIX,
    PACKED_GROUPS_SUFFIX,
    parse_comfy_quant,
    strip_wrap_prefixes,
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
_SCALED_FP8_MARKER = "scaled_fp8"

#: The activation rotation block both quantized exports mean when a layer's marker
#: leaves it out.
_CONVROT_GROUPSIZE = 256

#: The packed ConvRot W4A8/W6A8 export pairs its reduced-width codes with a group scale,
#: a per-row scale and the codebook the codes decode through, under comfy_kitchen's own
#: ``QuantizedTensor.state_dict`` suffixes. The marker's ``format`` is what says a weight
#: is this one: unlike INT8 and FP8, which circulated before the marker was settled, a
#: packed export always states itself.
_W4A8_FORMAT = "asym_w4a8_int8"
_W4A8_FIELDS = ((PACKED_GROUPS_SUFFIX, "scale"), (".weight_s_channel", "s_channel"),
                (".weight_codebook", "codebook"), (".weight_correction", "correction"))
#: The two scales a packed weight cannot be read without. The codebook is absent at
#: 6 bits and the correction only when the export was asymmetric.
_W4A8_REQUIRED = _W4A8_FIELDS[:2]
_W4A8_SUFFIXES = tuple(suffix for suffix, _ in _W4A8_FIELDS)

#: Every checkpoint suffix that fills one layout parameter, first match winning where a
#: scale has two spellings. One table serves the partition (a scale follows its weight),
#: the restore map (a weight is quantized when it has one) and the restore itself.
_QUANT_PARAM_FIELDS = tuple((suffix, "scale") for suffix in _WEIGHT_SCALE_SUFFIXES) + _W4A8_FIELDS

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
        convrot_groupsize=marker.get("convrot_groupsize", _CONVROT_GROUPSIZE),
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


def _build_w4a8_qt(qweight: torch.Tensor, parts: dict, marker: dict, key: str) -> QuantizedTensor:
    """Reconstruct a grouped W4A8/W6A8 weight from its codes and scale tensors.

    The group-scale grid states what the packed row width cannot: the true K. So
    ``orig_shape`` is rebuilt from it, the same way the header derived the shape it
    reports for a packed weight.
    """
    missing = [key[: -len(".weight")] + suffix
               for suffix, field in _W4A8_REQUIRED if field not in parts]
    if missing:
        raise RuntimeError(f"packed quantized weight {key!r} carries no {missing}")
    group_size = marker.get("group_size")
    if not isinstance(group_size, int) or group_size < 1:
        raise RuntimeError(
            f"packed quantized weight {key!r} has no group size in its comfy_quant marker")
    rows, grids = parts["scale"].shape
    params = AsymW4A8Int8Layout.Params(
        scale=parts["scale"],
        s_channel=parts["s_channel"],
        correction=parts.get("correction"),
        codebook=parts.get("codebook"),
        group_size=group_size,
        convrot_groupsize=marker.get("convrot_groupsize", _CONVROT_GROUPSIZE),
        orig_dtype=torch.bfloat16,
        orig_shape=(rows, grids * group_size),
    )
    return QuantizedTensor(qweight, "AsymW4A8Int8Layout", params)


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


def _is_input_scale_placeholder(scale: torch.Tensor) -> bool:
    flat = scale.detach().to(torch.float32).reshape(-1)
    return flat.numel() == 1 and flat.item() == 1.0


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


def _map_keys(state_dict: dict, key_map: Callable[[str], str], path: str) -> dict:
    """Apply ``key_map``, refusing a map under which two keys collide.

    A dict comprehension silently keeps the last of two keys that land on one module
    path, and the strict load afterwards cannot tell that from a checkpoint that never
    had the parameter.
    """
    mapped = {key_map(key): value for key, value in state_dict.items()}
    if len(mapped) != len(state_dict):
        raise RuntimeError(
            f"{path}: {len(state_dict) - len(mapped)} of {len(state_dict)} checkpoint "
            "keys map onto the same module parameter"
        )
    return mapped


def load_module_state(
    model: torch.nn.Module,
    state_dict: dict,
    path: str,
    *,
    what: str = "checkpoint",
    allow_missing: tuple[str, ...] = (),
) -> None:
    """Load ``state_dict`` into ``model``, or name the keys that did not line up.

    ``allow_missing`` lists substrings of the module paths no checkpoint carries
    (model-internal buffers); everything else has to match in both directions, which
    is what ``strict=True`` does not report.
    """
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    missing = [k for k in info.missing_keys
               if not any(allowed in k for allowed in allow_missing)]
    if missing or info.unexpected_keys:
        raise RuntimeError(
            f"{what} checkpoint {path!r} did not match the model: "
            f"missing={missing[:10]}, unexpected={info.unexpected_keys[:10]}"
        )


def _quantized_owner(key: str) -> Optional[str]:
    """The weight a quantization key describes, or None when it describes none."""
    for suffixes in (_WEIGHT_SCALE_SUFFIXES, _INPUT_SCALE_SUFFIXES, _W4A8_SUFFIXES):
        stem = _strip_suffix(key, suffixes)
        if stem is not None:
            return stem + ".weight"
    if key.endswith(COMFY_QUANT_SUFFIX):
        return key[:-len(COMFY_QUANT_SUFFIX)] + ".weight"
    return None


def _is_quant_metadata(key: str) -> bool:
    """True for the checkpoint keys that describe a quantized layer instead of being one."""
    return _quantized_owner(key) is not None or _is_scaled_fp8_marker(key)


def _packed_part(key: str) -> Optional[tuple[str, str]]:
    """``(weight path, layout field)`` for a packed layout's scale tensor, else None."""
    for suffix, field in _W4A8_FIELDS:
        if key.endswith(suffix):
            return key[: -len(suffix)], field
    return None


def partition_keys(
    header, parts: Mapping[str, Mapping[str, tuple[int, ...]]], *,
    part_of: Callable[[str], tuple[Optional[str], Optional[str]]],
    strict: frozenset = frozenset(),
) -> tuple[dict, dict]:
    """Sort one file's tensors into the parts they fill.

    Answers ``(part -> raw key -> module key, part -> raw key -> module key)``: what to
    read, and what the file spelled as a part while having no parameter for it. One walk
    of the header is the point -- a checkpoint that packs several components into one file
    is asked once, however many of them it answers.

    ``parts`` is each part's parameter shapes and ``part_of`` says which part a raw key
    spells, answering ``(None, None)`` for a tensor that fills nothing here.

    ``strict`` names the parts this file was given for. A tensor that spells such a part
    and has no home in it is refused; for a part found by looking it is only reported, so
    a file carrying more than the engine reads still answers the parts it does fit.
    """
    picked: dict = {name: {} for name in parts}
    rejected: dict = {name: {} for name in parts}

    def refuse(name: str, reason: str) -> None:
        if name in strict:
            raise ValueError(reason)

    for raw in header.raw_entries:
        if unwrap_key(raw) in CHECKPOINT_MARKER_KEYS or raw.endswith(RUNTIME_STATE_SUFFIXES):
            continue
        name, target = part_of(raw)
        if name not in parts or target is None:
            continue
        expected = parts[name]
        if _is_scaled_fp8_marker(target):
            picked[name][raw] = target
            continue
        if _is_quant_metadata(target):
            if _quantized_owner(target) not in expected:
                rejected[name][raw] = target
                refuse(name, f"{header.path}: {target!r} is quantization metadata "
                             f"for no {name} weight here")
                continue
        elif target not in expected:
            rejected[name][raw] = target
            refuse(name, f"{header.path}: unexpected {name} tensor {raw!r} -> {target!r}")
            continue
        elif header.logical_shape(raw) != expected[target]:
            rejected[name][raw] = target
            refuse(name, f"{header.path}: {raw!r} shape {header.logical_shape(raw)} "
                         f"!= {expected[target]}")
            continue
        if target in picked[name].values():
            refuse(name, f"{header.path}: duplicate {name} weight {target!r}")
            continue
        picked[name][raw] = target
    return picked, rejected


def load_selected_state(
    model: torch.nn.Module, sources: Sequence[tuple[str, Mapping[str, str]]], *,
    dtype: Optional[torch.dtype], device: Union[str, torch.device], what: str,
) -> None:
    """Fill one module from ``(file, {raw key: module key})`` pairs, reading only those keys.

    For the component that is a slice of a bigger file, where reading the whole file
    would materialize tensors belonging to something else. Every parameter of ``model``
    has to arrive, from one file or several; a parameter filled twice is a collision
    between two files and is refused.

    ``device`` is where the tensors are materialized, so a component the memory manager
    wants resident is read straight onto the device rather than staged through system
    RAM and copied. ``model`` may still be on meta: the tensors arrive and replace it.

    Quantized or plain is decided by the tensors that arrive rather than by the file,
    because a checkpoint that packs a quantized transformer with full-precision codecs is
    mixed, and the file-level answer is wrong for half of it.
    """
    device = torch.device(device)
    weights: dict[str, torch.Tensor] = {}
    origin: dict[str, tuple[str, str]] = {}
    used: list[str] = []
    for path, selected in sources:
        if not selected:
            continue
        data = load_safetensors(path, device, keys=set(selected))
        clash = set(selected.values()) & set(weights)
        if clash:
            raise ValueError(f"{path}: duplicate {what} weight {sorted(clash)[:3]}")
        for raw, tensor in data.items():
            target = selected[raw]
            weights[target] = tensor
            if target.endswith(".weight") and tensor.dtype in _QUANT_DTYPES:
                origin[target[: -len(".weight")]] = (path, raw)
        used.append(path)

    expected = model.state_dict()
    missing = sorted(set(expected) - set(weights))
    if missing:
        raise ValueError(f"{what} weights in {', '.join(used)} leave missing={missing[:8]}"
                         " (no random/meta weight fallback)")
    if any(_is_quant_metadata(key) for key in weights) or any(
            tensor.dtype in _QUANT_DTYPES for tensor in weights.values()):
        load_quantized_state_dict(model, weights, dtype=dtype)
    else:
        load_module_state(model, {k: v.to(dtype=dtype) if v.is_floating_point() else v
                                  for k, v in weights.items()},
                          ", ".join(used), what=what)
    leftover = [key for key, p in model.named_parameters() if p.is_meta]
    leftover += [key for key, b in model.named_buffers() if b.is_meta]
    if leftover:
        raise RuntimeError(f"{what} left uninitialized: {leftover[:8]}")
    # A baked quantized LoRA is undone by reloading this component's originals, and only
    # this route knows the name each weight arrived under in the file it was sliced from.
    model._quantized_restore_map = origin
    model.to(device)


def _load_into(
    model: torch.nn.Module,
    checkpoint: Union[str, Header],
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype],
    what: str,
    key_map: Optional[Callable[[str], str]] = None,
    value_map: Optional[Callable[[str, torch.Tensor], tuple[str, torch.Tensor]]] = None,
    state_map: Optional[Callable[[dict], dict]] = None,
    drop_prefixes: tuple[str, ...] = (),
    strip_wrappers: bool = False,
    allow_missing: tuple[str, ...] = (),
    quantized_restore: bool = False,
) -> torch.nn.Module:
    """Read one safetensors file into a module built on meta.

    The half every component shares: the file is read once, in its own dtype, with
    the runtime state and the checkpoint's own markers dropped before anything is
    compared; ``strip_wrappers`` folds a repackaging's ``model.diffusion_model.`` or
    ``net.`` prefix onto this repo's module names; ``key_map`` then renames what the
    checkpoint spells differently from the tree, ``value_map`` rewrites single
    tensors, and ``state_map`` folds the whole dict, for the case where ONE parameter
    is several checkpoint tensors (e.g. stacking ``to_q/to_k/to_v`` into a fused
    ``qkv``). A fold must keep every ``.weight_scale``/``.comfy_quant`` sibling with
    the weight it belongs to.

    Quantized and plain files take the same route up to that point, then diverge:
    the quantization profile is baked into the weights and scales at export and the
    ``comfy_quant`` markers reconstruct the ``QuantizedTensor`` layout, while a plain
    file is cast to ``dtype`` and loaded strictly.

    ``dtype`` stays the caller's to resolve. Some checkpoints must be read with their
    own dtypes and moved afterwards, because that move is where a model's FP32 islands
    pin themselves and quantized weights stay quantized; casting on the way in would
    round the modulation factors those islands produce.
    """
    device = torch.device(device)
    header = as_header(checkpoint)

    sd = load_safetensors(header.path, device=device, dtype=None)
    if strip_wrappers:
        sd = strip_wrap_prefixes(sd)
    sd = drop_runtime_state(sd)
    sd = drop_checkpoint_markers(sd)
    if drop_prefixes:
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}
    if key_map is not None:
        sd = _map_keys(sd, key_map, header.path)
    if value_map is not None:
        sd = {nk: nv for nk, nv in (value_map(k, v) for k, v in sd.items())}
    if state_map is not None:
        sd = state_map(sd)

    if is_quantized_checkpoint(header):
        load_quantized_state_dict(model, sd, dtype=dtype)
        if quantized_restore:
            # Raw checkpoint key per quantized layer, and the file to read it back
            # from: a LoRA undo reads the component's own, since two experts share
            # every key name.
            model._quantized_restore_map = build_quantized_restore_map(header, key_map)
        logger.info("Loaded quantized %s from %s", what, header.path)
    else:
        if dtype is not None:
            sd = {k: v.to(dtype=dtype) for k, v in sd.items()}
        load_module_state(model, sd, header.path, what=what, allow_missing=allow_missing)
        logger.info("Loaded %s from %s", what, header.path)

    # Includes buffers (e.g. RoPE) not present in the checkpoint.
    model.to(device)
    return model


def load_dit(
    model: torch.nn.Module,
    checkpoint: Union[str, Header],
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    drop_keys: Optional[tuple[str, ...]] = None,
    expected_missing: tuple[str, ...] = (),
    key_map: Optional[Callable[[str], str]] = None,
    value_map: Optional[Callable[[str, torch.Tensor], tuple[str, torch.Tensor]]] = None,
    state_map: Optional[Callable[[dict], dict]] = None,
) -> torch.nn.Module:
    """Load a DiT checkpoint into ``model`` (constructed on meta)."""
    return _load_into(
        model,
        checkpoint,
        device=device,
        dtype=dtype,
        what="DiT",
        key_map=key_map,
        value_map=value_map,
        state_map=state_map,
        drop_prefixes=tuple(drop_keys or ()),
        strip_wrappers=True,
        allow_missing=expected_missing,
        quantized_restore=True,
    )


def spiece_model(checkpoint: Union[str, Path]) -> bytes:
    """The serialized SentencePiece proto a text-encoder file ships beside its weights.

    It is tokenizer payload rather than module state, so the weight load drops the key
    and a tokenizer is built from these bytes instead of a vendored copy of them.
    """
    with MemoryEfficientSafeOpen(str(checkpoint)) as f:
        names = sorted(k for k in f.keys() if k.rsplit(".", 1)[-1] == "spiece_model")
        if not names:
            raise ValueError(f"{checkpoint} carries no SentencePiece tokenizer")
        if f.header[names[0]]["dtype"] != "U8":
            raise ValueError(f"{checkpoint} stores its spiece_model as something else")
        return f.get_tensor(names[0]).numpy().tobytes()


def load_text_encoder_weights(
    model: torch.nn.Module,
    checkpoint: Union[str, Header],
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    key_map: Optional[Callable[[str], str]] = None,
    drop_keys: Optional[tuple[str, ...]] = None,
) -> torch.nn.Module:
    """Load a text-encoder checkpoint (single safetensors file).

    The whole ``lm_head.`` subtree is dropped: encoders remove their ``lm_head``
    module before calling, and no LM head is ever run. ``drop_keys`` drops further key
    PREFIXES — a multimodal file ships an image tower the text-to-image conditioner
    does not build, and some store their tokenizer as a U8 payload tensor; dropping
    keeps the load strict.
    """
    return _load_into(
        model,
        checkpoint,
        device=device,
        dtype=dtype,
        what="text encoder",
        key_map=key_map,
        drop_prefixes=("lm_head", *tuple(drop_keys or ())),
    )


def load_codec(
    vae_path: str,
    device: Union[str, torch.device],
    *,
    what: str,
    construct: Callable[[Header], torch.nn.Module],
    dtype: Optional[torch.dtype] = None,
    signature: Optional["Signature"] = None,
    key_map: Optional[Callable[[str], str]] = None,
    state_map: Optional[Callable[[dict, torch.nn.Module], dict]] = None,
) -> torch.nn.Module:
    """Read one codec file: geometry out of the header, weights out of the same file.

    ``construct`` gets the parsed header, since a codec's widths and depth are what the
    file says they are, and ``signature`` is the claim the catalog makes too.

    ``key_map`` renames the checkpoint's layout onto the module tree, on the header as
    well as the tensors, so geometry and weights are read under one set of names.
    ``state_map`` then rewrites the dict — dropping a half the model does not build,
    restoring weights the export left out — and needs the module for its shapes.

    The module keeps its own :attr:`compute_dtype` unless ``dtype`` overrides it.
    Anything after the load — FP32 islands, fused attention, freezing — is the
    caller's, on the returned module.
    """
    device = torch.device(device)
    header = as_header(vae_path)
    if key_map is not None:
        header = header.relabel(key_map)
    if signature is not None:
        header.require(signature)
    logger.info("Loading %s from %s", what, header.path)

    model = construct(header)
    state_dict = strip_wrap_prefixes(load_safetensors(header.path, device=device, dtype=None))
    if key_map is not None:
        state_dict = _map_keys(state_dict, key_map, header.path)
    if state_map is not None:
        state_dict = state_map(state_dict, model)
    load_module_state(model, state_dict, header.path, what=what)
    model.to(device, dtype if dtype is not None else model.compute_dtype)
    logger.info("Loaded %s from %s", what, header.path)
    return model


def build_dit(
    construct: Callable[[], torch.nn.Module],
    checkpoint: Union[str, Header],
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype],
    **load_kwargs,
) -> torch.nn.Module:
    """Build a DiT on meta, then load ``checkpoint`` into it — the half every model shares.

    The meta tree costs nothing, so the checkpoint's own geometry can size it before a
    single weight is read. ``construct`` is a thunk rather than a class because the
    constructor arguments are the model's business: its own params, a compute dtype, a
    legacy config dict.

    ``dtype`` stays the caller's to resolve, and is passed straight to
    :func:`load_dit`. Some checkpoints must be read with their own dtypes and moved
    afterwards, because that move is where a model's FP32 islands pin themselves and
    quantized weights stay quantized; casting on the way in would round the modulation
    factors those islands produce. Finish such a model after this returns.
    """
    header = as_header(checkpoint)
    with init_empty_weights():
        model = construct()
    return load_dit(model, header, device=device, dtype=dtype, **load_kwargs)


# --------------------------------------------------------------------------- quantized checkpoint helpers

#: A quantized file's remaining full-precision tensors follow the quantized kernels,
#: which emit BF16; whatever dtype the export left them in is not worth trusting.
QUANTIZED_COMPUTE_DTYPE = torch.bfloat16


def is_quantized_checkpoint(checkpoint: Union[str, Header]) -> bool:
    """Return True if ``checkpoint`` stores its weights quantized.

    Any key that describes a quantized weight rather than being one answers here: a
    weight-scale key in either spelling ComfyUI's quantizer used, a packed layout's
    group or row scale, or a per-layer ``comfy_quant`` profile. Missing one would read
    a low-bit file as if it were BF16.
    """
    return any(_is_quant_metadata(name) for name in as_header(checkpoint).names)


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
    """Populate ``model`` from a quantized state dict.

    Quantized linear weights (a low-bit ``.weight`` paired with its scales) land on
    modules that implement ``load_quantized``; every other leaf parameter is replaced
    with the loaded tensor, cast to ``dtype`` when given (quantized kernels emit BF16,
    so full-precision params must match).

    A calibrated FP8 ``.scale_input`` goes to the same ``load_quantized`` call, so
    the whole-state and streaming loaders honour activation scaling identically.

    The per-layer ``.comfy_quant`` JSON marker says which layout a weight is and is
    carried into the ``QuantizedTensor``, so each module rotates activations with the
    exact group size it was quantized at, and only when the layer was actually
    ConvRot-rotated. A packed layout also pairs its codes with a group scale, a row
    scale and a codebook, which travel to the same call.

    ``state_dict`` must already have generic wrapper prefixes stripped.
    """
    # A low-bit ``.weight`` needs its scale siblings, and dict order does not
    # guarantee they come first.
    scales: dict[str, torch.Tensor] = {}
    input_scales: dict[str, torch.Tensor] = {}
    packed: dict[str, dict[str, torch.Tensor]] = {}
    markers: dict[str, dict] = {}
    full_precision_mm = False
    for key, tensor in state_dict.items():
        stem = _strip_suffix(key, _WEIGHT_SCALE_SUFFIXES)
        if stem is not None:
            if stem in scales:
                raise RuntimeError(f"duplicate weight scale for {stem!r}")
            scales[stem] = tensor
        elif (stem := _strip_suffix(key, _INPUT_SCALE_SUFFIXES)) is not None:
            if not _is_input_scale_placeholder(tensor):
                input_scales[stem] = tensor
        elif (part := _packed_part(key)) is not None:
            packed.setdefault(part[0], {})[part[1]] = tensor
        elif key.endswith(COMFY_QUANT_SUFFIX):
            markers[key[: -len(COMFY_QUANT_SUFFIX)]] = parse_comfy_quant(tensor)
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
                _build_quantized_tensor(
                    tensor, scales.pop(module_path, None), markers.get(module_path, {}),
                    key, packed.pop(module_path, None)),
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
    if packed:
        raise RuntimeError(
            f"orphan packed scale keys in quantized checkpoint: {list(packed)[:5]}"
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
    parts: Optional[dict] = None,
) -> QuantizedTensor:
    """Wrap a stored low-bit ``weight`` and its scales into a ``QuantizedTensor``.

    A packed layout states itself in the ``comfy_quant`` marker and carries its scales
    as separate tensors; the INT8 and FP8 exports are read off the stored weight dtype,
    because files circulated that state no format at all (see ``_QUANT_FORMATS``). Both
    builders read the marker profile. Quantized kernels emit BF16, so ``orig_dtype`` is
    bf16.
    """
    fmt = marker.get("format")
    if parts is not None or fmt == _W4A8_FORMAT:
        if fmt != _W4A8_FORMAT:
            raise RuntimeError(
                f"quantized weight {key!r} carries packed scales while its comfy_quant "
                f"marker states format {fmt!r}")
        return _build_w4a8_qt(qweight, parts or {}, marker, key)
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
    checkpoint: Union[str, Header],
    key_map: Optional[Callable[[str], str]] = None,
) -> dict[str, tuple[str, str]]:
    """Map quantized module paths (post key-map) to the file and raw key behind them.

    So a later LoRA undo can reload the original quantized weights from disk by raw
    key: wrapper-prefix stripping and ``key_map`` are already resolved here, but the
    file is spelled under the raw name, which is why the header's own spelling is kept.
    """
    header = as_header(checkpoint)
    restore: dict[str, tuple[str, str]] = {}
    for raw_key, entry in header.raw_entries.items():
        if entry["dtype"] not in _QUANT_HEADER_DTYPES or not raw_key.endswith(".weight"):
            continue
        stem = raw_key[: -len(".weight")]
        if not any(stem + suffix in header.raw_entries
                   for suffix, _ in _QUANT_PARAM_FIELDS):
            continue
        mapped = key_map(unwrap_key(raw_key)) if key_map is not None else unwrap_key(raw_key)
        if not mapped.endswith(".weight"):
            continue
        restore[mapped[: -len(".weight")]] = (str(header.path), raw_key)
    return restore


def restore_quantized_layer(
    module: torch.nn.Module,
    path: str,
    raw_key: str,
) -> None:
    """Restore a quantized layer's weight from a checkpoint by raw key.

    Reads the low-bit ``weight`` and every scale tensor the file pairs with it straight
    from disk, and rebuilds a ``QuantizedTensor`` in the layer's existing layout profile.
    """
    qt = module.weight
    stem = raw_key[: -len(".weight")]
    with MemoryEfficientSafeOpen(path) as f:
        qdata = f.get_tensor(raw_key, device=qt.device, dtype=qt.storage_dtype)
        fields = {}
        for suffix, field in _QUANT_PARAM_FIELDS:
            raw = stem + suffix
            if field in fields or raw not in f.header:
                continue
            stored = getattr(qt.params, field, None)
            fields[field] = f.get_tensor(
                raw, device=stored.device if stored is not None else qt.device)
    if "scale" not in fields:
        raise RuntimeError(f"{path} carries no scale for {raw_key!r}, so it cannot be restored")
    qt = qt._copy_with(
        qdata=qdata,
        params=dataclasses.replace(qt.params, **fields),
        clone_params=False,
    )
    # The activation scale belongs to the checkpoint's export, not to this layer's
    # history: restoring the weight must not silently drop it.
    module.load_quantized(qt, input_scale=getattr(module, "input_scale", None))




__all__ = [
    "load_dit",
    "build_dit",
    "load_codec",
    "load_module_state",
    "load_text_encoder_weights",
    "is_quantized_checkpoint",
    "load_quantized_state_dict",
    "build_quantized_restore_map",
    "restore_quantized_layer",
    "RUNTIME_STATE_SUFFIXES",
    "drop_runtime_state",
    "drop_checkpoint_markers",
]
