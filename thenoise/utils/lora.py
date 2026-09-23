"""LoRA naming, matching, fusing, folding and application to a model.

Everything up to ``apply_lora_to_model`` is pure state-dict algebra: nothing here
knows what kind of layer a LoRA lands on. A module takes one by exposing
``apply_lora(factors) -> LoraMode`` (``thenoise.dit.quantized.QuantizedLinear``,
the only such layer here); a LoRA naming anything else is reported unused.
"""
from __future__ import annotations

import enum
import logging
import re
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import torch

from thenoise.utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class LoraMode(enum.Enum):
    """How a target took a LoRA, i.e. what undo has to revert for it."""

    BAKED = "baked"                      # added to a float weight
    BAKED_QUANTIZED = "baked_quantized"  # requantized into a low-bit weight
    RUNTIME = "runtime"                  # low-rank branch, weight untouched


class LoraFactors(NamedTuple):
    """One target's whole LoRA as a single rank-summed factor pair.

    ``up`` carries every alpha and strength, so the weight delta is exactly
    ``up @ down`` and a runtime branch is ``(x @ down.T) @ up.T``.
    """

    down: torch.Tensor
    up: torch.Tensor

    def delta(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """The full ``[out, in]`` weight delta this pair stands for."""
        down = self.down.to(device=device, dtype=dtype)
        up = self.up.to(device=device, dtype=dtype)
        return up @ down


#: PEFT writes ``<target>.lora_A.<adapter>.weight``, with the adapter name
#: between the factor and the leaf.
_PEFT_FACTOR = re.compile(r"^(?P<base>.+\.lora_[AB])\.[^.]+\.weight$")

_RENAMES = (
    (".lora_down.weight", ".lora_A.weight"),   # sd-scripts
    (".lora_up.weight", ".lora_B.weight"),
    (".lora.down.weight", ".lora_A.weight"),   # ComfyUI
    (".lora.up.weight", ".lora_B.weight"),
)


def _normalize_lora_suffix(lora_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rewrite every factor spelling to the canonical ``lora_A``/``lora_B`` one.

    Alphas and already-canonical keys pass through unchanged.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in lora_sd.items():
        for old, new in _RENAMES:
            if k.endswith(old):
                k = k[: -len(old)] + new
                break
        else:
            if (m := _PEFT_FACTOR.match(k)):
                k = f"{m['base']}.weight"
        out[k] = v
    return out


def _match_lora_keys(
    model_weight_key: str,
    lora_weight_keys: set,
) -> Optional[Tuple[str, str, str]]:
    """The (down, up, alpha) keys of the LoRA targeting a model weight, if any.

    Tries the training tools' conventions in order: sd-scripts underscore-joined
    (with and without the ``lora_unet_`` prefix), then the dotted
    diffusers/ComfyUI forms.
    """
    if not model_weight_key.endswith(".weight"):
        return None

    dotted = model_weight_key[: -len(".weight")]
    underscored = dotted.replace(".", "_")
    for name in (
        f"lora_unet_{underscored}",
        underscored,
        f"diffusion_model.{dotted}",
        f"transformer.{dotted}",
        dotted,
    ):
        a_key = f"{name}.lora_A.weight"
        b_key = f"{name}.lora_B.weight"
        if a_key in lora_weight_keys and b_key in lora_weight_keys:
            return a_key, b_key, f"{name}.alpha"
    return None


#: Fusion specs for the sub-projection stackings used here: fused module name ->
#: the sub-projections a LoRA trains it as, in the fused matrix's row order.
#: Adapters list the ones their modules use in ``DiffusionModel.lora_fusions``.
FUSE_QKV: Dict[str, Tuple[str, ...]] = {"qkv": ("to_q", "to_k", "to_v")}
FUSE_GATE_UP: Dict[str, Tuple[str, ...]] = {"gate_up": ("gate_layer", "proj")}

_FACTOR_A = ".lora_A.weight"
_FACTOR_B = ".lora_B.weight"


def _projection_scale(lora_sd: Dict[str, torch.Tensor], alpha_key: str, rank: int) -> float:
    """ComfyUI's ``alpha / rank``, and exactly ``1.0`` when there is no alpha key."""
    alpha = lora_sd.get(alpha_key)
    return 1.0 if alpha is None else float(alpha) / max(rank, 1)


def _fuse_stacked(
    lora_sd: Dict[str, torch.Tensor],
    parts: Tuple[str, ...],
    fused: str,
) -> Dict[str, torch.Tensor]:
    """Fuse per-sub-projection factors into one stacked-projection pair.

    Several projections come trained separately (``to_q``/``to_k``/``to_v``,
    ``gate_layer``/``proj``) but stored fused, their parts as row blocks:
    ``A_fused = cat([A_part], dim=0)`` and ``B_fused = block_diag(*B_part)``. A
    fused pair carries one rank, so each part's ``alpha / rank`` is folded into its
    ``lora_A`` first (consuming its ``.alpha``), which makes the fused delta exactly
    the stack of the separate merges.

    Parts the LoRA skips get zero rows in ``B_fused``, which needs the parts to
    share one output width; anything that cannot be laid out unambiguously is left
    untouched and logged rather than written to the wrong rows.
    """
    ends = tuple(f"{p}{_FACTOR_A}" for p in parts)
    prefixes = sorted({k[: -len(e)] for k in lora_sd for e in ends if k.endswith(e)})
    if not prefixes:
        return lora_sd

    new_sd = dict(lora_sd)
    for prefix in prefixes:
        factors = {
            p: (new_sd.get(f"{prefix}{p}{_FACTOR_A}"), new_sd.get(f"{prefix}{p}{_FACTOR_B}"))
            for p in parts
        }
        present = [p for p in parts if all(f is not None for f in factors[p])]
        if not present:
            logger.warning(
                "LoRA %s has no complete %s factor pair; skipping %s fusion",
                prefix, "/".join(parts), fused,
            )
            continue

        rows = {p: factors[p][1].size(0) for p in present}
        # A missing part's slice height is only knowable from the parts present,
        # and a fused A only exists if every part shares one input width.
        unlayable = len({factors[p][0].size(1) for p in present}) > 1 or (
            len(present) < len(parts) and len(set(rows.values())) > 1
        )
        if unlayable:
            logger.warning(
                "LoRA %s has mismatched %s factor shapes; skipping %s fusion",
                prefix, "/".join(parts), fused,
            )
            continue

        blocks = []
        for p in parts:
            if p not in present:
                blocks.append((rows[present[0]], None, None))
                continue
            a, b = factors[p]
            alpha_key = f"{prefix}{p}.alpha"
            scale = _projection_scale(new_sd, alpha_key, a.size(0))
            if scale != 1.0:
                a = (a.to(torch.float32) * scale).to(a.dtype)
            new_sd.pop(alpha_key, None)
            new_sd.pop(f"{prefix}{p}{_FACTOR_A}", None)
            new_sd.pop(f"{prefix}{p}{_FACTOR_B}", None)
            blocks.append((rows[p], a, b))

        a_fused = torch.cat([a for _, a, _ in blocks if a is not None], dim=0)
        b_fused = torch.zeros(
            sum(r for r, _, _ in blocks),
            a_fused.size(0),
            dtype=factors[present[0]][1].dtype,
            device=factors[present[0]][1].device,
        )
        row_off = col_off = 0
        for rows_p, a, b in blocks:
            if a is not None:
                b_fused[row_off : row_off + rows_p, col_off : col_off + a.size(0)] = b
                col_off += a.size(0)
            row_off += rows_p

        new_sd[f"{prefix}{fused}{_FACTOR_A}"] = a_fused
        new_sd[f"{prefix}{fused}{_FACTOR_B}"] = b_fused
    return new_sd


def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap ``torch.compile``'s ``OptimizedModule``.

    The compiled kernels reference the same parameter tensors, so mutating the
    original module's weights is visible to them, while ``state_dict`` handling
    on the wrapper is not.
    """
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _normalize_lora_sd(
    lora_sd: Dict[str, torch.Tensor],
    key_map: Optional[Callable[[str], str]],
    fusions: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, torch.Tensor]:
    """Normalize an externally-named LoRA state dict for matching.

    Canonicalize the factor naming, apply the model's ``fusions`` (in declaration
    order), then its ``key_map`` (schema renames).
    """
    lora_sd = _normalize_lora_suffix(lora_sd)
    for fused, parts in (fusions or {}).items():
        lora_sd = _fuse_stacked(lora_sd, tuple(parts), fused)
    if key_map is not None:
        lora_sd = {key_map(k): v for k, v in lora_sd.items()}
    return lora_sd


def _fold(pairs: Sequence[Tuple[torch.Tensor, torch.Tensor, float]]) -> LoraFactors:
    """Fold ``(down, up, scale)`` triples into the one pair summing ``scale * up @ down``.

    The rank axis is the one the product sums over, so concatenating along it is
    the exact sum of any number of LoRAs on a target. A lone unscaled pair is
    passed through untouched.
    """
    if len(pairs) == 1 and pairs[0][2] == 1.0:
        down, up, _ = pairs[0]
        return LoraFactors(down, up)
    down = torch.cat([d.to(torch.float32) for d, _, _ in pairs], dim=0)
    up = torch.cat([u.to(torch.float32) * s for _, u, s in pairs], dim=1)
    return LoraFactors(down, up)


def _match_pairs(
    model_key: str,
    lora_keys: List[set],
    lora_sds: List[Dict[str, torch.Tensor]],
    multipliers: List[float],
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Every LoRA targeting ``model_key``, as ``(down, up, alpha/rank * strength)``.

    Consumes the matched keys so they are not reported as unused, and so one LoRA
    can only land on one target.
    """
    pairs = []
    for keys, lora_sd, multiplier in zip(lora_keys, lora_sds, multipliers):
        match = _match_lora_keys(model_key, keys)
        if match is None:
            continue
        down_key, up_key, alpha_key = match
        down = lora_sd[down_key]
        pairs.append(
            (
                down,
                lora_sd[up_key],
                _projection_scale(lora_sd, alpha_key, down.size(0)) * multiplier,
            )
        )
        keys.difference_update(match)
    return pairs


def _lora_targets(model: torch.nn.Module):
    """Yield ``(path, module)`` for every module that can carry a LoRA itself."""
    for path, module in model.named_modules():
        if hasattr(module, "apply_lora"):
            yield path, module


class _Undo(NamedTuple):
    mode: LoraMode
    factors: Optional[LoraFactors] = None   # BAKED: the delta is recomputed to undo it
    raw_key: Optional[str] = None           # BAKED_QUANTIZED: raw checkpoint key


class LoRAApplyResult(TypedDict):
    """Undo state for ``apply_lora_to_model``: one ``_Undo`` per touched target.

    The rank-reduced factors of the BAKED targets are kept (recomputing their delta
    on undo is far cheaper than caching full-sized ones); quantized targets only
    need the checkpoint key they are reloaded from.
    """

    dit_path: Optional[str]
    targets: Dict[str, _Undo]


def apply_lora_to_model(
    model: torch.nn.Module,
    lora_sds: List[Dict[str, torch.Tensor]],
    multipliers: Optional[List[float]] = None,
    dit_path: Optional[str] = None,
    key_map: Optional[Callable[[str], str]] = None,
    fusions: Optional[Mapping[str, Sequence[str]]] = None,
) -> LoRAApplyResult:
    """Apply LoRAs to a model in-place, returning the state needed to undo them.

    All the LoRAs hitting one target are folded into a single ``LoraFactors``, so a
    layer is handed one LoRA and answers with the ``LoraMode`` it used. ``key_map``
    and ``fusions`` are the model's own naming corrections (see
    ``_normalize_lora_sd``) and ``dit_path`` the checkpoint baked quantized layers
    reload their originals from.
    """
    base_model = _unwrap_compiled(model)
    lora_sds = [_normalize_lora_sd(sd, key_map, fusions) for sd in lora_sds]
    multipliers = list(multipliers or [])[: len(lora_sds)]
    multipliers += [1.0] * (len(lora_sds) - len(multipliers))
    logger.info("Applying LoRA to model. multipliers: %s", multipliers)

    lora_keys = [set(sd.keys()) for sd in lora_sds]
    restore_map = getattr(base_model, "_quantized_restore_map", {})
    targets: Dict[str, _Undo] = {}

    with torch.no_grad():
        for path, module in _lora_targets(base_model):
            pairs = _match_pairs(f"{path}.weight", lora_keys, lora_sds, multipliers)
            if not pairs:
                continue
            factors = _fold(pairs)
            mode = module.apply_lora(factors)
            targets[path] = _Undo(
                mode,
                factors if mode is LoraMode.BAKED else None,
                restore_map.get(path) if mode is LoraMode.BAKED_QUANTIZED else None,
            )

    for i, unused in enumerate(lora_keys):
        if unused:
            logger.warning(
                "LoRA %d has unused keys (not applied): %s",
                i, ", ".join(list(unused)[:10]),
            )

    runtime = [p for p, u in targets.items() if u.mode is LoraMode.RUNTIME]
    baked = sum(u.mode is LoraMode.BAKED_QUANTIZED for u in targets.values())
    if baked or runtime:
        logger.info("Quantized LoRA: %d baked, %d runtime", baked, len(runtime))
        if runtime:
            logger.debug(
                "LoRA finer than the quantization step, applied at runtime: %s",
                ", ".join(runtime),
            )

    return {"dit_path": dit_path, "targets": targets}


def undo_lora_on_model(
    model: torch.nn.Module,
    result: LoRAApplyResult,
) -> None:
    """Undo a previous ``apply_lora_to_model`` (in-place).

    Baked BF16 weights get their recomputed delta subtracted, baked quantized
    layers are reloaded from the checkpoint, runtime branches are dropped.
    """
    targets = result["targets"]
    if not targets:
        return

    logger.debug("Undoing LoRA on model (%d target(s))", len(targets))
    dit_path = result["dit_path"]
    base_model = _unwrap_compiled(model)

    with torch.no_grad():
        for path, undo in targets.items():
            module = base_model.get_submodule(path)
            if undo.mode is LoraMode.RUNTIME:
                module.clear_runtime_lora()
            elif undo.mode is LoraMode.BAKED_QUANTIZED:
                module.undo_lora(dit_path, undo.raw_key)
            elif undo.factors is not None:
                param = base_model.get_parameter(f"{path}.weight")
                param.data.sub_(undo.factors.delta(param.device, param.dtype))
