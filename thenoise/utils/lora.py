import os
import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union
import torch
import torch.nn.functional as F

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


#: PEFT writes the factors as ``<target>.lora_A.<adapter>.weight``, with the
#: adapter name (``default`` for anything saved by ``save_pretrained``) sitting
#: between the factor and the leaf. Nothing else in the wild puts a segment there.
_PEFT_ADAPTER_FACTOR = re.compile(r"^(?P<base>.+\.lora_[AB])\.[^.]+\.weight$")


def _normalize_lora_suffix(lora_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rewrite LoRA factor spellings to a canonical ``lora_A``/``lora_B`` form.

    Training tools name the factors variously: sd-scripts ``lora_down``/
    ``lora_up``, diffusers ``lora_A``/``lora_B``, ComfyUI ``lora.down``/
    ``lora.up``, PEFT ``lora_A.<adapter>``. Normalizing to ``lora_A`` (down) /
    ``lora_B`` (up) means the rest of the pipeline (fusing and matching) needs to
    know only one form. ``.alpha`` and already-canonical keys are left unchanged.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in lora_sd.items():
        for old, new in [
            (".lora_down.weight", ".lora_A.weight"),
            (".lora_up.weight", ".lora_B.weight"),
            (".lora.down.weight", ".lora_A.weight"),
            (".lora.up.weight", ".lora_B.weight"),
        ]:
            if k.endswith(old):
                k = k[: -len(old)] + new
                break
        else:
            if (m := _PEFT_ADAPTER_FACTOR.match(k)):
                k = f"{m['base']}.weight"
        out[k] = v
    return out


def _match_prefixed_lora_keys(
    lora_name: str,
    lora_weight_keys: set,
) -> Optional[Tuple[str, str, str]]:
    """Find canonical down/up/alpha keys for a LoRA target name."""
    a_key = lora_name + ".lora_A.weight"
    b_key = lora_name + ".lora_B.weight"
    alpha_key = lora_name + ".alpha"
    if a_key in lora_weight_keys and b_key in lora_weight_keys:
        return (a_key, b_key, alpha_key)
    return None


def _match_lora_keys(
    model_weight_key: str,
    lora_weight_keys: set,
) -> Optional[Tuple[str, str, str]]:
    """Find matching LoRA down/up/alpha keys for a model weight key.

    Returns (down_key, up_key, alpha_key) or None if no match. Handles the
    common training-tool naming conventions (sd-scripts underscore-joined,
    diffusers/ComfyUI dotted), with the factor suffix already normalized.
    """
    if not model_weight_key.endswith(".weight"):
        return None

    lora_name_without_prefix = model_weight_key.rsplit(".", 1)[0]

    # sd-scripts naming: underscore-joined path
    for prefix in ["lora_unet_", ""]:
        lora_name = prefix + lora_name_without_prefix.replace(".", "_")
        res = _match_prefixed_lora_keys(lora_name, lora_weight_keys)
        if res:
            return res

    # diffusers/ComfyUI naming: dotted path
    for prefix in ["diffusion_model.", "transformer.", ""]:
        lora_name = prefix + lora_name_without_prefix
        res = _match_prefixed_lora_keys(lora_name, lora_weight_keys)
        if res:
            return res

    return None


#: Fusion specs for the sub-projection stackings this engine's models use: the
#: fused module's name -> the sub-projections a LoRA trains it as, in the fused
#: matrix's row order. Adapters list the ones their modules use in
#: ``DiffusionModel.lora_fusions``; a model with its own naming can as well write
#: an equivalent ``{fused: (part, ...)}`` literal.
FUSE_QKV: Dict[str, Tuple[str, ...]] = {"qkv": ("to_q", "to_k", "to_v")}
FUSE_GATE_UP: Dict[str, Tuple[str, ...]] = {"gate_up": ("gate_layer", "proj")}


def _projection_scale(
    lora_sd: Dict[str, torch.Tensor],
    alpha_key: str,
    rank: int,
) -> float:
    """ComfyUI's per-projection scale ``alpha / rank`` (1.0 when no alpha key).

    Matches ``comfy.weight_adapter.lora.LoRAAdapter.calculate_weight``, which
    uses ``alpha / down.size(0)`` when a ``.alpha`` entry exists and exactly
    ``1.0`` when it does not.
    """
    alpha = lora_sd.get(alpha_key)
    if alpha is None:
        return 1.0
    return float(alpha.item()) / max(rank, 1)


def _fuse_stacked(
    lora_sd: Dict[str, torch.Tensor],
    parts: Tuple[str, ...],
    fused: str,
) -> Dict[str, torch.Tensor]:
    """Fuse per-sub-projection LoRA factors into one stacked-projection pair.

    Several projections come in two layouts: trained separately (attention
    ``to_q``/``to_k``/``to_v``, SwiGLU ``gate_layer``/``proj``) and fused into one
    matrix whose rows are those sub-projections stacked (``qkv``, ``gate_up``). A
    LoRA trained on the split names lands on the fused weight as
    ``A_fused = cat([A_part], dim=0)`` and ``B_fused = block_diag(*B_part)``, each
    part's block at its own row offset in the fused weight.

    A fused pair carries a single rank (the sum over the parts), so one shared
    ``alpha/dim`` cannot reproduce per-projection scales. Each part's
    ``alpha / rank`` is therefore folded into its ``lora_A`` factor before
    concatenating (and its ``.alpha`` key consumed), which makes the fused
    delta exactly the stack of the separate merges — including LoRAs whose alpha
    differs from the rank, or whose parts have different alphas. When no alpha
    key is present the scale is 1 and nothing is folded, so the factors are
    passed through bit-identically.

    LoRAs that train only a subset of the parts are fused with zero rows in
    ``B_fused`` for the missing ones (they contribute no rank), which requires
    the parts to share one output width — the equal-slices layout every fused
    matrix here uses. Anything that cannot be laid out unambiguously (incomplete
    factor pairs, or missing parts alongside differing output widths) is left
    untouched and logged instead of being written to the wrong rows.

    No-op if the LoRA has none of ``parts``. The input is not mutated.
    """
    names = "|".join(re.escape(p) for p in parts)
    groups = set()
    for k in lora_sd:
        m = re.match(rf"^(.*?)({names})\.lora_[AB]\.weight$", k)
        if m:
            groups.add(m.group(1))
    if not groups:
        return lora_sd

    new_sd = dict(lora_sd)
    for prefix in sorted(groups):
        factors = {
            p: (
                new_sd.get(f"{prefix}{p}.lora_A.weight"),
                new_sd.get(f"{prefix}{p}.lora_B.weight"),
            )
            for p in parts
        }
        present = [
            p
            for p in parts
            if factors[p][0] is not None and factors[p][1] is not None
        ]
        if not present:
            logger.warning(
                "LoRA %s has no complete %s factor pair; skipping %s fusion",
                prefix,
                "/".join(parts),
                fused,
            )
            continue

        out_rows = {p: factors[p][1].size(0) for p in present}
        in_dims = {p: factors[p][0].size(1) for p in present}
        missing = [p for p in parts if p not in present]
        in_ok = len(set(in_dims.values())) == 1
        out_ok = not missing or len(set(out_rows.values())) == 1
        if not (in_ok and out_ok):
            logger.warning(
                "LoRA %s has mismatched %s factor shapes (%s present, %s differ); "
                "skipping %s fusion",
                prefix,
                "/".join(parts),
                "/".join(present),
                "input" if not in_ok else "output",
                fused,
            )
            continue

        slice_rows = out_rows[present[0]]
        rows = {p: out_rows.get(p, slice_rows) for p in parts}

        blocks = []
        rank_total = 0
        for p in parts:
            if p not in present:
                blocks.append((rows[p], 0, None, None))
                continue
            a, b = factors[p]
            alpha_key = f"{prefix}{p}.alpha"
            scale = _projection_scale(new_sd, alpha_key, a.size(0))
            if scale != 1.0:
                a = (a.to(torch.float32) * scale).to(a.dtype)
            new_sd.pop(alpha_key, None)
            blocks.append((rows[p], a.size(0), a, b))
            rank_total += a.size(0)
            new_sd.pop(f"{prefix}{p}.lora_A.weight", None)
            new_sd.pop(f"{prefix}{p}.lora_B.weight", None)

        a_dtype, a_device = factors[present[0]][0].dtype, factors[present[0]][0].device
        b_dtype, b_device = factors[present[0]][1].dtype, factors[present[0]][1].device
        a_fused = torch.zeros(
            rank_total, in_dims[present[0]], dtype=a_dtype, device=a_device
        )
        b_fused = torch.zeros(
            sum(rows.values()), rank_total, dtype=b_dtype, device=b_device
        )

        row_off = col_off = 0
        for rows_p, rank_p, a, b in blocks:
            if a is not None:
                a_fused[col_off : col_off + rank_p] = a
                b_fused[row_off : row_off + rows_p, col_off : col_off + rank_p] = b
                col_off += rank_p
            row_off += rows_p

        new_sd[f"{prefix}{fused}.lora_A.weight"] = a_fused
        new_sd[f"{prefix}{fused}.lora_B.weight"] = b_fused
    return new_sd


def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a torch.compile OptimizedModule to get the original module.

    torch.compile wraps the model in an OptimizedModule whose state_dict()/load_state_dict()
    may not delegate correctly. Operating on the original module ensures LoRA key matching 
    and weight modification work correctly. The compiled kernels reference the same underlying 
    parameter tensors, so they see updates.
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

    Pipeline: normalize the factor naming, apply the model's ``fusions`` (each
    ``{fused: parts}`` entry stacking one set of separately-trained factors onto
    the fused module, in declaration order), then the model family's ``key_map``
    (schema renames, e.g. ComfyUI ``transformer_blocks`` -> ``double_blocks``).
    """
    lora_sd = _normalize_lora_suffix(lora_sd)
    for fused, parts in (fusions or {}).items():
        lora_sd = _fuse_stacked(lora_sd, tuple(parts), fused)
    if key_map is not None:
        lora_sd = {key_map(k): v for k, v in lora_sd.items()}
    return lora_sd



def compute_lora_delta(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    alpha,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Compute a LoRA delta ``[out, in]`` for a linear or conv weight.

    ``multiplier * (up @ down) * (alpha/r)``, computed in BF16 on
    ``calc_device``. The branch (linear vs conv 1x1 vs conv 3x3) is inferred
    from the shape of ``down_weight``.
    """
    r = down_weight.size(0)
    if isinstance(alpha, torch.Tensor):
        scale = float(alpha.to(calc_device)) / r * multiplier
    else:
        scale = alpha / r * multiplier

    down_weight = down_weight.to(device=calc_device, dtype=torch.bfloat16)
    up_weight = up_weight.to(device=calc_device, dtype=torch.bfloat16)

    if down_weight.ndim == 2:
        # linear (LoRA factors may be stored 4D, e.g. diffusers conv-style)
        if up_weight.ndim == 4:
            up_weight = up_weight.squeeze(3).squeeze(2)
            down_weight = down_weight.squeeze(3).squeeze(2)
        delta = up_weight @ down_weight
    elif down_weight.size(2, 3) == (1, 1):
        # conv2d 1x1
        delta = (
            up_weight.squeeze(3).squeeze(2)
            @ down_weight.squeeze(3).squeeze(2)
        ).unsqueeze(2).unsqueeze(3)
    else:
        # conv2d 3x3
        delta = F.conv2d(
            down_weight.permute(1, 0, 2, 3), up_weight
        ).permute(1, 0, 2, 3)

    return delta * scale


class LoRAApplyResult(TypedDict):
    """Result from ``apply_lora_to_model``: cached state for undo.

    Keeps the small rank-reduced LoRA factors in memory instead of full-sized
    delta tensors. On undo the BF16 deltas are recomputed from these factors.
    ``affected_keys`` tracks exactly which BF16 model parameters were modified,
    so undo can skip the unaffected ones without iterating the full state dict.

    For quantized ``QuantizedLinear`` layers the LoRA is baked into the
    quantized weights (see ``QuantizedLinear.bake_lora``), so undo reloads the
    original weights from disk: ``quantized_affected`` lists their module paths
    (parallel to ``quantized_restore_keys``, the raw checkpoint keys), and
    ``dit_path`` is the checkpoint file to read them back from.
    """

    lora_sds: List[Dict[str, torch.Tensor]]
    multipliers: List[float]
    affected_keys: Tuple[str, ...]
    quantized_affected: Tuple[str, ...]
    quantized_restore_keys: Tuple[str, ...]
    dit_path: Optional[str]


def apply_lora_to_model(
    model: torch.nn.Module,
    lora_sds: List[Dict[str, torch.Tensor]],
    multipliers: List[float],
    calc_device: torch.device,
    dit_path: Optional[str] = None,
    key_map: Optional[Callable[[str], str]] = None,
    fusions: Optional[Mapping[str, Sequence[str]]] = None,
) -> LoRAApplyResult:
    """Apply LoRA weights directly to a model's parameters (in-place).

    Returns a ``LoRAApplyResult`` holding the LoRA state dicts and multipliers,
    which can be passed to ``undo_lora_on_model`` to restore the original weights.
    The LoRA state dicts are small (rank-reduced factors) compared to the full
    model weights, so keeping them in memory is cheap.

    Param keys use the same naming as ``model.state_dict()`` (e.g. "blocks.0.attn.gate.weight").

    ``fusions`` is the model's fusion spec (``{fused: parts}``, e.g.
    ``FUSE_QKV``): the sub-projection stackings a LoRA trained on the separate
    names has to be fused onto before it can match this model's weights.
    """
    if not lora_sds:
        return {
            "lora_sds": [],
            "multipliers": [],
            "affected_keys": (),
            "quantized_affected": (),
            "quantized_restore_keys": (),
            "dit_path": dit_path,
        }

    if multipliers is None:
        multipliers = [1.0] * len(lora_sds)
    while len(multipliers) < len(lora_sds):
        multipliers.append(1.0)
    multipliers = multipliers[: len(lora_sds)]

    logger.info("Applying LoRA to model. multipliers: %s", multipliers)

    base_model = _unwrap_compiled(model)

    # Normalize each LoRA state dict to the model's naming: normalize the factor
    # naming, apply the model's fusions, then its ``key_map`` (schema renames,
    # e.g. ComfyUI -> repo).
    lora_sds = [_normalize_lora_sd(sd, key_map, fusions) for sd in lora_sds]

    # Build key sets for each LoRA
    lora_weight_keys_list = [set(sd.keys()) for sd in lora_sds]

    # Accumulate LoRA deltas per target module across all LoRAs (multiple
    # LoRAs can affect the same layer). ``deltas`` maps module path -> delta.
    deltas: Dict[str, torch.Tensor] = {}
    affected_bf16: List[str] = []
    affected_quantized: List[str] = []

    def _accumulate(
        module_path: str,
        delta: torch.Tensor,
        *,
        quantized: bool,
    ) -> None:
        if module_path in deltas:
            deltas[module_path] = deltas[module_path] + delta
        else:
            deltas[module_path] = delta
        (affected_quantized if quantized else affected_bf16).append(module_path)

    # BF16 path: ``.weight`` parameters (plain linears, convs, norms).
    for model_key, model_weight in base_model.named_parameters():
        if not model_key.endswith(".weight"):
            continue
        module_path = model_key.rsplit(".", 1)[0]
        module = base_model.get_submodule(module_path)
        if isinstance(module, QuantizedLinear) and module._quantized:
            continue  # quantized weight is a buffer, handled below

        for lora_weight_keys, lora_sd, multiplier in zip(
            lora_weight_keys_list, lora_sds, multipliers
        ):
            match = _match_lora_keys(model_key, lora_weight_keys)
            if match is None:
                continue

            down_key, up_key, alpha_key = match
            delta = compute_lora_delta(
                lora_sd[down_key],
                lora_sd[up_key],
                lora_sd.get(alpha_key, lora_sd[down_key].size(0)),
                multiplier,
                calc_device,
            )
            _accumulate(module_path, delta, quantized=False)

            # Remove consumed keys
            lora_weight_keys.discard(down_key)
            lora_weight_keys.discard(up_key)
            lora_weight_keys.discard(alpha_key)

    # Quantized path: quantized ``QuantizedLinear`` layers have no bf16
    # ``.weight`` parameter to mutate; their LoRA is baked into the quantized
    # weights at switch time (dequantize -> add delta -> requantize), so the
    # runtime forward is a single quantized GEMM with zero LoRA cost. Deltas
    # from multiple LoRAs are accumulated per module and baked once, avoiding
    # repeated lossy requantization. This must run before the unused-key
    # warning so the consumed keys are not reported.
    for module_path, module in base_model.named_modules():
        if not isinstance(module, QuantizedLinear) or not module._quantized:
            continue
        model_key = f"{module_path}.weight"
        for lora_weight_keys, lora_sd, multiplier in zip(
            lora_weight_keys_list, lora_sds, multipliers
        ):
            match = _match_lora_keys(model_key, lora_weight_keys)
            if match is None:
                continue
            down_key, up_key, alpha_key = match
            delta = compute_lora_delta(
                lora_sd[down_key],
                lora_sd[up_key],
                lora_sd.get(alpha_key, lora_sd[down_key].size(0)),
                multiplier,
                calc_device,
            )
            _accumulate(module_path, delta, quantized=True)
            lora_weight_keys.discard(down_key)
            lora_weight_keys.discard(up_key)
            lora_weight_keys.discard(alpha_key)

    # Warn about unused LoRA keys
    for i, lora_weight_keys in enumerate(lora_weight_keys_list):
        if len(lora_weight_keys) > 0:
            logger.warning("LoRA %d has unused keys: %s", i, ", ".join(list(lora_weight_keys)[:10]))

    # Apply the accumulated delta to each target layer (in-place, no state_dict
    # copy). Each layer owns how to mutate itself: BF16 adds the delta, quantized
    # bakes it in.
    with torch.no_grad():
        for module_path, delta in deltas.items():
            module = base_model.get_submodule(module_path)
            if isinstance(module, QuantizedLinear):
                module.apply_lora(delta)
            else:
                # Non-QuantizedLinear weight (e.g. a conv): plain in-place add.
                param = base_model.get_parameter(f"{module_path}.weight")
                param.data.add_(delta.to(param.device, param.dtype))

    # For baked quantized LoRAs, record the raw checkpoint keys so undo can
    # reload the original weights from disk (captured at load time in the model).
    quantized_affected_unique = tuple(dict.fromkeys(affected_quantized))
    restore_map = getattr(base_model, "_quantized_restore_map", {})
    quantized_restore_keys = tuple(restore_map.get(p) for p in quantized_affected_unique)

    return {
        "lora_sds": lora_sds,
        "multipliers": multipliers,
        "affected_keys": tuple(f"{p}.weight" for p in dict.fromkeys(affected_bf16)),
        "quantized_affected": quantized_affected_unique,
        "quantized_restore_keys": quantized_restore_keys,
        "dit_path": dit_path,
    }


def undo_lora_on_model(
    model: torch.nn.Module,
    result: LoRAApplyResult,
    calc_device: torch.device,
) -> None:
    """Undo a previous LoRA application by recomputing and subtracting deltas.

    Restores the model's parameters to their pre-LoRA state (in-place).
    Deltas are recomputed from the cached LoRA state dicts, so no full-sized
    delta tensors need to be kept in memory.
    Only the affected parameters are touched — no full state_dict copy.
    """
    lora_sds = result["lora_sds"]
    multipliers = result["multipliers"]
    affected_keys = result.get("affected_keys")
    quantized_affected = result.get("quantized_affected")
    quantized_restore_keys = result.get("quantized_restore_keys")
    dit_path = result.get("dit_path")
    if not lora_sds and not quantized_affected:
        return

    base_model = _unwrap_compiled(model)

    # Baked quantized LoRAs: reload the original weights from the checkpoint
    # file (by the raw keys captured at load time) and restore them in place.
    for module_path, raw_key in zip(quantized_affected or (), quantized_restore_keys or ()):
        base_model.get_submodule(module_path).undo_lora(
            None, raw_key=raw_key, dit_path=dit_path
        )

    if not lora_sds or not affected_keys:
        return

    logger.debug("Undoing LoRA on model (%d LoRA(s), %d keys)", len(lora_sds), len(affected_keys))

    # Build key sets for each LoRA (copy so we can mutate)
    lora_weight_keys_list = [set(sd.keys()) for sd in lora_sds]

    with torch.no_grad():
        for model_key in affected_keys:
            module_path = model_key.rsplit(".", 1)[0]
            module = base_model.get_submodule(module_path)

            accumulated_delta: Optional[torch.Tensor] = None

            for lora_weight_keys, lora_sd, multiplier in zip(
                lora_weight_keys_list, lora_sds, multipliers
            ):
                match = _match_lora_keys(model_key, lora_weight_keys)
                if match is None:
                    continue

                down_key, up_key, alpha_key = match
                delta = compute_lora_delta(
                    lora_sd[down_key],
                    lora_sd[up_key],
                    lora_sd.get(alpha_key, lora_sd[down_key].size(0)),
                    multiplier,
                    calc_device,
                )

                if accumulated_delta is None:
                    accumulated_delta = delta
                else:
                    accumulated_delta = accumulated_delta + delta.to(
                        accumulated_delta.device, accumulated_delta.dtype
                    )

            if accumulated_delta is not None:
                if isinstance(module, QuantizedLinear):
                    module.undo_lora(accumulated_delta)
                else:
                    param = base_model.get_parameter(model_key)
                    param.data.sub_(accumulated_delta.to(param.device, param.dtype))

