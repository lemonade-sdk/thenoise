import os
import re
from typing import Dict, List, Optional, Tuple, TypedDict, Union
import torch

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.loader import restore_quantized_layer
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _match_lora_keys(
    model_weight_key: str,
    lora_weight_keys: set,
) -> Optional[Tuple[str, str, str]]:
    """Find matching LoRA down/up/alpha keys for a model weight key.

    Returns (down_key, up_key, alpha_key) or None if no match.
    """
    if not model_weight_key.endswith(".weight"):
        return None

    lora_name_without_prefix = model_weight_key.rsplit(".", 1)[0]

    # sd-scripts naming: underscore-joined path, lora_down/lora_up.
    for prefix in ["lora_unet_", ""]:
        lora_name = prefix + lora_name_without_prefix.replace(".", "_")
        down_key = lora_name + ".lora_down.weight"
        up_key = lora_name + ".lora_up.weight"
        alpha_key = lora_name + ".alpha"
        if down_key in lora_weight_keys and up_key in lora_weight_keys:
            return (down_key, up_key, alpha_key)

    # diffusers-style naming: dotted path, lora_A/lora_B.
    for prefix in ["diffusion_model.", ""]:
        lora_name = prefix + lora_name_without_prefix
        a_key = lora_name + ".lora_A.weight"
        b_key = lora_name + ".lora_B.weight"
        alpha_key = lora_name + ".alpha"
        if a_key in lora_weight_keys and b_key in lora_weight_keys:
            return (a_key, b_key, alpha_key)

    return None


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


def _convert_fused_attention_lora(
    lora_sd: Dict[str, torch.Tensor],
    use_fused: bool,
) -> Dict[str, torch.Tensor]:
    """Convert a diffusers-layout S3-DiT attention LoRA to the fused-qkv layout.

    Diffusers' ``ZImageTransformer2DModel`` uses separate ``to_q``/``to_k``/
    ``to_v``/``to_out`` projections, while thenoise's Z-Image port (ComfyUI /
    Lumina layout) fuses QKV into a single ``qkv`` projection plus ``out``.
    LoRAs trained on the diffusers layout therefore use keys that do not match
    the model's parameters (``to_q``/``to_k``/``to_v`` -> ``qkv``, ``to_out.0``
    -> ``out``). This helper remaps those factors so they apply correctly.

    The fused q/k/v factors are combined as:
      * A_qkv = concat([A_q; A_k; A_v], dim=0)              -> [3r, dim]
      * B_qkv = block_diag(B_q, B_k, B_v)                   -> [3*dim, 3r]
    so that ``B_qkv @ A_qkv`` reproduces the per-projection block-diagonal
    delta (q rows, then k, then v) that the model's fused ``qkv`` expects.
    Because the fused rank is ``3r``, ``_compute_lora_delta``'s default scale
    ``alpha/dim`` with ``alpha = down.size(0) = 3r`` evaluates to 1, matching
    each original projection's ``r/r = 1`` scaling.

    ``feed_forward`` (w1/w2/w3) already shares identical naming and is left
    untouched, as is everything else. If the LoRA isn't in the diffusers
    attention layout, it is returned unchanged. The input is not mutated.
    """
    if not use_fused:
        return lora_sd
    keys = list(lora_sd.keys())
    if not any("attention.to_q.lora_A.weight" in k for k in keys):
        return lora_sd

    # Group the to_q/to_k/to_v factors by their shared attention prefix
    # (e.g. "diffusion_model.layers.0.attention.").
    groups = set()
    for k in keys:
        m = re.match(r"^(.*?\.attention\.)to_[qkv]\.lora_[AB]\.weight$", k)
        if m:
            groups.add(m.group(1))
    if not groups:
        return lora_sd

    new_sd = dict(lora_sd)
    for prefix in groups:
        # Fuse q/k/v factors into a single qkv projection (rows q, k, v).
        for side in ("A", "B"):
            parts = [
                new_sd.pop(f"{prefix}to_{p}.lora_{side}.weight")
                for p in ("q", "k", "v")
            ]
            if side == "A":
                new_sd[f"{prefix}qkv.lora_A.weight"] = torch.cat(parts, dim=0)
            else:
                new_sd[f"{prefix}qkv.lora_B.weight"] = torch.block_diag(*parts)
        # to_out.0 -> out (single projection, rename only).
        out_a = f"{prefix}to_out.0.lora_A.weight"
        out_b = f"{prefix}to_out.0.lora_B.weight"
        if out_a in new_sd:
            new_sd[f"{prefix}out.lora_A.weight"] = new_sd.pop(out_a)
        if out_b in new_sd:
            new_sd[f"{prefix}out.lora_B.weight"] = new_sd.pop(out_b)
    return new_sd


def _compute_lora_delta(
    model_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    alpha,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Compute the LoRA delta for a single weight: multiplier * (up @ down) * scale.

    Returns a tensor of the same shape as model_weight.
    """
    dim = down_weight.size()[0]
    if isinstance(alpha, torch.Tensor):
        scale = float(alpha.to(calc_device)) / dim
    else:
        scale = alpha / dim

    down_weight = down_weight.to(calc_device)
    up_weight = up_weight.to(calc_device)

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:  # fp8
        down_weight = down_weight.to(torch.float16)
        up_weight = up_weight.to(torch.float16)

    if len(model_weight.size()) == 2:
        # linear
        if len(up_weight.size()) == 4:
            up_weight = up_weight.squeeze(3).squeeze(2)
            down_weight = down_weight.squeeze(3).squeeze(2)
        delta = multiplier * (up_weight @ down_weight) * scale
    elif down_weight.size()[2:4] == (1, 1):
        # conv2d 1x1
        delta = (
            multiplier
            * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
            .unsqueeze(2).unsqueeze(3)
            * scale
        )
    else:
        # conv2d 3x3
        conved = torch.nn.functional.conv2d(
            down_weight.permute(1, 0, 2, 3), up_weight
        ).permute(1, 0, 2, 3)
        delta = multiplier * conved * scale

    if original_dtype.itemsize == 1:  # fp8
        delta = delta.to(original_dtype)

    return delta


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


def _lora_delta_2d(down, up, alpha, multiplier, calc_device) -> torch.Tensor:
    """Compute the BF16 LoRA delta ``[out, in]`` for a 2D linear weight.

    ``multiplier * (up @ down) * (alpha/r)``, matching ``_compute_lora_delta``'s
    linear branch (INT8 layers are always 2D linears with no bf16 weight to
    reference, so this avoids the fp8/conv branches there).
    """
    r = down.size(0)
    if isinstance(alpha, torch.Tensor):
        scale = float(alpha.to(calc_device)) / r * multiplier
    else:
        scale = alpha / r * multiplier
    down = down.to(device=calc_device, dtype=torch.bfloat16)
    up = up.to(device=calc_device, dtype=torch.bfloat16)
    return up @ down * scale


def apply_lora_to_model(
    model: torch.nn.Module,
    lora_sds: List[Dict[str, torch.Tensor]],
    multipliers: List[float],
    calc_device: torch.device,
    dit_path: Optional[str] = None,
) -> LoRAApplyResult:
    """Apply LoRA weights directly to a model's parameters (in-place).

    Returns a ``LoRAApplyResult`` holding the LoRA state dicts and multipliers,
    which can be passed to ``undo_lora_on_model`` to restore the original weights.
    The LoRA state dicts are small (rank-reduced factors) compared to the full
    model weights, so keeping them in memory is cheap.

    Param keys use the same naming as ``model.state_dict()`` (e.g. "blocks.0.attn.gate.weight").
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

    # Detect whether the model uses a fused qkv attention (S3-DiT ComfyUI
    # layout). If so, convert any diffusers-layout (separate to_q/to_k/to_v)
    # LoRA factors to match the model's parameters.
    model_weight_keys = [
        k for k, _ in base_model.named_parameters() if k.endswith(".weight")
    ]
    use_fused = any(k.endswith(".attention.qkv.weight") for k in model_weight_keys)
    lora_sds = [_convert_fused_attention_lora(sd, use_fused) for sd in lora_sds]

    # Build key sets for each LoRA
    lora_weight_keys_list = [set(sd.keys()) for sd in lora_sds]

    # Iterate over named parameters directly (no full state_dict copy)
    undo_deltas: Dict[str, torch.Tensor] = {}

    for model_key, model_weight in base_model.named_parameters():
        if not model_key.endswith(".weight"):
            continue

        original_device = model_weight.device
        weight_on_calc = model_weight if original_device == calc_device else model_weight.to(calc_device)

        for lora_weight_keys, lora_sd, multiplier in zip(lora_weight_keys_list, lora_sds, multipliers):
            match = _match_lora_keys(model_key, lora_weight_keys)
            if match is None:
                continue

            down_key, up_key, alpha_key = match
            down_weight = lora_sd[down_key]
            up_weight = lora_sd[up_key]
            alpha = lora_sd.get(alpha_key, down_weight.size()[0])

            delta = _compute_lora_delta(weight_on_calc, down_weight, up_weight, alpha, multiplier, calc_device)

            # Accumulate delta (multiple LoRAs can affect the same layer)
            if model_key in undo_deltas:
                undo_deltas[model_key] = undo_deltas[model_key] + delta.to(undo_deltas[model_key].device, undo_deltas[model_key].dtype)
            else:
                undo_deltas[model_key] = delta

            # Remove consumed keys
            lora_weight_keys.remove(down_key)
            lora_weight_keys.remove(up_key)
            if alpha_key in lora_weight_keys:
                lora_weight_keys.remove(alpha_key)

    # Quantized QuantizedLinear layers have no bf16 ``.weight`` parameter to
    # mutate; their LoRA is baked into the quantized weights at switch time
    # (dequantize -> add delta -> requantize), so the runtime forward is a single
    # quantized GEMM (for INT8) with zero LoRA cost. Deltas from multiple LoRAs
    # are accumulated per module and baked once, avoiding repeated lossy
    # requantization. This must run before the unused-key warning so the consumed
    # keys are not reported.
    quantized_deltas: Dict[str, torch.Tensor] = {}
    quantized_affected: List[str] = []
    for module_path, module in base_model.named_modules():
        if not isinstance(module, QuantizedLinear) or not module._quantized:
            continue
        model_key = f"{module_path}.weight"
        for lora_weight_keys, lora_sd, multiplier in zip(lora_weight_keys_list, lora_sds, multipliers):
            match = _match_lora_keys(model_key, lora_weight_keys)
            if match is None:
                continue
            down_key, up_key, alpha_key = match
            delta = _lora_delta_2d(
                lora_sd[down_key],
                lora_sd[up_key],
                lora_sd.get(alpha_key, lora_sd[down_key].size(0)),
                multiplier,
                calc_device,
            )
            if module_path in quantized_deltas:
                quantized_deltas[module_path] = quantized_deltas[module_path] + delta
            else:
                quantized_deltas[module_path] = delta
            lora_weight_keys.discard(down_key)
            lora_weight_keys.discard(up_key)
            lora_weight_keys.discard(alpha_key)
            quantized_affected.append(module_path)

    for module_path, delta in quantized_deltas.items():
        base_model.get_submodule(module_path).bake_lora(delta)

    # Warn about unused LoRA keys
    for i, lora_weight_keys in enumerate(lora_weight_keys_list):
        if len(lora_weight_keys) > 0:
            logger.warning("LoRA %d has unused keys: %s", i, ", ".join(list(lora_weight_keys)[:10]))

    # Apply accumulated deltas to model parameters (in-place, no state_dict copy)
    if undo_deltas:
        with torch.no_grad():
            for param_key, delta in undo_deltas.items():
                param = base_model.get_parameter(param_key)
                param.data.add_(delta.to(param.device, param.dtype))

    # For baked quantized LoRAs, record the raw checkpoint keys so undo can
    # reload the original weights from disk (captured at load time in the model).
    quantized_affected_unique = tuple(dict.fromkeys(quantized_affected))
    restore_map = getattr(base_model, "_quantized_restore_map", {})
    quantized_restore_keys = tuple(restore_map.get(p) for p in quantized_affected_unique)

    return {
        "lora_sds": lora_sds,
        "multipliers": multipliers,
        "affected_keys": tuple(undo_deltas.keys()),
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
        module = base_model.get_submodule(module_path)
        if raw_key is None:
            raise RuntimeError(
                f"cannot undo quantized LoRA on {module_path}: no raw checkpoint key "
                "was recorded at load time"
            )
        if dit_path is None:
            raise RuntimeError(
                f"cannot undo quantized LoRA on {module_path}: no dit_path was "
                "recorded at apply time"
            )
        restore_quantized_layer(module, dit_path, raw_key)

    if not lora_sds or not affected_keys:
        return

    logger.debug("Undoing LoRA on model (%d LoRA(s), %d keys)", len(lora_sds), len(affected_keys))

    # Build key sets for each LoRA (copy so we can mutate)
    lora_weight_keys_list = [set(sd.keys()) for sd in lora_sds]

    with torch.no_grad():
        for model_key in affected_keys:
            param = base_model.get_parameter(model_key)
            original_device = param.device
            weight_on_calc = param if original_device == calc_device else param.to(calc_device)

            accumulated_delta: Optional[torch.Tensor] = None

            for lora_weight_keys, lora_sd, multiplier in zip(
                lora_weight_keys_list, lora_sds, multipliers
            ):
                match = _match_lora_keys(model_key, lora_weight_keys)
                if match is None:
                    continue

                down_key, up_key, alpha_key = match
                down_weight = lora_sd[down_key]
                up_weight = lora_sd[up_key]
                alpha = lora_sd.get(alpha_key, down_weight.size()[0])

                delta = _compute_lora_delta(
                    weight_on_calc, down_weight, up_weight, alpha, multiplier, calc_device
                )

                if accumulated_delta is None:
                    accumulated_delta = delta
                else:
                    accumulated_delta = accumulated_delta + delta.to(
                        accumulated_delta.device, accumulated_delta.dtype
                    )

            if accumulated_delta is not None:
                param.data.sub_(accumulated_delta.to(param.device, param.dtype))

