import os
import re
from typing import Dict, List, Optional, Tuple, TypedDict, Union
import torch
import torch.nn.functional as F

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _match_prefixed_lora_keys(
    lora_name: str,
    lora_weight_keys: str,
) -> Optional[Tuple[str, str, str]]:
    for (suffix_a, suffix_b) in [(".lora_down", ".lora_up"), (".lora_A", ".lora_B")]:
        a_key = lora_name + suffix_a + ".weight"
        b_key = lora_name + suffix_b + ".weight"
        alpha_key = lora_name + ".alpha"
        if a_key in lora_weight_keys and b_key in lora_weight_keys:
            return (a_key, b_key, alpha_key)
    return None        

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

    # sd-scripts naming: underscore-joined path
    for prefix in ["lora_unet_", ""]:
        lora_name = prefix + lora_name_without_prefix.replace(".", "_")
        res = _match_prefixed_lora_keys(lora_name, lora_weight_keys)
        if res:
            return res

    # diffusers-style naming: dotted path
    for prefix in ["diffusion_model.", ""]:
        lora_name = prefix + lora_name_without_prefix
        res = _match_prefixed_lora_keys(lora_name, lora_weight_keys)
        if res:
            return res

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
    Because the fused rank is ``3r``, ``compute_lora_delta``'s default scale
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
    # LoRA factors to match the model's parameters. The detection must include
    # buffers: quantized ``QuantizedLinear`` weights are buffers, not params.
    model_weight_keys = [
        k for k, _ in base_model.named_parameters() if k.endswith(".weight")
    ] + [
        k for k, _ in base_model.named_buffers() if k.endswith(".weight")
    ]
    use_fused = any(k.endswith(".attention.qkv.weight") for k in model_weight_keys)
    lora_sds = [_convert_fused_attention_lora(sd, use_fused) for sd in lora_sds]

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

