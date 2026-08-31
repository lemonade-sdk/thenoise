"""Central quant-aware DiT loader.

Every DiT adapter loads its weights through a single entry point,
``load_dit``, which automatically selects the correct quantization:

* **INT8** checkpoint (detected via ``is_int8_checkpoint``): weights
  land via ``load_int8_state_dict`` — quantized linears on modules exposing
  ``load_int8`` (``QuantizedLinear``), full-precision params cast to ``dtype``.
* **BF16 / plain** checkpoint: weights land via ``load_state_dict(assign=True)``,
  with the loaded tensors cast to ``dtype``.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import torch

from thenoise.utils.int8 import (
    build_int8_restore_map,
    is_int8_checkpoint,
    load_int8_state_dict,
)
from thenoise.utils.safetensors import load_dit_safetensors
from thenoise.utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_dit(
    model: torch.nn.Module,
    path: str,
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    drop_keys: Optional[tuple[str, ...]] = None,
    expected_missing: tuple[str, ...] = (),
    int8_key_map: Optional[Callable[[str], str]] = None,
) -> torch.nn.Module:
    """Load a DiT checkpoint into ``model``, selecting INT8 vs BF16 automatically.

    Args:
        model: the (meta-constructed) DiT to populate.
        path: safetensors checkpoint path.
        device: final device for all parameters/buffers.
        dtype: dtype for full-precision (BF16-path and INT8 non-quantized) params.
        drop_keys: optional tuple of key prefixes to drop from the checkpoint
            before loading (e.g. Krea2's unused ``last.down``/``last.up``).
            Applied on BOTH the INT8 and BF16 paths.
        expected_missing: optional substrings allowed to be absent from the
            checkpoint (model-internal buffers, e.g. Anima's RoPE buffers). If
            non-empty the load is non-strict and only these are tolerated;
            otherwise loading is strict.
        int8_key_map: optional transform applied to checkpoint keys on the INT8
            path only (e.g. Flux Klein's ComfyUI norm ``weight`` -> ``scale``
            rename). BF16 checkpoints carry the canonical names already.

    Returns:
        ``model`` (loaded in place, moved to ``device``).
    """
    device = torch.device(device)

    # Load the state dict once (stripping generic wrapper prefixes inside
    # ``load_dit_safetensors``) and apply ``drop_keys`` before branching, so the
    # INT8 and BF16 paths see the same prepared dict.
    sd = load_dit_safetensors(path, device=device, disable_mmap=True, dtype=None)
    if drop_keys:
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_keys)}

    if is_int8_checkpoint(path):
        if int8_key_map is not None:
            sd = {int8_key_map(k): v for k, v in sd.items()}
        # ConvRot group size / rotation flag is baked into each layer's weights
        # and scales at export time and must match the activation rotation at
        # inference; ``load_int8_state_dict`` reads it from each layer's
        # ``comfy_quant`` marker in the state dict (defaults to convrot, 256).
        load_int8_state_dict(model, sd, dtype=dtype)
        # Record each quantized layer's raw checkpoint key so a later LoRA undo
        # can reload the original INT8 weights from disk by key.
        model._int8_restore_map = build_int8_restore_map(path, int8_key_map)
        logger.info("Loaded INT8 checkpoint from %s", path)
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


__all__ = ["load_dit"]
