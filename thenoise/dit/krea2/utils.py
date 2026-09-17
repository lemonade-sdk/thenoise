"""Shared loaders / helpers for the Krea 2 (K2) integration."""

import logging
from typing import Optional, Union

import torch
from accelerate import init_empty_weights

from thenoise.dit.krea2.encoder import (
    QWEN3_VL_4B_INSTRUCT_REPO_ID,
    Qwen3VLConditioner,
    TextEncoderConfig,
    load_qwen3_vl_conditioner,
)
from thenoise.dit.krea2.mmdit import SingleMMDiTConfig, SingleStreamDiT
from thenoise.utils.loader import load_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.text_encoder import find_tokenizer_dir

logger = logging.getLogger(__name__)



# The single config shipped with the OSS checkpoints (single_mmdit_large_wide).
single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


def _krea2_rms_map(key: str, tensor):
    """Reconcile Krea2's zero-centered RMSNorm ``scale`` with the shared ``weight``.

    The shared ``RMSNorm`` (``thenoise.utils.rms_norm``) stores the effective weight
    (ones-init). Krea2's checkpoint keeps a zero-centered ``scale`` where
    ``weight = scale + 1``, so rename ``.scale -> .weight`` and shift the value up
    by one. Applied at load time only; the runtime module is the shared one.
    """
    if key.endswith(".scale"):
        return key[: -len(".scale")] + ".weight", tensor + 1.0
    return key, tensor


def load_krea2_dit(
    dit_path: str,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    config: SingleMMDiTConfig = single_mmdit_large_wide,
) -> SingleStreamDiT:
    """Build the K2 single-stream MMDiT on meta and load weights."""
    device = torch.device(device)

    logger.info(f"Loading Krea 2 DiT weights from {dit_path}")
    with init_empty_weights():
        dit = SingleStreamDiT(config)

    return load_dit(
        dit,
        dit_path,
        device=device,
        dtype=dtype,
        key_map=lambda k: qk_norm_key_map(k, "qknorm.qnorm", "qknorm.knorm"),
        drop_keys=("last.down", "last.up"),
        value_map=_krea2_rms_map,
    )


def load_krea2_text_encoder(
    path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple = TextEncoderConfig.select_layers,
    tokenizer_dir: Optional[str] = None,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
) -> Qwen3VLConditioner:
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``path`` (local safetensors,
    ComfyUI or official key layout), tokenizer from ``tokenizer_dir`` (a local directory) when
    given, else from the vendored ``configs/tokenizer/`` directory, else from ``tokenizer_repo``."""
    return load_qwen3_vl_conditioner(
        path,
        dtype=dtype,
        device=device,
        max_length=max_length,
        select_layers=select_layers,
        tokenizer_dir=tokenizer_dir,
        tokenizer_repo=tokenizer_repo,
    )


__all__ = [
    "single_mmdit_large_wide",
    "SingleStreamDiT",
    "load_krea2_dit",
    "load_krea2_text_encoder",
    "find_tokenizer_dir",
    "Qwen3VLConditioner",
    "TextEncoderConfig",
]
