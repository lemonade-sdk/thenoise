"""Shared loaders / helpers for the Krea 2 (K2) integration."""

import logging
from typing import Optional, Union

import torch

from thenoise.dit.krea2.encoder import (
    QWEN3_VL_4B_INSTRUCT_REPO_ID,
    Qwen3VLConditioner,
    TextEncoderConfig,
    load_qwen3_vl_conditioner,
)
from thenoise.dit.krea2.mmdit import SingleMMDiTConfig, SingleStreamDiT
from thenoise.utils.safetensors import load_dit_safetensors

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


def load_krea2_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    config: SingleMMDiTConfig = single_mmdit_large_wide,
    loading_device: Optional[Union[str, torch.device]] = None,
) -> SingleStreamDiT:
    """Build the K2 single-stream MMDiT on meta and load weights (assign=True).

    bf16 only: fp8 is dropped. ``lora_weights`` (a list of loaded LoRA state dicts, with
    optional ``lora_multipliers``) are merged into the base weights at load time.
    """
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)

    logger.info(f"Loading Krea 2 DiT weights from {dit_path}")
    with torch.device("meta"):
        dit = SingleStreamDiT(config)

    sd = load_dit_safetensors(
        dit_path,
        device=loading_device,
        disable_mmap=True,
        dtype=dtype,
        # Some older Krea 2 checkpoints carry leftover unused ``last.down.*`` /
        # ``last.up.*`` keys — drop them so the strict load still passes.
        drop_keys=("last.down", "last.up"),
    )

    dit.load_state_dict(sd, strict=True, assign=True)

    return dit


def load_krea2_text_encoder(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
) -> Qwen3VLConditioner:
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``path`` (local safetensors,
    ComfyUI or official key layout), tokenizer from ``tokenizer_repo`` (Hub id or local dir)."""
    return load_qwen3_vl_conditioner(
        path,
        dtype=dtype,
        device=device,
        max_length=max_length,
        select_layers=select_layers,
        tokenizer_repo=tokenizer_repo,
    )
