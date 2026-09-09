"""Z-Image model loading utilities.

The DiT is the S3-DiT transformer; the text encoder is a Qwen3-4B model whose hidden
states feed the DiT's caption embedder.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Union

import torch
from accelerate import init_empty_weights

from thenoise.dit.zimage.models import ZImageTransformer2DModel
from thenoise.utils.loader import load_dit
from thenoise.utils.qwen_configs import QWEN3_4B_CONFIG
from thenoise.utils.text_encoder import (
    QWEN25_TOKENIZER_CONFIG_DIR,
    load_qwen3_model,
    load_tokenizer,
)

logger = logging.getLogger(__name__)


ZIMAGE_DIT_CONFIG = dict(
    patch_size=2,
    f_patch_size=1,
    in_channels=16,
    dim=3840,
    n_layers=30,
    n_refiner_layers=2,
    n_heads=30,
    n_kv_heads=30,
    norm_eps=1e-5,
    qk_norm=True,
    cap_feat_dim=2560,
    rope_theta=256.0,
    axes_dims=(32, 48, 48),
)


def load_zimage_dit(
    dit_path: str,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    config: Optional[dict] = None,
) -> ZImageTransformer2DModel:
    """Build the Z-Image S3-DiT on meta and load weights."""
    device = torch.device(device)
    cfg = dict(ZIMAGE_DIT_CONFIG)
    if config:
        cfg.update(config)

    logger.info(f"Loading Z-Image DiT weights from {dit_path}")
    with init_empty_weights():
        dit = ZImageTransformer2DModel(**cfg)

    return load_dit(dit, dit_path, device=device, dtype=dtype)


def load_zimage_text_encoder(
    path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    tokenizer_dir: Optional[str] = None,
) -> tuple:
    """Load the Z-Image Qwen3 text encoder + tokenizer.

    ``path`` is a single safetensors file (e.g. ComfyUI's
    ``text_encoders/qwen_3_4b.safetensors``) in the bare HF Qwen3 layout
    (``model.layers.N.*``, ``model.embed_tokens.weight``, ``model.norm.weight``;
    ``lm_head.weight`` tied to the embeddings). The model config is vendored, so no
    ``config.json`` is needed next to the weights.

    The tokenizer is loaded from ``tokenizer_dir`` if given (a local directory), else
    from the vendored ``configs/tokenizer/`` directory. Either must carry the Qwen
    chat template used by the caption encoder.

    Returns ``(text_encoder, tokenizer)`` where ``text_encoder`` is the bare Qwen3
    model (LM head dropped) whose ``hidden_states`` feed the DiT's caption embedder.
    """
    if not path.endswith(".safetensors"):
        raise ValueError(
            f"Z-Image text encoder must be a single .safetensors file, got {path!r}. "
            "Download it with `python scripts/download_zimage.py`."
        )

    qwen3 = load_qwen3_model(path, config=QWEN3_4B_CONFIG, dtype=dtype, device=device)

    tokenizer_dir = tokenizer_dir or QWEN25_TOKENIZER_CONFIG_DIR
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(
            f"Z-Image tokenizer config directory not found at {tokenizer_dir}. "
            "Expected configs/tokenizer/ with tokenizer.json and tokenizer_config.json."
        )
    tokenizer = load_tokenizer(tokenizer_dir)

    qwen3.config.use_cache = False
    model = qwen3.model  # bare Qwen3Model; hidden_states feed the caption embedder
    logger.info(f"Loaded Z-Image text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer
