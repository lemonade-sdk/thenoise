"""Z-Image model loading utilities.

The DiT is the S3-DiT transformer; the text encoder is a Qwen3-4B model whose hidden
states feed the DiT's caption embedder. The DiT accepts both single-file and sharded
(HF) checkpoints. The text encoder is a *single file* (e.g. ComfyUI's
``text_encoders/qwen_3_4b.safetensors``): its ``Qwen3Config`` is vendored here so no
``config.json`` is fetched from the Hub, weights are loaded directly from the
safetensors file, and the tokenizer config files are vendored under ``configs/`` so
no tokenizer is fetched from the Hub either.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Union

import torch

from thenoise.dit.zimage.models import ZImageTransformer2DModel
from thenoise.utils.loader import load_dit
from thenoise.utils.safetensors import load_split_weights

logger = logging.getLogger(__name__)

#: The Qwen3 tokenizer config files are vendored in the package under ``configs/``
#: (mirroring the ``tokenizer/`` subfolder of the official Z-Image-Turbo repo). It
#: carries the Qwen chat template used by the caption encoder, so the tokenizer loads
#: offline with ``local_files_only=True`` and is never fetched from the Hub.
ZIMAGE_TOKENIZER_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "tokenizer")

# Vendored copy of the Z-Image Qwen3-4B ``text_encoder/config.json`` so the text
# encoder is built without fetching the config from the Hub. Qwen3 is natively
# supported by transformers (no remote code), so ``Qwen3Config(**...)`` reproduces
# ``AutoConfig.from_pretrained`` exactly.
QWEN3_4B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


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
    t_scale=1000.0,
    axes_dims=(32, 48, 48),
    axes_lens=(1024, 512, 512),
)


def load_zimage_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    loading_device: Optional[Union[str, torch.device]] = None,
    config: Optional[dict] = None,
) -> ZImageTransformer2DModel:
    """Build the Z-Image S3-DiT on meta and load weights."""
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)
    cfg = dict(ZIMAGE_DIT_CONFIG)
    if config:
        cfg.update(config)

    logger.info(f"Loading Z-Image DiT weights from {dit_path}")
    with torch.device("meta"):
        dit = ZImageTransformer2DModel(**cfg)

    return load_dit(dit, dit_path, device=loading_device, dtype=dtype)


def _load_qwen3(
    path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = True,
) -> "Qwen3ForCausalLM":
    """Build Qwen3-4B from the vendored config and load weights from a single file."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from accelerate import init_empty_weights

    config = Qwen3Config(**QWEN3_4B_CONFIG)
    with init_empty_weights():
        qwen3 = Qwen3ForCausalLM._from_config(config)

    logger.info(f"Loading Z-Image text encoder (Qwen3-4B) weights from {path}")
    sd = load_split_weights(path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    # Qwen3-4B ties the LM head to the input embeddings (tie_word_embeddings=true), so
    # the checkpoint omits lm_head.weight; re-tie so the strict load passes.
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]

    info = qwen3.load_state_dict(sd, strict=True, assign=True)
    if info.unexpected_keys or info.missing_keys:
        raise RuntimeError(
            f"Z-Image text encoder checkpoint did not match Qwen3-4B: "
            f"missing={info.missing_keys[:10]}, unexpected={info.unexpected_keys[:10]}"
        )

    qwen3.to(device)
    if dtype is not None:
        qwen3.to(dtype)
    return qwen3.eval().requires_grad_(False)


def load_zimage_text_encoder(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
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
    from transformers import AutoTokenizer

    if not path.endswith(".safetensors"):
        raise ValueError(
            f"Z-Image text encoder must be a single .safetensors file, got {path!r}. "
            "Download it with `python scripts/download_zimage.py`."
        )

    qwen3 = _load_qwen3(path, dtype=dtype, device=device)

    tokenizer_dir = tokenizer_dir or ZIMAGE_TOKENIZER_CONFIG_DIR
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(
            f"Z-Image tokenizer config directory not found at {tokenizer_dir}. "
            "Expected configs/tokenizer/ with tokenizer.json, tokenizer_config.json, "
            "vocab.json and merges.txt."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

    qwen3.config.use_cache = False
    model = qwen3.model  # bare Qwen3Model; hidden_states feed the caption embedder
    logger.info(f"Loaded Z-Image text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def find_zimage_tokenizer_dir(text_encoder_path: str, max_depth: int = 3) -> Optional[str]:
    """Locate a local ``tokenizer/`` directory near the text encoder file.

    The downloader drops the tokenizer under the output root (``<out>/tokenizer/``)
    while the text encoder lands under ``<out>/split_files/text_encoders/``. Search
    ``max_depth`` parent directories of the text encoder for a ``tokenizer/`` dir so
    the tokenizer is loaded offline when present. Returns ``None`` to fall back to the
    vendored ``configs/tokenizer/`` directory.
    """
    base = os.path.dirname(os.path.abspath(text_encoder_path))
    for _ in range(max_depth):
        cand = os.path.join(base, "tokenizer")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(base)
        if parent == base:
            break
        base = parent
    return None
