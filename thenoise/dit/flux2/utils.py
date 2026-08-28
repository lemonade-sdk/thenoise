"""Flux.2 (Flux Klein) model-loading utilities.

The DiT is the ``Flux2`` transformer; the text encoder is a Qwen3 (4B or 8B)
language model whose hidden states from layers [9, 18, 27] are concatenated to
form the DiT's context (context width = 3 * Qwen3 hidden). The Klein DiT variant
(4B vs 9B) is read from the checkpoint's ``img_in`` width, which selects the
matching Qwen3 text encoder.

The Qwen3 tokenizer config files are vendored under the Z-Image package
(``thenoise/dit/zimage/configs/tokenizer/``) and reused here — it is the same
Qwen3 tokenizer, and its chat template accepts ``enable_thinking``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Union

import torch
from einops import rearrange
from transformers import Qwen3Config, Qwen3ForCausalLM
from accelerate import init_empty_weights

from thenoise.dit.flux2.models import Flux2, Flux2Params, Klein4BParams, Klein9BParams
from thenoise.dit.zimage.utils import QWEN3_4B_CONFIG, ZIMAGE_TOKENIZER_CONFIG_DIR
from thenoise.utils.loader import load_dit
from thenoise.utils.safetensors import (
    WRAP_PREFIXES,
    MemoryEfficientSafeOpen,
    load_split_weights,
)

logger = logging.getLogger(__name__)

#: Qwen3 hidden layers whose outputs are concatenated to build the DiT context.
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512

# Vendored Qwen3-8B config (the 4B config is imported from the Z-Image package).
QWEN3_8B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
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
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

#: hidden_size -> Klein params. Used to pick the variant from the DiT checkpoint.
_KLEIN_VARIANTS = {3072: Klein4BParams, 4096: Klein9BParams}


# ComfyUI's INT8 exporter stores RMSNorm ``scale`` params under the ``weight``
# name. Reconcile those keys so the shared loader assigns them to the model's
# ``scale`` parameters (only the QKNorm scales carry this suffix).
_NORM_WEIGHT_SUFFIXES = (".norm.key_norm.weight", ".norm.query_norm.weight")


def _flux2_int8_key_map(key: str) -> str:
    for suffix in _NORM_WEIGHT_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(".weight")] + ".scale"
    return key


def detect_klein_params(dit_path: str) -> Flux2Params:
    """Return the Klein variant params (4B / 9B) from the DiT's ``img_in`` width."""
    with MemoryEfficientSafeOpen(dit_path) as f:
        for key in f.keys():
            k = key
            for prefix in WRAP_PREFIXES:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            if k == "img_in.weight":
                hidden = f.header[key]["shape"][0]
                cls = _KLEIN_VARIANTS.get(hidden)
                if cls is None:
                    raise ValueError(
                        f"Flux Klein img_in width {hidden} is not a known variant "
                        "(expected 3072 for 4B or 4096 for 9B)"
                    )
                return cls()
    raise ValueError(
        f"could not determine Flux Klein variant from {dit_path} (no img_in.weight key)"
    )


def load_flux2_dit(
    dit_path: str,
    params: Flux2Params,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> Flux2:
    """Build the Flux2 DiT on meta and load the checkpoint weights."""
    device = torch.device(device)
    logger.info(f"Loading Flux Klein DiT weights from {dit_path}")
    with torch.device("meta"):
        dit = Flux2(params)
    return load_dit(
        dit,
        dit_path,
        device=device,
        dtype=dtype,
        int8_key_map=_flux2_int8_key_map,
    )


def _load_qwen3(
    path: str,
    is_8b: bool,
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> "Qwen3ForCausalLM":
    """Build Qwen3 (4B/8B) from the vendored config and load a checkpoint."""
    config = Qwen3Config(**(QWEN3_8B_CONFIG if is_8b else QWEN3_4B_CONFIG))
    with init_empty_weights():
        qwen3 = Qwen3ForCausalLM._from_config(config)

    logger.info(f"Loading Flux Klein text encoder (Qwen3-{'8B' if is_8b else '4B'}) weights from {path}")
    sd = load_split_weights(path, device=str(device), disable_mmap=True, dtype=dtype)
    if not is_8b:
        # Qwen3-4B ties the LM head to the input embeddings (tie_word_embeddings=true),
        # so the checkpoint omits lm_head.weight; re-tie so the strict load passes.
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]

    info = qwen3.load_state_dict(sd, strict=True, assign=True)
    if info.unexpected_keys or info.missing_keys:
        raise RuntimeError(
            f"Flux Klein text encoder checkpoint did not match Qwen3-{'8B' if is_8b else '4B'}: "
            f"missing={info.missing_keys[:10]}, unexpected={info.unexpected_keys[:10]}"
        )
    qwen3.to(device)
    if dtype is not None:
        qwen3.to(dtype)
    return qwen3.eval().requires_grad_(False)


class Qwen3Embedder:
    """Qwen3 -> DiT context embedder (concatenates hidden states [9, 18, 27]).

    Mirrors the Flux.2 pipeline: applies the Qwen chat template with
    ``enable_thinking=False`` and returns ``[1, 512, 3 * hidden_size]``.
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = MAX_LENGTH
        self.device = next(model.parameters()).device

    def __call__(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")


def load_qwen3_embedder(
    path: str,
    is_8b: bool,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    tokenizer_dir: Optional[str] = None,
) -> Qwen3Embedder:
    """Load the Qwen3 text encoder + tokenizer and wrap it as a context embedder.

    ``path`` is a safetensors checkpoint (single file or the first ``00001-of-N``
    shard). The tokenizer is loaded from ``tokenizer_dir`` if given, else from the
    vendored Z-Image Qwen3 tokenizer directory.
    """
    from transformers import AutoTokenizer

    tokenizer_dir = tokenizer_dir or ZIMAGE_TOKENIZER_CONFIG_DIR
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(
            f"Flux Klein tokenizer config directory not found at {tokenizer_dir}."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

    qwen3 = _load_qwen3(path, is_8b=is_8b, dtype=dtype, device=device)
    qwen3.config.use_cache = False
    embedder = Qwen3Embedder(tokenizer, qwen3.model)  # bare Qwen3Model -> hidden_states
    logger.info(
        f"Loaded Flux Klein text encoder. Parameters: "
        f"{sum(p.numel() for p in qwen3.parameters()):,}"
    )
    return embedder


def find_flux2_tokenizer_dir(text_encoder_path: str, max_depth: int = 3) -> Optional[str]:
    """Locate a local ``tokenizer/`` dir near the text encoder (falls back to vendored).

    The downloader drops the tokenizer under the output root (``<out>/tokenizer/``)
    while the text encoder lands under ``<out>/split_files/text_encoders/``. Searches
    ``max_depth`` parent directories; returns ``None`` to fall back to the vendored
    Z-Image tokenizer.
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


__all__ = [
    "detect_klein_params",
    "load_flux2_dit",
    "load_qwen3_embedder",
    "find_flux2_tokenizer_dir",
    "QWEN3_8B_CONFIG",
    "OUTPUT_LAYERS_QWEN3",
    "MAX_LENGTH",
]
