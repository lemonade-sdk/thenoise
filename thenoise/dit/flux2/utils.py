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
from accelerate import init_empty_weights

from thenoise.dit.flux2.models import Flux2, Flux2Params, Klein4BParams, Klein9BParams
from thenoise.utils.loader import load_dit
from thenoise.utils.qwen_configs import QWEN3_4B_CONFIG, QWEN3_8B_CONFIG
from thenoise.utils.safetensors import WRAP_PREFIXES, MemoryEfficientSafeOpen
from thenoise.utils.text_encoder import (
    QWEN3_TOKENIZER_CONFIG_DIR,
    load_qwen3_model,
    load_tokenizer,
)

logger = logging.getLogger(__name__)

#: Qwen3 hidden layers whose outputs are concatenated to build the DiT context.
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512

#: hidden_size -> Klein params. Used to pick the variant from the DiT checkpoint.
_KLEIN_VARIANTS = {3072: Klein4BParams, 4096: Klein9BParams}


# ComfyUI's INT8 exporter stores RMSNorm ``scale`` params under the ``weight``
# name. Reconcile those keys so the shared loader assigns them to the model's
# ``scale`` parameters (only the QKNorm scales carry this suffix).
_NORM_WEIGHT_SUFFIXES = (".norm.key_norm.weight", ".norm.query_norm.weight")


def _flux2_key_map(key: str) -> str:
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
    with init_empty_weights():
        dit = Flux2(params)
    return load_dit(
        dit,
        dit_path,
        device=device,
        dtype=dtype,
        key_map=_flux2_key_map,
    )


class Qwen3Embedder:
    """Qwen3 -> DiT context embedder (concatenates hidden states [9, 18, 27]).

    Mirrors the Flux.2 pipeline: applies the Qwen chat template with
    ``enable_thinking=False`` and returns ``[1, 512, 3 * hidden_size]``.
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = MAX_LENGTH

    @property
    def device(self):
        """The embedder's model device (follows the model when it is moved)."""
        return next(self.model.parameters()).device

    def to(self, device):
        """Forward placement to the wrapped model (so the memory manager can move it)."""
        self.model.to(device)
        return self

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
        device = self.device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
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

    ``path`` is a safetensors checkpoint.  The tokenizer is loaded from ``tokenizer_dir`` 
    if given, else from the vendored Z-Image Qwen3 tokenizer directory.
    """
    tokenizer_dir = tokenizer_dir or QWEN3_TOKENIZER_CONFIG_DIR
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(
            f"Flux Klein tokenizer config directory not found at {tokenizer_dir}."
        )
    tokenizer = load_tokenizer(tokenizer_dir)

    qwen3 = load_qwen3_model(
        path,
        config=(QWEN3_8B_CONFIG if is_8b else QWEN3_4B_CONFIG),
        dtype=dtype,
        device=device,
        label=f"Flux Klein (Qwen3-{'8B' if is_8b else '4B'})",
    )
    qwen3.config.use_cache = False
    embedder = Qwen3Embedder(tokenizer, qwen3.model)  # bare Qwen3Model -> hidden_states
    logger.info(
        f"Loaded Flux Klein text encoder. Parameters: "
        f"{sum(p.numel() for p in qwen3.parameters()):,}"
    )
    return embedder


__all__ = [
    "detect_klein_params",
    "load_flux2_dit",
    "load_qwen3_embedder",
    "OUTPUT_LAYERS_QWEN3",
    "MAX_LENGTH",
]
