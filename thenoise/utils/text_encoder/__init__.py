"""Shared text-encoder / tokenizer loading for the DiT adapters.

Every adapter (Flux Klein, Z-Image, Anima, Krea 2, Qwen-Image) loads its text
encoder and tokenizer through this module, so a new encoder only needs a config
(see ``thenoise.utils.qwen_configs``) plus a load function here. The vendored
tokenizer data lives under ``configs/``:

* ``qwen25_tokenizer`` -- the single Qwen BPE tokenizer shared by all Qwen
  variants (Flux Klein, Z-Image, Anima, Krea 2, Qwen-Image). The vocab, merges,
  normalizer, pre/post-processor and decoder are byte-identical across variants;
  only a few ``tokenizer_config.json`` fields differ, applied here as overrides.
* ``t5``             -- T5 tokenizer (Anima's LLM-adapter target tokens)

The model configs themselves are vendored in ``thenoise.utils.qwen_configs`` so
the encoders are built without fetching ``config.json`` from the Hub.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from transformers import (
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    T5TokenizerFast,
)

from thenoise.dit.quantized import replace_linears
from thenoise.utils.loader import load_text_encoder_weights
from thenoise.utils.qwen_configs import (
    QWEN2_5_VL_CONFIG,
    QWEN2_5_VL_PREPROCESSOR_CONFIG,
    QWEN3_0_6B_CONFIG,
    QWEN3_VL_4B_INSTRUCT_CONFIG,
)

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
QWEN25_TOKENIZER_CONFIG_DIR = os.path.join(_CONFIG_DIR, "qwen25_tokenizer")
T5_TOKENIZER_CONFIG_DIR = os.path.join(_CONFIG_DIR, "t5")

# The shared ``qwen25_tokenizer`` config is the Qwen3 (Flux Klein / Z-Image) one.
# Only these ``tokenizer_config.json`` fields differ across the other Qwen variants
# (the vocab/merges are byte-identical), so they are applied as post-load overrides.
QWEN3_06B_TOKENIZER_OVERRIDES = {"eos_token": "<|endoftext|>"}
QWEN3_VL_TOKENIZER_OVERRIDES = {"model_max_length": 262144}
QWEN2_5_VL_TOKENIZER_OVERRIDES: dict = {}


def find_tokenizer_dir(text_encoder_path: str, max_depth: int = 3) -> Optional[str]:
    """Locate a local ``tokenizer/`` directory near a text encoder file.

    The downloader drops the tokenizer under the output root (``<out>/tokenizer/``)
    while the text encoder lands under ``<out>/split_files/text_encoders/``. Searches
    ``max_depth`` parent directories of the text encoder for a ``tokenizer/`` dir so
    the tokenizer is loaded offline when present. Returns ``None`` to fall back to the
    vendored ``configs/`` directory.
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


def load_tokenizer(
    tokenizer_dir: str = QWEN25_TOKENIZER_CONFIG_DIR,
    *,
    max_length: Optional[int] = None,
    local_files_only: bool = True,
    overrides: Optional[dict] = None,
):
    """Load a tokenizer from a local directory via HF ``AutoTokenizer``.

    ``tokenizer_dir`` must contain ``tokenizer.json`` (and optionally a
    ``tokenizer_config.json``); ``vocab.json``/``merges.txt`` are not required as
    they are embedded in ``tokenizer.json``. ``overrides`` is an optional dict of
    ``tokenizer_config.json`` fields applied after loading, used to specialize the
    shared ``qwen25_tokenizer`` for the per-adapter Qwen variants.
    """
    kwargs = {}
    if max_length is not None:
        kwargs["max_length"] = max_length
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=local_files_only, **kwargs
    )
    if overrides:
        for key, value in overrides.items():
            setattr(tokenizer, key, value)
    return tokenizer


def load_qwen3_model(
    path: str,
    *,
    config: dict,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    label: str,
) -> Qwen3ForCausalLM:
    """Build a Qwen3 (0.6B/4B/8B) text encoder from a vendored config and load weights.

    The model is built on meta (via ``init_empty_weights``), its ``lm_head``
    dropped (tied/absent in the checkpoints), linears swapped for quantized ones,
    then populated by the shared loader. Returns the eval'd ``Qwen3ForCausalLM``
    (use ``.model`` for the bare model whose hidden states feed a DiT).
    """
    qwen3_config = Qwen3Config(**config)
    with init_empty_weights():
        qwen3 = Qwen3ForCausalLM._from_config(qwen3_config)
        del qwen3.lm_head
        replace_linears(qwen3)

    logger.info("Loading %s text encoder (Qwen3) weights from %s", label, path)
    load_text_encoder_weights(qwen3, path, device=device, dtype=dtype)
    if dtype is not None:
        qwen3.to(dtype)
    return qwen3.eval().requires_grad_(False)


def load_qwen3_text_encoder(
    path: str,
    *,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
) -> tuple:
    """Load the Anima Qwen3-0.6B text encoder + tokenizer (single safetensors file).

    Returns ``(model, tokenizer)`` where ``model`` is the bare ``Qwen3Model``
    (LM head dropped) whose hidden states feed the Anima LLM adapter.
    """
    qwen3 = load_qwen3_model(
        path, config=QWEN3_0_6B_CONFIG, dtype=dtype, device=device, label="Anima"
    )
    qwen3.config.use_cache = False
    tokenizer = load_qwen3_tokenizer(overrides=QWEN3_06B_TOKENIZER_OVERRIDES)
    model = qwen3.model
    logger.info(f"Loaded Anima text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_qwen3_vl_model(
    path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> Qwen3VLForConditionalGeneration:
    """Build the Krea 2 Qwen3-VL-4B text encoder and load weights from a local safetensors.

    Accepts the official HF layout and ComfyUI's ``model.``/``visual.`` keys.
    """
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)
        del model.lm_head
        replace_linears(model)

    logger.info("Loading Krea 2 text encoder (Qwen3-VL) weights from %s", path)
    load_text_encoder_weights(
        model,
        path,
        device=device,
        dtype=dtype,
        key_map=_convert_comfyui_qwen3vl_state_dict,
    )
    if dtype is not None:
        model.to(dtype)
    return model.eval().requires_grad_(False)


def load_qwen2_5_vl_model(
    path: str,
    *,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
) -> Qwen2_5_VLForConditionalGeneration:
    """Build the Qwen-Image Qwen2.5-VL-7B text encoder and load weights.

    Accepts the official HF layout and ComfyUI's ``model.``/``visual.`` keys.
    """
    config = Qwen2_5_VLConfig(**QWEN2_5_VL_CONFIG)
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration._from_config(config)
        del model.lm_head
        replace_linears(model)
    logger.info("Loading Qwen2.5-VL text encoder from %s", path)
    load_text_encoder_weights(
        model,
        path,
        device=device,
        dtype=dtype,
        key_map=_convert_qwen2_5_vl_keys,
    )
    return model.eval().requires_grad_(False)


def load_qwen3_tokenizer(
    tokenizer_dir: Optional[str] = None,
    *,
    overrides: Optional[dict] = None,
):
    """Load a Qwen3 tokenizer and ensure a pad token (Anima's LLM adapter).

    ``tokenizer_dir`` defaults to the shared ``qwen25_tokenizer`` vendored config;
    pass an external directory to load a tokenizer from there instead.
    """
    tokenizer = load_tokenizer(tokenizer_dir or QWEN25_TOKENIZER_CONFIG_DIR, overrides=overrides)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qwen2_tokenizer(
    tokenizer_dir: Optional[str] = None, max_length: int = 1024, *, overrides: Optional[dict] = None
):
    """Load the Qwen-Image Qwen2 tokenizer (capped at 1024 tokens)."""
    return load_tokenizer(
        tokenizer_dir or QWEN25_TOKENIZER_CONFIG_DIR, max_length=max_length, overrides=overrides
    )


def load_qwen3_vl_tokenizer(
    tokenizer_dir: Optional[str] = None, *, max_length: int, overrides: Optional[dict] = None
) -> tuple:
    """Load the Krea 2 Qwen3-VL tokenizer + fast processor (the same AutoTokenizer)."""
    tokenizer = load_tokenizer(
        tokenizer_dir or QWEN25_TOKENIZER_CONFIG_DIR, max_length=max_length, overrides=overrides
    )
    return tokenizer, tokenizer


def load_qwen2_5_vl_processor(tokenizer):
    """Build a Qwen2.5-VL image/video processor from the vendored preprocessor config."""
    image_processor = Qwen2VLImageProcessor.from_dict(QWEN2_5_VL_PREPROCESSOR_CONFIG)
    video_processor = Qwen2VLVideoProcessor.from_dict(QWEN2_5_VL_PREPROCESSOR_CONFIG)
    return Qwen2VLProcessor(
        image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor
    )


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load the T5 tokenizer used for Anima's LLM-adapter target tokens.

    ``t5_tokenizer_path`` is an optional local directory; else the vendored
    ``configs/t5/`` is used.
    """
    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)
    return T5TokenizerFast(
        vocab_file=os.path.join(T5_TOKENIZER_CONFIG_DIR, "spiece.model"),
        tokenizer_file=os.path.join(T5_TOKENIZER_CONFIG_DIR, "tokenizer.json"),
    )


# --------------------------------------------------------------------------- key maps


def _convert_comfyui_qwen3vl_state_dict(key: str) -> str:
    """Map a ComfyUI-style (bare ``model.`` / ``visual.``) Qwen3-VL state dict key onto the HF
    ``Qwen3VLForConditionalGeneration`` layout. Official HF checkpoints already use the
    ``model.language_model.`` / ``model.visual.`` layout and pass through unchanged.
    """
    if key.startswith("model.language_model.") or key.startswith("model.visual."):
        return key
    if key.startswith("visual."):
        return "model.visual." + key[len("visual.") :]
    if key.startswith("language_model."):
        return "model." + key
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    return key


def _convert_qwen2_5_vl_keys(key: str) -> str:
    """Normalize the raw Qwen2.5-VL layout (``model.``/``visual.``) to the
    ``Qwen2_5_VLForConditionalGeneration`` layout (``model.language_model.`` /
    ``model.visual.``)."""
    if key.startswith("model."):
        return key.replace("model.", "model.language_model.", 1)
    if key.startswith("visual."):
        return key.replace("visual.", "model.visual.", 1)
    return key


__all__ = [
    "find_tokenizer_dir",
    "load_tokenizer",
    "load_qwen3_model",
    "load_qwen3_text_encoder",
    "load_qwen3_vl_model",
    "load_qwen2_5_vl_model",
    "load_qwen3_tokenizer",
    "load_qwen2_tokenizer",
    "load_qwen3_vl_tokenizer",
    "load_qwen2_5_vl_processor",
    "load_t5_tokenizer",
    "QWEN25_TOKENIZER_CONFIG_DIR",
    "T5_TOKENIZER_CONFIG_DIR",
    "QWEN3_06B_TOKENIZER_OVERRIDES",
    "QWEN3_VL_TOKENIZER_OVERRIDES",
    "QWEN2_5_VL_TOKENIZER_OVERRIDES",
]
