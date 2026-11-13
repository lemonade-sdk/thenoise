"""What the DiT adapters share to turn a prompt into conditioning: the text encoders,
their vendored tokenizers under ``configs/``, the chat templates, and the kernel
corrections no dependency makes for us.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Union

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
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    T5TokenizerFast,
)

from thenoise.dit.quantized import replace_linears
from thenoise.utils.loader import load_text_encoder_weights
from thenoise.utils.qwen_configs import (
    QWEN2_5_VL_CONFIG,
    QWEN2_5_VL_PREPROCESSOR_CONFIG,
    QWEN3_0_6B_CONFIG,
    QWEN3_VL_4B_INSTRUCT_CONFIG,
    QWEN3_VL_PREPROCESSOR_CONFIG,
    QWEN35_9B_CONFIG,
)

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
logger = logging.getLogger(__name__)
QWEN25_TOKENIZER_CONFIG_DIR = os.path.join(_CONFIG_DIR, "qwen25_tokenizer")
QWEN35_TOKENIZER_CONFIG_DIR = os.path.join(_CONFIG_DIR, "qwen35_tokenizer")
T5_TOKENIZER_CONFIG_DIR = os.path.join(_CONFIG_DIR, "t5")

# The shared ``qwen25_tokenizer`` config is the Qwen3 one; the fields that differ
# per variant are applied as post-load overrides.
QWEN3_06B_TOKENIZER_OVERRIDES = {"eos_token": "<|endoftext|>"}

# The VL token names a multimodal processor looks up on its tokenizer. The shared
# ``qwen25_tokenizer`` has them in its added-token table, it just does not name them.
QWEN3_VL_TOKENIZER_OVERRIDES = {
    "model_max_length": 262144,
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
}
QWEN2_5_VL_TOKENIZER_OVERRIDES: dict = {}

#: The chat and vision marker spellings shared by every multimodal conditioner here.
#: A processor looks the vision ones up by name on its tokenizer; ``VISION_BLOCK`` is
#: one reference image as the language model sees it.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VISION_BLOCK = VISION_START + IMAGE_PAD + VISION_END

#: Shared Qwen image-description prompt template (text-to-image path): the system
#: prompt, the closing ``user`` header, and the number of tokens that prefix
#: occupies (the user content begins at ``QWEN_VL_DROP_IDX``).
QWEN_VL_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
#: The ``assistant``-turn tail appended after the user content.
QWEN_VL_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
#: Token index where the user message content begins (after ``<|im_start|>user\n``).
QWEN_VL_DROP_IDX = 34

#: Token id of the chat-start marker: an added special token, so it does not move with
#: the vocabulary variant.
QWEN_CHAT_START_ID = 151644


def compute_drop_idx(
    input_ids: torch.Tensor, im_start_id: int = QWEN_CHAT_START_ID
) -> int:
    """Index where the user message content begins (after the ``user`` header).

    The user turn is the SECOND chat-start marker (the first opens the system turn)
    and its content starts three tokens later. Counted from the token ids because the
    prefix length depends on how the system prompt tokenizes. Returns 0 (drop
    nothing) when there is no second marker. ``input_ids`` has no batch axis.
    """
    ids = input_ids.tolist()
    count = 0
    for i, id_ in enumerate(ids):
        if id_ == im_start_id:
            count += 1
            if count == 2:
                return i + 3  # ``<|im_start|>`` ``user`` ``\n``
    return 0


def find_tokenizer_dir(text_encoder_path: str, max_depth: int = 3) -> Optional[str]:
    """Locate a local ``tokenizer/`` directory near a text encoder file.

    The downloader drops the tokenizer under the output root while the text encoder
    lands under ``<out>/split_files/text_encoders/``, so it searches ``max_depth``
    parent directories. Returns ``None`` to fall back to the vendored ``configs/``.
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

    ``tokenizer_dir`` must contain ``tokenizer.json`` (``vocab.json``/``merges.txt``
    are embedded in it). ``overrides`` are ``tokenizer_config.json`` fields applied
    after loading.
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
) -> Qwen3ForCausalLM:
    """Build a Qwen3 (0.6B/4B/8B) text encoder from a vendored config and load weights.

    The model is built on meta (via ``init_empty_weights``), its ``lm_head`` dropped
    (tied/absent in the checkpoints), linears swapped for quantized ones, then
    populated by the shared loader.
    """
    qwen3_config = Qwen3Config(**config)
    with init_empty_weights():
        qwen3 = Qwen3ForCausalLM._from_config(qwen3_config)
        del qwen3.lm_head
        replace_linears(qwen3)

    load_text_encoder_weights(qwen3, path, device=device, dtype=dtype)
    if dtype is not None:
        qwen3.to(dtype)
    return qwen3


def load_qwen3_text_encoder(
    path: str,
    *,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
) -> tuple:
    """Load a Qwen3-0.6B text encoder + tokenizer (single safetensors file).

    Returns ``(model, tokenizer)`` where ``model`` is the bare ``Qwen3Model``
    (LM head dropped).
    """
    qwen3 = load_qwen3_model(path, config=QWEN3_0_6B_CONFIG, dtype=dtype, device=device)
    qwen3.config.use_cache = False
    tokenizer = load_qwen3_tokenizer(overrides=QWEN3_06B_TOKENIZER_OVERRIDES)
    return qwen3.model, tokenizer


#: FLA's conv kernel ignores, silently, an activation name it does not know.
CONV_ACTIVATIONS = (None, "silu", "swish")

_PATCH_MARK = "_thenoise_fla_conv"


def _fla_conv(shipped: Callable, fwd: Callable) -> Callable:
    """The convolution run by ``fwd``. It works in ``[B, L, C]`` where the layer hands
    over and expects ``[B, C, L]``, views both ways through the projection's own buffer.
    """

    def conv(hidden_states, weight, bias=None, activation=None, **kwargs):
        if activation not in CONV_ACTIVATIONS:
            return shipped(hidden_states, weight, bias, activation, **kwargs)
        y = fwd(hidden_states.transpose(1, 2), weight, bias, None, activation=activation)
        return (y[0] if isinstance(y, tuple) else y).transpose(1, 2)

    return conv


def apply_qwen35_kernels() -> bool:
    """Put a Qwen3.5 conditioner's DeltaNet convolution on flash-linear-attention.

    transformers asks for the ``causal_conv1d`` CUDA package here, and without it hands
    MIOpen a depthwise conv measured three orders of magnitude off this part's GEMM rate.
    FLA has that kernel under a name transformers never looks for; its gated-delta-rule
    counterpart needs no patch, only an importable ``fla``. Idempotent.
    """
    # Here rather than at the top: the kernels want triton, which a CPU torch lacks.
    try:
        from fla.modules.causal_conv1d import causal_conv1d_fwd
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35
    except ImportError:
        logger.debug("flash-linear-attention's conv1d is unavailable; keeping the "
                     "shipped convolution")
        return False

    shipped = getattr(qwen35, "causal_conv1d_fn", None)
    if shipped is None:
        logger.warning("transformers no longer has causal_conv1d_fn to replace; the "
                       "Qwen3.5 conditioner runs on its own conv1d")
        return False
    if getattr(shipped, _PATCH_MARK, False):
        return True

    conv = _fla_conv(shipped, causal_conv1d_fwd)
    setattr(conv, _PATCH_MARK, True)
    qwen35.causal_conv1d_fn = conv
    logger.info("Qwen3.5 conditioner: causal conv1d on flash-linear-attention")
    return True


def load_qwen35_model(
    path: str,
    *,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    config: Optional[dict] = None,
) -> Qwen3_5ForConditionalGeneration:
    """Build a Qwen3.5 text encoder (9B by default) and load a local safetensors into it.

    The checkpoint is already in the ``Qwen3_5ForConditionalGeneration`` layout, so it
    needs no key map. The ``mtp.`` speculative-decoding block is dropped; the vision
    tower loads and never runs.
    """
    qwen35_config = Qwen3_5Config.from_dict(config or QWEN35_9B_CONFIG)
    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration._from_config(qwen35_config)
        del model.lm_head
        replace_linears(model)

    load_text_encoder_weights(
        model, path, device=device, dtype=dtype, drop_keys=("mtp.",)
    )
    if dtype is not None:
        model.to(dtype)
    return model


def load_qwen3_vl_model(
    path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    config: Optional[dict] = None,
    model_factory: Optional[Callable[["Qwen3VLConfig"], torch.nn.Module]] = None,
) -> Qwen3VLForConditionalGeneration:
    """Build a Qwen3-VL text encoder and load a local safetensors into it.

    Accepts the official HF layout and ComfyUI's ``model.``/``visual.`` keys.
    ``config`` is one of the vendored ``QWEN3_VL_*_CONFIG`` dicts.

    ``model_factory`` replaces the empty-model build (still under ``init_empty_weights``)
    for a file whose architecture is not the plain HF one.
    """
    config = Qwen3VLConfig.from_dict(config or QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        if model_factory is None:
            model = Qwen3VLForConditionalGeneration._from_config(config)
            del model.lm_head
            replace_linears(model)
        else:
            model = model_factory(config)

    load_text_encoder_weights(
        model,
        path,
        device=device,
        dtype=dtype,
        key_map=_convert_comfyui_qwen3vl_state_dict,
    )
    if dtype is not None:
        model.to(dtype)
    return model


def load_qwen2_5_vl_model(
    path: str,
    *,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
) -> Qwen2_5_VLForConditionalGeneration:
    """Build a Qwen2.5-VL-7B text encoder and load weights.

    Accepts the official HF layout and ComfyUI's ``model.``/``visual.`` keys.
    """
    config = Qwen2_5_VLConfig(**QWEN2_5_VL_CONFIG)
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration._from_config(config)
        del model.lm_head
        replace_linears(model)
    load_text_encoder_weights(
        model,
        path,
        device=device,
        dtype=dtype,
        key_map=_convert_qwen2_5_vl_keys,
    )
    return model


def load_qwen3_tokenizer(
    tokenizer_dir: Optional[str] = None,
    *,
    overrides: Optional[dict] = None,
):
    """Load a Qwen3 tokenizer and ensure a pad token."""
    tokenizer = load_tokenizer(tokenizer_dir or QWEN25_TOKENIZER_CONFIG_DIR, overrides=overrides)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qwen2_tokenizer(
    tokenizer_dir: Optional[str] = None, *, overrides: Optional[dict] = None
):
    """Load a Qwen2 tokenizer."""
    return load_tokenizer(
        tokenizer_dir or QWEN25_TOKENIZER_CONFIG_DIR, overrides=overrides
    )


def load_qwen35_tokenizer(tokenizer_dir: Optional[str] = None, *, overrides: Optional[dict] = None):
    """Load the Qwen3.5 tokenizer."""
    return load_tokenizer(tokenizer_dir or QWEN35_TOKENIZER_CONFIG_DIR, overrides=overrides)


def load_qwen3_vl_tokenizer(
    tokenizer_dir: Optional[str] = None, *, max_length: int, overrides: Optional[dict] = None
) -> tuple:
    """Load a Qwen3-VL tokenizer + fast processor (the same AutoTokenizer)."""
    tokenizer = load_tokenizer(
        tokenizer_dir or QWEN25_TOKENIZER_CONFIG_DIR, max_length=max_length, overrides=overrides
    )
    return tokenizer, tokenizer


def load_qwen3_vl_processor(tokenizer, *, size: Optional[dict] = None):
    """Build a Qwen3-VL processor (image preprocessor + the same tokenizer) locally.

    The image config is vendored (``QWEN3_VL_PREPROCESSOR_CONFIG``) so no
    ``preprocessor_config.json`` is fetched. The video processor is built with its
    defaults: this engine feeds the encoder single images only, but the processor
    requires one.

    ``size`` replaces the vendored grid limits, which are areas under the
    ``shortest_edge``/``longest_edge`` names: a model trained to see a narrower range
    of pixels gets its own, and the ``smart_resize`` round trip cannot quietly move an
    image off the grid it was pre-sized to.
    """
    image_config = dict(QWEN3_VL_PREPROCESSOR_CONFIG)
    if size is not None:
        image_config["size"] = dict(size)
    return Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor.from_dict(image_config),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
    )


def load_qwen2_5_vl_processor(tokenizer):
    """Build a Qwen2.5-VL image/video processor from the vendored preprocessor config."""
    image_processor = Qwen2VLImageProcessor.from_dict(QWEN2_5_VL_PREPROCESSOR_CONFIG)
    video_processor = Qwen2VLVideoProcessor.from_dict(QWEN2_5_VL_PREPROCESSOR_CONFIG)
    return Qwen2VLProcessor(
        image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor
    )


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load the T5 tokenizer used for LLM-adapter target tokens."""
    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)
    return T5TokenizerFast(
        vocab_file=os.path.join(T5_TOKENIZER_CONFIG_DIR, "spiece.model"),
        tokenizer_file=os.path.join(T5_TOKENIZER_CONFIG_DIR, "tokenizer.json"),
    )


# --------------------------------------------------------------------------- key maps


def _convert_comfyui_qwen3vl_state_dict(key: str) -> str:
    """Map a ComfyUI-style (bare ``model.`` / ``visual.``) Qwen3-VL key onto the HF
    ``Qwen3VLForConditionalGeneration`` layout. Official HF checkpoints pass through.
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
    ``Qwen2_5_VLForConditionalGeneration`` layout.
    """
    if key.startswith("model."):
        return key.replace("model.", "model.language_model.", 1)
    if key.startswith("visual."):
        return key.replace("visual.", "model.visual.", 1)
    return key


__all__ = [
    "apply_qwen35_kernels",
    "CONV_ACTIVATIONS",
    "find_tokenizer_dir",
    "load_tokenizer",
    "load_qwen3_model",
    "load_qwen3_text_encoder",
    "load_qwen35_model",
    "load_qwen35_tokenizer",
    "load_qwen3_vl_model",
    "load_qwen2_5_vl_model",
    "load_qwen3_tokenizer",
    "load_qwen2_tokenizer",
    "load_qwen3_vl_tokenizer",
    "load_qwen3_vl_processor",
    "load_qwen2_5_vl_processor",
    "load_t5_tokenizer",
    "QWEN25_TOKENIZER_CONFIG_DIR",
    "QWEN35_TOKENIZER_CONFIG_DIR",
    "T5_TOKENIZER_CONFIG_DIR",
    "QWEN_VL_SYSTEM_PROMPT",
    "QWEN_VL_PROMPT_SUFFIX",
    "QWEN_VL_DROP_IDX",
    "QWEN_CHAT_START_ID",
    "compute_drop_idx",
    "IM_START",
    "IM_END",
    "VISION_START",
    "VISION_END",
    "IMAGE_PAD",
    "VISION_BLOCK",
    "QWEN3_06B_TOKENIZER_OVERRIDES",
    "QWEN3_VL_TOKENIZER_OVERRIDES",
    "QWEN2_5_VL_TOKENIZER_OVERRIDES",
]
