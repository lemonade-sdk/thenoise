"""Qwen-Image text encoder (Qwen2.5-VL-7B), prompt-embedding helpers, and latents.

Ported from kohya-ss/musubi-tuner's ``qwen_image/qwen_image_utils.py``.
The text encoder is a ``Qwen2_5_VLForConditionalGeneration`` (7B) that both
encodes the prompt alone (text-to-image) and, when an image is supplied (edit),
the prompt together with the input image as vision tokens. Weights load through
``thenoise.utils.loader.load_text_encoder_weights``.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2Tokenizer, Qwen2VLProcessor

from thenoise.utils.loader import load_text_encoder_weights
from thenoise.utils.setup_logging import setup_logging
from thenoise.dit.quantized import replace_linears

from PIL import Image

setup_logging()
import logging

logger = logging.getLogger(__name__)

# Vendored Qwen2.5-VL tokenizer + processor config dir (offline-safe).
TOKENIZER_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "tokenizer")

QWEN2_5_VL_CONFIG_JSON = """{
  "architectures": ["Qwen2_5_VLForConditionalGeneration"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "image_token_id": 151655,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 128000,
  "max_window_layers": 28,
  "model_type": "qwen2_5_vl",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {"mrope_section": [16, 24, 24], "rope_type": "default", "type": "default"},
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.53.1",
  "use_cache": true,
  "use_sliding_window": false,
  "video_token_id": 151656,
  "vision_config": {
    "depth": 32,
    "fullatt_block_indexes": [7, 15, 23, 31],
    "hidden_act": "silu",
    "hidden_size": 1280,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 3420,
    "model_type": "qwen2_5_vl",
    "num_heads": 16,
    "out_hidden_size": 3584,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2,
    "tokens_per_second": 2,
    "torch_dtype": "float32",
    "window_size": 112
  },
  "vocab_size": 152064
}"""


def _convert_qwen2_5_vl_keys(key: str) -> str:
    """Normalize the raw Qwen2.5-VL layout (``model.``/``visual.``) to the
    ``Qwen2_5_VLForConditionalGeneration`` layout (``model.language_model.`` /
    ``model.visual.``)."""
    if key.startswith("model."):
        return key.replace("model.", "model.language_model.", 1)
    if key.startswith("visual."):
        return key.replace("visual.", "model.visual.", 1)
    return key


def load_qwen2_5_vl(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
) -> Qwen2_5_VLForConditionalGeneration:
    """Build and load the Qwen2.5-VL-7B text encoder from a local safetensors.

    The 7B model is constructed on ``meta`` (via ``init_empty_weights``) so only
    the checkpoint weights are materialized; the loader assigns them directly and
    moves the model to ``device``.
    """
    config = Qwen2_5_VLConfig(**json.loads(QWEN2_5_VL_CONFIG_JSON))
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration._from_config(config)
        del model.lm_head
        replace_linears(model)
    logger.info("Loading Qwen2.5-VL text encoder from %s", ckpt_path)
    load_text_encoder_weights(
        model,
        ckpt_path,
        device=device,
        dtype=dtype,
        key_map=_convert_qwen2_5_vl_keys,
    )
    return model.eval().requires_grad_(False)


def load_qwen2_tokenizer(tokenizer_dir: str) -> Qwen2Tokenizer:
    return Qwen2Tokenizer.from_pretrained(tokenizer_dir, max_length=1024, local_files_only=True)


def extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    split_hidden_states = [hidden_states[i][mask[i].bool()] for i in range(hidden_states.shape[0])]
    return split_hidden_states


def _mask_and_stack(split_hidden_states, drop_idx: int):
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    return prompt_embeds, encoder_attention_mask


def get_qwen_prompt_embeds(
    tokenizer: Qwen2Tokenizer,
    vlm: Qwen2_5_VLForConditionalGeneration,
    prompt: Union[str, List[str]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode the prompt alone (text-to-image) -> (prompt_embeds, mask)."""
    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    drop_idx = 34
    tokenizer_max_length = 1024

    prompt = [prompt] if isinstance(prompt, str) else prompt
    txt = [prompt_template_encode.format(e) for e in prompt]
    txt_tokens = tokenizer(
        txt, max_length=tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
    ).to(vlm.device)
    with torch.no_grad():
        encoder_hidden_states = vlm.model(
            input_ids=txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask, output_hidden_states=True
        )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    split_hidden_states = extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    return _mask_and_stack(split_hidden_states, drop_idx)


def _resize_for_vlm(image: Image.Image, max_pixels: int = 384 * 384) -> Image.Image:
    """Scale a PIL image to fit within ``max_pixels`` (area-based, aspect-preserving).

    The vision encoder tokenizes each 14x14 patch into image tokens; full-resolution
    edit images would inject thousands of tokens, drowning out the short instruction
    and overflowing the RoPE buffer. Comfy scales to 384x384 for the same reason.
    """
    w, h = image.size
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    if (new_w, new_h) != (w, h):
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


def _compute_drop_idx(input_ids: torch.Tensor) -> int:
    """Index where the user message content begins (after ``<|im_start|>user\n``).

    The edit template drops the system prompt + user header so the DiT text
    conditioning is the image/instruction content (matching Comfy's
    ``template_end`` logic). The user content follows the second ``<|im_start|>``
    (the user turn); we drop through the ``user\n`` header tokens that follow it.
    """
    ids = input_ids[0].tolist()
    im_start = 151644
    count = 0
    for i, id_ in enumerate(ids):
        if id_ == im_start:
            count += 1
            if count == 2:
                return i + 3  # ``<|im_start|>`` ``user`` ``\n``
    return 0


def get_qwen_prompt_embeds_with_image(
    vl_processor: Qwen2VLProcessor,
    vlm: Qwen2_5_VLForConditionalGeneration,
    prompt: Union[str, List[str]],
    images=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode prompt + input image(s) (edit) -> (prompt_embeds, mask)."""
    system = (
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
        "size, texture, objects, background), then explain how the user's text "
        "instruction should alter or modify the image. Generate a new image that "
        "meets the user's requirements while maintaining consistency with the "
        "original input where appropriate.<|im_end|>\n"
        "<|im_start|>user\n"
    )

    if images is None:
        images = []
    elif isinstance(images, (list, tuple)):
        images = list(images)
    else:
        images = [images]

    image_prompt = "".join(
        f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"
        for i in range(len(images))
    )
    template = system + image_prompt + "{}<|im_end|>\n<|im_start|>assistant\n"

    prompt = [prompt] if isinstance(prompt, str) else prompt
    vl_image_inputs = [_resize_for_vlm(img) for img in images] or None

    txt = [template.format(e) for e in prompt]
    model_inputs = vl_processor(text=txt, images=vl_image_inputs, padding=True, return_tensors="pt").to(vlm.device)
    with torch.no_grad():
        encoder_hidden_states = vlm.model(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values if vl_image_inputs is not None else None,
            image_grid_thw=model_inputs.image_grid_thw if vl_image_inputs is not None else None,
            output_hidden_states=True,
        )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    split_hidden_states = extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    drop_idx = _compute_drop_idx(model_inputs.input_ids)
    return _mask_and_stack(split_hidden_states, drop_idx)


# ------------------------------------------------------------------- latents
def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack canonical ``[B, C, H, W]`` (or ``[B, C, 1, H, W]``) -> ``[B, H//2*W//2, C*4]``."""
    batch_size = latents.shape[0]
    if latents.ndim == 4 or latents.shape[2] == 1:
        num_channels_latents = latents.shape[1]
        height = latents.shape[-2]
        width = latents.shape[-1]
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    num_layers = latents.shape[1]
    num_channels_latents = latents.shape[2]
    height = latents.shape[-2]
    width = latents.shape[-1]
    latents = latents.view(batch_size, num_layers, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
    return latents.reshape(batch_size, num_layers * (height // 2) * (width // 2), num_channels_latents * 4)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack ``[B, H//2*W//2, C*4]`` -> canonical ``[B, C, H, W]``."""
    batch_size = latents.shape[0]
    num_channels_latents = latents.shape[2] // 4
    height = height // 2
    width = width // 2
    latents = latents.reshape(batch_size, height, width, num_channels_latents, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, num_channels_latents, height * 2, width * 2)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


__all__ = [
    "load_qwen2_5_vl",
    "load_qwen2_tokenizer",
    "get_qwen_prompt_embeds",
    "get_qwen_prompt_embeds_with_image",
    "pack_latents",
    "unpack_latents",
    "calculate_shift",
    "TOKENIZER_CONFIG_DIR",
]
