"""Qwen-Image prompt-embedding helpers and latents.

Ported from kohya-ss/musubi-tuner's ``qwen_image/qwen_image_utils.py``.
The text encoder (a ``Qwen2_5_VLForConditionalGeneration`` 7B) and its tokenizer
are loaded through ``thenoise.utils.text_encoder``; this module only encodes the
prompt (alone for text-to-image, or with an input image for edits) into the DiT
conditioning, and packs/unpacks the latent layout.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import torch
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from thenoise.utils.setup_logging import setup_logging
from thenoise.utils.image_tensor import resize_to_area
from thenoise.utils.text_encoder import QWEN_IMAGE_DROP_IDX, QWEN_IMAGE_PROMPT_SUFFIX, QWEN_IMAGE_SYSTEM_PROMPT

setup_logging()
import logging

logger = logging.getLogger(__name__)


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
    prompt_template_encode = QWEN_IMAGE_SYSTEM_PROMPT + "{}" + QWEN_IMAGE_PROMPT_SUFFIX
    drop_idx = QWEN_IMAGE_DROP_IDX
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
    vl_image_inputs = [resize_to_area(img) for img in images] or None

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


__all__ = [
    "get_qwen_prompt_embeds",
    "get_qwen_prompt_embeds_with_image",
]
