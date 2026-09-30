"""Mage-Flow text encoder — Qwen3-VL-4B, conditioned on the *normed* last hidden state.

Same conditioner shape as the other Qwen-VL adapters: the LM (vision tower included)
turns ``prompt`` + optional reference images into a ``[1, L, 2560]`` conditioning
tensor the DiT's ``txt_in`` eats. Three things are Mage's own:

  * **which hidden state** — Mage conditions on the LM's last hidden state WITH the
    final RMSNorm applied (ComfyUI's ``layer_norm_hidden_state = True``), i.e. plain
    ``.last_hidden_state``. Qwen-Image 2.1 taps the same layer *before* that norm, so
    it hooks it; here the model's own output is the answer.
  * **what is kept** — only the template prefix is removed (the system turn and the
    ``user`` header, via the shared :func:`compute_drop_idx`: 34 tokens t2i, 64 for
    the edit template's longer system prompt). The reference images' *vision tokens
    stay in the conditioning*: an edit's reference reaches the DiT twice, once here
    as text-stream tokens and once as latent tokens appended to the image stream.
  * **the templates** — the t2i one is the shared
    ``QWEN_VL_SYSTEM_PROMPT``/``QWEN_VL_PROMPT_SUFFIX`` pair verbatim; the edit one
    swaps in the image-instruction system prompt and leads the user turn with one
    ``Image N: <vision>`` block per reference. Neither appends a thinking block: both
    end at the assistant header.

The reference images are downsized for the *conditioning* path only (long edge
capped at :data:`VL_COND_LONG_EDGE`): a full-resolution edit image would inject
thousands of vision tokens and drown the instruction. The VAE reference latent the
DiT gets is untouched by this — it is encoded separately, at full working resolution.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image

from thenoise.utils.image_tensor import resize_to_long_edge
from thenoise.utils.qwen_configs import QWEN3_VL_4B_INSTRUCT_CONFIG
from thenoise.utils.sequence import make_key_padding_mask, pad_to_batch
from thenoise.utils.text_encoder import (
    IM_END,
    IM_START,
    QWEN25_TOKENIZER_CONFIG_DIR,
    QWEN3_VL_TOKENIZER_OVERRIDES,
    QWEN_VL_PROMPT_SUFFIX,
    QWEN_VL_SYSTEM_PROMPT,
    VISION_BLOCK,
    compute_drop_idx,
    find_tokenizer_dir,
    load_qwen3_vl_model,
    load_qwen3_vl_processor,
    load_qwen3_vl_tokenizer,
)

logger = logging.getLogger(__name__)

#: Long-edge cap of the image fed to the vision tower (upstream's ``vl_cond_long_edge``).
VL_COND_LONG_EDGE = 384

#: The edit system prompt: the Qwen-Image-Edit image-instruction prompt.
_EDIT_SYSTEM_PROMPT = (
    IM_START + "system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should alter "
    "or modify the image. Generate a new image that meets the user's requirements "
    "while maintaining consistency with the original input where appropriate."
    + IM_END + "\n"
    + IM_START + "user\n"
)

#: Chat-wrapped prompt templates, with ``{}`` standing for the user content.
MAGE_T2I_TEMPLATE = QWEN_VL_SYSTEM_PROMPT + "{}" + QWEN_VL_PROMPT_SUFFIX
MAGE_EDIT_TEMPLATE = _EDIT_SYSTEM_PROMPT + "{}" + QWEN_VL_PROMPT_SUFFIX


def prompt_template(prompt: str, num_images: int) -> str:
    """The chat-wrapped prompt, with one labelled vision block per reference.

    The user turn leads with ``Image 1: ``, ``Image 2: ``, ... each followed by its
    vision block (the training-time multi-reference layout). An empty prompt becomes
    a single space so the user turn is never empty — an empty turn changes how the
    surrounding markers tokenize.
    """
    if not prompt:
        prompt = " "
    if not num_images:
        return MAGE_T2I_TEMPLATE.format(prompt)
    refs = "".join(f"Image {j}: {VISION_BLOCK}" for j in range(1, num_images + 1))
    return MAGE_EDIT_TEMPLATE.format(refs + prompt)


def _as_image_list(images) -> list:
    """Normalize ``None`` / a single image / a sequence of them to a list."""
    if images is None:
        return []
    return list(images) if isinstance(images, (list, tuple)) else [images]


class MageFlowTextEncoder(nn.Module):
    """Qwen3-VL-4B conditioner: ``(prompt, images) -> (embeddings, mask)``.

    Registered with the memory manager like any other text encoder; the
    tokenizer/processor are pure Python and stay put.
    """

    def __init__(self, qwen: nn.Module, tokenizer, processor) -> None:
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor

    @property
    def dtype(self) -> torch.dtype:
        return next(self.qwen.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    def _to_model(self, value, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(value).to(self.device, dtype)

    def encode(
        self, prompt: str, images: Optional[Union[Image.Image, Sequence[Image.Image]]] = None
    ) -> Tuple[Tensor, Tensor]:
        """Encode one prompt -> ``(embeds [1, L, hidden], mask [1, L])``.

        ``images`` are PIL images in prompt order (single image or list); they are
        capped to :data:`VL_COND_LONG_EDGE` here, so a caller passing a
        full-resolution edit image is not accidentally conditioning on it at that
        size. The mask is all-ones — one sequence is never padded, and it exists to
        report the conditioning length like Qwen-Image's does.
        """
        images = _as_image_list(images)
        condition_images = [resize_to_long_edge(img, VL_COND_LONG_EDGE) for img in images]
        text = prompt_template(prompt, len(condition_images))
        inputs = self.processor(text=[text], images=condition_images or None)

        input_ids = self._to_model(inputs["input_ids"], torch.long)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": self._to_model(inputs["attention_mask"], torch.long),
            "use_cache": False,
        }
        if condition_images:
            model_inputs["pixel_values"] = self._to_model(inputs["pixel_values"], self.dtype)
            model_inputs["image_grid_thw"] = self._to_model(inputs["image_grid_thw"], torch.long)
            # M-RoPE needs the per-token modality map; the model refuses to guess it.
            model_inputs["mm_token_type_ids"] = self._to_model(
                inputs["mm_token_type_ids"], torch.long
            )

        # ``.model`` is the bare Qwen3VLModel (no LM head) and its ``last_hidden_state``
        # is already post-final-RMSNorm — exactly the tensor Mage conditions on.
        hidden = self.qwen.model(**model_inputs).last_hidden_state

        drop = compute_drop_idx(input_ids[0])
        kept, _, seqlens = pad_to_batch([hidden[0, drop:].to(self.dtype)])
        mask = make_key_padding_mask(seqlens, self.device, always=True)
        return kept, mask

    def forward(
        self, prompt: str, images: Optional[Union[Image.Image, Sequence[Image.Image]]] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.encode(prompt, images)


def load_mage_flow_text_encoder(
    path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    tokenizer_dir: Optional[str] = None,
) -> MageFlowTextEncoder:
    """Load the Qwen3-VL-4B conditioner + tokenizer/processor for Mage-Flow.

    ``path`` is a single safetensors file (official HF or ComfyUI layout, bf16 or
    int8-convrot) — the same file Krea 2 loads. The tokenizer defaults to a
    ``tokenizer/`` directory next to the checkpoint, else the vendored Qwen config.
    """
    qwen = load_qwen3_vl_model(
        path, dtype=dtype, device=device, config=QWEN3_VL_4B_INSTRUCT_CONFIG
    )
    tokenizer_dir = (
        tokenizer_dir or find_tokenizer_dir(path) or QWEN25_TOKENIZER_CONFIG_DIR
    )
    tokenizer, _ = load_qwen3_vl_tokenizer(
        tokenizer_dir,
        max_length=QWEN3_VL_4B_INSTRUCT_CONFIG["text_config"]["max_position_embeddings"],
        overrides=QWEN3_VL_TOKENIZER_OVERRIDES,
    )
    encoder = MageFlowTextEncoder(qwen, tokenizer, load_qwen3_vl_processor(tokenizer))
    logger.info("Loaded Mage-Flow text encoder (Qwen3-VL-4B) from %s", path)
    return encoder


__all__ = [
    "MAGE_EDIT_TEMPLATE",
    "MAGE_T2I_TEMPLATE",
    "VL_COND_LONG_EDGE",
    "MageFlowTextEncoder",
    "load_mage_flow_text_encoder",
    "prompt_template",
]
