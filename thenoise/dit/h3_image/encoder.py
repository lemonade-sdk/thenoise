"""The H3-Image conditioner — Qwen3-VL-32B truncated to 50 layers, no final norm.

Two constraints of the release shape it. The prompt carries no chat wrapper: the model
was trained on raw text with labelled vision blocks spliced in, so a system or user turn
moves the conditioning off its distribution — and with no prefix to drop, every token is
conditioning. And the cut took the final RMSNorm with it: the DiT reads raw layer output,
where an RMSNorm would fix a magnitude the DiT uses.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from transformers import Qwen3VLForConditionalGeneration

from thenoise.dit.quantized import replace_linears
from thenoise.utils.qwen_configs import QWEN3_VL_32B_H3_CONFIG
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key
from thenoise.utils.text_encoder import (
    QWEN25_TOKENIZER_CONFIG_DIR,
    QWEN3_VL_TOKENIZER_OVERRIDES,
    VISION_BLOCK,
    find_tokenizer_dir,
    load_qwen3_vl_model,
    load_qwen3_vl_processor,
    load_qwen3_vl_tokenizer,
)

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image

logger = logging.getLogger(__name__)

#: The label in front of each reference, 1-based, matching how the prompt refers to
#: them. ``<Picture n>: `` with angle brackets belongs to the video models.
PICTURE_LABEL = "Picture {}: "

#: Modality tag of a DiT-modulation group: 0 is visual, 1 is text.
TAG_VISUAL, TAG_TEXT = 0, 1

#: The token id the language model substitutes every expanded vision patch with, and
#: the pad token a wholly empty prompt falls back to.
IMAGE_PAD_ID = QWEN3_VL_32B_H3_CONFIG["image_token_id"]
PAD_TOKEN_ID = QWEN3_VL_32B_H3_CONFIG["text_config"]["bos_token_id"]


def prompt_with_pictures(prompt: str, num_images: int) -> str:
    """The conditioning text: one labelled vision block per reference, then the prompt."""
    labels = "".join(PICTURE_LABEL.format(i + 1) + VISION_BLOCK
                     for i in range(num_images))
    return labels + prompt


def vision_token_tags(input_ids: torch.Tensor, image_pad_id: int = IMAGE_PAD_ID
                      ) -> Tensor:
    """Per-token modality tags: 0 across each vision block, 1 everywhere else.

    Each run of ``image_pad`` is widened by one token on either side: the markers
    framing a picture describe the picture, and the DiT modulates them visually.
    """
    ids = input_ids.view(-1).tolist()
    tags = [TAG_TEXT] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i] != image_pad_id:
            i += 1
            continue
        start = i
        while i < len(ids) and ids[i] == image_pad_id:
            i += 1
        for j in range(max(0, start - 1), min(len(ids), i + 1)):
            tags[j] = TAG_VISUAL
    return torch.tensor(tags, dtype=torch.long)


def _truncate_qwen3_vl(config, layers: int) -> nn.Module:
    """A Qwen3-VL with ``layers`` decoder layers and no final norm.

    The depth is a property of the released file, not of the vendored config, and the
    norm is replaced rather than deleted because the language model's forward calls it.
    """
    config.text_config.num_hidden_layers = layers
    model = Qwen3VLForConditionalGeneration._from_config(config)
    del model.lm_head
    model.model.language_model.norm = nn.Identity()
    replace_linears(model)
    return model


def layer_count(path: str) -> int:
    """How many decoder layers a text-encoder file holds, from its header."""
    with MemoryEfficientSafeOpen(path) as f:
        indices = {
            int(k.split(".")[2])
            for k in map(unwrap_key, f.keys())
            if k.startswith("model.layers.") and k.split(".")[2].isdigit()
        }
    if not indices:
        raise ValueError(f"{path} has no Qwen3-VL decoder layers")
    return max(indices) + 1


class H3ImageTextEncoder(nn.Module):
    """Qwen3-VL-32B conditioner: ``(prompt, images) -> (embeds, mask, tags)``."""

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

    def encode(
        self, prompt: str, images: Optional[Sequence["Image.Image"]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """``(embeds [1, L, 5120], mask [1, L], tags [1, L])`` for one prompt.

        ``images`` are the already fitted PIL images, in the order the prompt labels
        them. One sequence is encoded at a time, so ``mask`` is all ones.
        """
        images = [] if images is None else list(images)
        text = prompt_with_pictures(prompt, len(images))
        inputs = self.processor(text=[text], images=images or None)

        input_ids = torch.as_tensor(inputs["input_ids"]).to(self.device, torch.long)
        if input_ids.shape[-1] == 0:
            # An empty prompt with no references tokenizes to nothing at all.
            input_ids = torch.full((1, 1), PAD_TOKEN_ID, dtype=torch.long, device=self.device)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "use_cache": False,
        }
        if images:
            model_inputs["pixel_values"] = torch.as_tensor(inputs["pixel_values"]).to(
                self.device, self.dtype
            )
            model_inputs["image_grid_thw"] = torch.as_tensor(inputs["image_grid_thw"]).to(
                self.device, torch.long
            )
            # M-RoPE needs the per-token modality map to place the vision spans.
            model_inputs["mm_token_type_ids"] = torch.as_tensor(
                inputs["mm_token_type_ids"]
            ).to(self.device, torch.long)

        # The truncated model ends in an identity: nothing normalizes this output.
        embeds = self.qwen.model(**model_inputs).last_hidden_state
        tags = vision_token_tags(input_ids[0]).view(input_ids.shape).to(self.device)
        mask = torch.ones_like(input_ids)
        return embeds.to(self.dtype), mask, tags

    def forward(self, prompt: str, images=None):
        return self.encode(prompt, images)


def load_h3_image_text_encoder(
    path: str,
    *,
    dtype: torch.dtype,
    device,
    tokenizer_dir: Optional[str] = None,
) -> H3ImageTextEncoder:
    """Load the truncated Qwen3-VL-32B conditioner plus its tokenizer and processor."""
    layers = layer_count(path)
    qwen = load_qwen3_vl_model(
        path,
        dtype=dtype,
        device=device,
        config=QWEN3_VL_32B_H3_CONFIG,
        model_factory=lambda config: _truncate_qwen3_vl(config, layers),
    )
    tokenizer_dir = tokenizer_dir or find_tokenizer_dir(path) or QWEN25_TOKENIZER_CONFIG_DIR
    tokenizer, _ = load_qwen3_vl_tokenizer(
        tokenizer_dir,
        max_length=QWEN3_VL_32B_H3_CONFIG["text_config"]["max_position_embeddings"],
        overrides=QWEN3_VL_TOKENIZER_OVERRIDES,
    )
    encoder = H3ImageTextEncoder(qwen, tokenizer, load_qwen3_vl_processor(tokenizer))
    logger.info("Loaded H3-Image text encoder (Qwen3-VL-32B, %d layers) from %s",
                layers, path)
    return encoder


__all__ = [
    "PICTURE_LABEL",
    "H3ImageTextEncoder",
    "layer_count",
    "load_h3_image_text_encoder",
    "prompt_with_pictures",
    "vision_token_tags",
]
