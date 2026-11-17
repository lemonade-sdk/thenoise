"""MOSS-Image text encoder — Qwen3.5-9B read at its last hidden state.

The choices here that a checkpoint will not complain about if they are wrong:

  * conditioning on ``.last_hidden_state``, final RMSNorm applied;
  * the ``user`` turn **left open inside a thinking block** — ``<think>\\n`` with no
    closing tag, which is how the model was trained;
  * nothing dropped, nothing padded. The template's tokens are part of the
    conditioning and each prompt keeps its true length: the DiT places the image
    *after* the text, so a padded branch would move the image.

The speculative-decoding (``mtp``) block is loaded and never runs. The vision tower
runs only when a request carries a reference, and it is then the reference's only
route to the DiT: this architecture has no reference-latent path, so a picture can
only ever be conditioning, never something the run redraws.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from thenoise.utils.text_encoder import (
    IM_END,
    IM_START,
    find_tokenizer_dir,
    load_qwen35_model,
    load_qwen35_processor,
    load_qwen35_tokenizer,
    prompt_with_pictures,
)

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image

logger = logging.getLogger(__name__)

#: One ``user`` turn, left open in a thinking block.
MOSS_TEMPLATE = IM_START + "user\n{}" + IM_END + "\n" + IM_START + "assistant\n<think>\n"

#: Cap on the words of a conditioning sequence. A reference's placeholder run is not
#: part of it: that run is exactly as long as the vision grid needs it, so the only
#: answer to a prompt too long to fit beside one is to say so.
MAX_SEQUENCE_LENGTH = 4096


def prompt_template(prompt: str) -> str:
    return MOSS_TEMPLATE.format(prompt)


class MossImageTextEncoder(nn.Module):
    """Qwen3.5-9B conditioner: ``(prompt, images) -> embeds [1, L, hidden]``."""

    def __init__(self, qwen: nn.Module, processor) -> None:
        super().__init__()
        self.qwen = qwen
        self.processor = processor

    @property
    def dtype(self) -> torch.dtype:
        return next(self.qwen.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    def encode(self, prompt: str, images: Optional[Sequence["Image.Image"]] = None) -> Tensor:
        """One prompt (plus any reference) -> the raw ``[1, L, text_dim]``, unpadded.

        The language model substitutes the vision tower's features for the
        placeholders, so a reference arrives in the conditioning at the position the
        prompt labels it — ahead of the words, which is where the releases that read
        pictures this way put them.
        """
        images = [] if images is None else list(images)
        text = prompt_template(
            prompt_with_pictures(prompt, len(images)) if images else prompt
        )
        inputs = self.processor(
            text=[text],
            images=images or None,
            **({} if images else {"truncation": True,
                                  "max_length": MAX_SEQUENCE_LENGTH}),
        )
        input_ids = torch.as_tensor(inputs["input_ids"], dtype=torch.long,
                                    device=self.device)
        if images and input_ids.shape[-1] > MAX_SEQUENCE_LENGTH:
            # Truncation is off with a picture present — cutting a placeholder run
            # would leave fewer tokens than the tower has features for — so the
            # conditioner's window is the real limit on how many can be shown.
            raise ValueError(
                f"{len(images)} reference image(s) plus this prompt need "
                f"{input_ids.shape[-1]} conditioning tokens; the Qwen3.5 conditioner "
                f"holds {MAX_SEQUENCE_LENGTH}")
        model_inputs = {
            "input_ids": input_ids,
            # One sequence per call, hence an all-valid mask.
            "attention_mask": torch.ones_like(input_ids),
            "use_cache": False,
        }
        if images:
            # ``.model`` stops before the LM head, so its output is the post-RMSNorm
            # conditioning, placeholders and all.
            model_inputs["pixel_values"] = torch.as_tensor(
                inputs["pixel_values"], dtype=self.dtype, device=self.device)
            model_inputs["image_grid_thw"] = torch.as_tensor(
                inputs["image_grid_thw"], dtype=torch.long, device=self.device)
            # M-RoPE needs the per-token modality map to place the vision spans.
            model_inputs["mm_token_type_ids"] = torch.as_tensor(
                inputs["mm_token_type_ids"], dtype=torch.long, device=self.device)
        out = self.qwen.model(**model_inputs)
        return out.last_hidden_state.to(self.dtype)

    def forward(self, prompt: str, images=None):
        return self.encode(prompt, images)


def load_moss_image_text_encoder(
    path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    tokenizer_dir: Optional[str] = None,
) -> MossImageTextEncoder:
    """Load the Qwen3.5-9B conditioner + tokenizer for MOSS-Image.

    The tokenizer defaults to a ``tokenizer/`` directory next to the checkpoint, else
    the vendored one.
    """
    qwen = load_qwen35_model(path, dtype=dtype, device=device)
    tokenizer = load_qwen35_tokenizer(tokenizer_dir or find_tokenizer_dir(path))
    encoder = MossImageTextEncoder(qwen, load_qwen35_processor(tokenizer))
    logger.info("Loaded MOSS-Image text encoder (Qwen3.5-9B) from %s", path)
    return encoder


__all__ = [
    "MAX_SEQUENCE_LENGTH",
    "MOSS_TEMPLATE",
    "MossImageTextEncoder",
    "load_moss_image_text_encoder",
    "prompt_template",
]
