"""MOSS-Image text encoder — Qwen3.5-9B read at its last hidden state.

The choices here that a checkpoint will not complain about if they are wrong:

  * conditioning on ``.last_hidden_state``, final RMSNorm applied;
  * the ``user`` turn **left open inside a thinking block** — ``<think>\\n`` with no
    closing tag, which is how the model was trained;
  * nothing dropped, nothing padded. The template's tokens are part of the
    conditioning and each prompt keeps its true length: the DiT places the image
    *after* the text, so a padded branch would move the image.

The checkpoint's vision tower and its speculative-decoding (``mtp``) block are loaded
and never run: this is a text-to-image model.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from torch import Tensor, nn

from thenoise.utils.text_encoder import (
    IM_END,
    IM_START,
    find_tokenizer_dir,
    load_qwen35_model,
    load_qwen35_tokenizer,
)

logger = logging.getLogger(__name__)

#: One ``user`` turn, left open in a thinking block.
MOSS_TEMPLATE = IM_START + "user\n{}" + IM_END + "\n" + IM_START + "assistant\n<think>\n"

#: Cap on the rendered prompt.
MAX_SEQUENCE_LENGTH = 4096


def prompt_template(prompt: str) -> str:
    return MOSS_TEMPLATE.format(prompt)


class MossImageTextEncoder(nn.Module):
    """Qwen3.5-9B conditioner: ``prompt -> embeds [1, L, hidden]``."""

    def __init__(self, qwen: nn.Module, tokenizer) -> None:
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer

    @property
    def dtype(self) -> torch.dtype:
        return next(self.qwen.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    def encode(self, prompt: str) -> Tensor:
        """One prompt -> the raw ``[1, L, text_dim]`` conditioning, unpadded."""
        encoded = self.tokenizer(
            prompt_template(prompt),
            add_special_tokens=False,  # the template IS the sequence
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device)
        # ``.model`` stops before the LM head, so its output is the post-RMSNorm
        # conditioning. One sequence per call, hence an all-valid mask.
        out = self.qwen.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )
        return out.last_hidden_state.to(self.dtype)

    def forward(self, prompt: str) -> Tensor:
        return self.encode(prompt)


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
    encoder = MossImageTextEncoder(qwen, tokenizer)
    logger.info("Loaded MOSS-Image text encoder (Qwen3.5-9B) from %s", path)
    return encoder


__all__ = [
    "MAX_SEQUENCE_LENGTH",
    "MOSS_TEMPLATE",
    "MossImageTextEncoder",
    "load_moss_image_text_encoder",
    "prompt_template",
]
