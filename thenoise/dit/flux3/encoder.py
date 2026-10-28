"""Flux.3 text encoder — Qwen3-VL-4B read at eight taps, prompt padding kept.

The LM turns ``prompt`` into the ``[1, L, 20480]`` conditioning the DiT's ``txt_in``
eats. Flux.3's own choices, all of them load-bearing:

  * the context is the **concatenation of eight hidden states** — layers
    4, 8, ... , 32, none of them finally normed — so ``8 x 2560 = 20480`` channels;
  * the template is the plain ``user`` turn with **no thinking block** and no system
    prompt, and the prompt is never truncated;
  * the sequence is **right-padded with ``<|endoftext|>`` out to a multiple of 80
    tokens**, and that padding is part of the conditioning: the DiT attends to every
    text token and has no mask for it. The pads attend to the prompt too — the
    reference gives them attention weight 1.0, and causality keeps the prompt clean of
    them — so they are real forward outputs rather than inert fillers.
  * references do NOT go through here — Flux.3 conditions on reference *latents*
    (see the adapter), and its reference positions are told apart on the RoPE ``l``
    axis.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from einops import rearrange
from torch import Tensor, nn

from thenoise.utils.qwen_configs import QWEN3_VL_4B_INSTRUCT_CONFIG
from thenoise.utils.sequence import pad_len_to_multiple
from thenoise.utils.text_encoder import (
    IM_END,
    IM_START,
    QWEN25_TOKENIZER_CONFIG_DIR,
    QWEN3_VL_TOKENIZER_OVERRIDES,
    find_tokenizer_dir,
    load_qwen3_vl_model,
    load_qwen3_vl_tokenizer,
)

logger = logging.getLogger(__name__)

#: Qwen3-VL hidden layers whose outputs are concatenated into the DiT context.
OUTPUT_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]

#: ``<|endoftext|>``, the id the sequence is aligned with.
PAD_TOKEN_ID = 151643

#: Alignment of the (attended) prompt sequence.
PAD_MULTIPLE = 80

#: Chat-wrapped prompt template: one ``user`` turn, no system prompt, no think block.
FLUX3_TEMPLATE = IM_START + "user\n{}" + IM_END + "\n" + IM_START + "assistant\n"


def prompt_template(prompt: str) -> str:
    """The chat-wrapped prompt."""
    return FLUX3_TEMPLATE.format(prompt)


class Flux3TextEncoder(nn.Module):
    """Qwen3-VL-4B conditioner: ``prompt -> embeds [1, L, 8 * hidden]``."""

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
        """One prompt -> the raw ``[1, L, context_in_dim]`` conditioning (padding included)."""
        encoded = self.tokenizer(prompt_template(prompt), add_special_tokens=False)
        length = len(encoded["input_ids"])
        pad = pad_len_to_multiple(length, PAD_MULTIPLE) - length

        input_ids = torch.cat(
            [
                torch.tensor([encoded["input_ids"]], dtype=torch.long),
                torch.full((1, pad), PAD_TOKEN_ID, dtype=torch.long),
            ],
            dim=1,
        )
        # Nothing is masked: the reference weights the padding 1.0, so the pads read
        # the prompt (causality keeps the prompt from reading them) and the DiT gets a
        # full padded block. The position sequence is then a plain count — the
        # ``(attention_mask.cumsum(-1) - 1)`` the pinned transformers computes for it,
        # which would park *masked* padding on rope position 1.
        attention_mask = torch.ones(1, length + pad, dtype=torch.long)
        position_ids = attention_mask.cumsum(-1) - 1

        device = self.device
        # ``.model`` is the bare Qwen3VLModel: the taps are the pre-final-norm decoder
        # outputs, which is what the DiT was trained on. A 2D ``position_ids`` is
        # broadcast over the three M-RoPE axes.
        out = self.qwen.model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            position_ids=position_ids.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        taps = torch.stack([out.hidden_states[i] for i in OUTPUT_LAYERS], dim=1)
        return rearrange(taps, "b c l d -> b l (c d)").to(self.dtype)

    def forward(self, prompt: str) -> Tensor:
        return self.encode(prompt)


def load_flux3_text_encoder(
    path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    tokenizer_dir: Optional[str] = None,
) -> Flux3TextEncoder:
    """Load the Qwen3-VL-4B conditioner + tokenizer for Flux.3.

    ``path`` is a single safetensors file (official HF or ComfyUI layout, bf16 or
    int8-convrot). The tokenizer defaults to a ``tokenizer/`` directory next to the
    checkpoint, else the vendored Qwen config. The vision tower is loaded but never
    run: Flux.3 feeds references to the DiT as latents.
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
    encoder = Flux3TextEncoder(qwen, tokenizer)
    logger.info("Loaded Flux.3 text encoder (Qwen3-VL-4B) from %s", path)
    return encoder


__all__ = [
    "FLUX3_TEMPLATE",
    "OUTPUT_LAYERS",
    "PAD_MULTIPLE",
    "PAD_TOKEN_ID",
    "Flux3TextEncoder",
    "load_flux3_text_encoder",
    "prompt_template",
]
