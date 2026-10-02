"""Krea 2 (K2) text encoder: Qwen3-VL-4B conditioner.

Returns the stacked selected hidden states (b, seq, num_select_layers, dim) plus the
attention mask; the layerwise fusion lives inside the DiT.

Loading follows musubi conventions: the model config is vendored in
``thenoise.utils.qwen_configs``, and weights load directly from a local safetensors
file in either the ComfyUI-style ``model.``/``visual.`` or the official HF key
layout.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from thenoise.utils.text_encoder import (
    QWEN25_TOKENIZER_CONFIG_DIR,
    QWEN3_VL_TOKENIZER_OVERRIDES,
    QWEN_VL_DROP_IDX,
    QWEN_VL_PROMPT_SUFFIX,
    QWEN_VL_SYSTEM_PROMPT,
    load_qwen3_vl_model,
    load_qwen3_vl_tokenizer,
)

logger = logging.getLogger(__name__)

# Repo id used when no local tokenizer directory is available.
QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def load_qwen3_vl_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_dir: Optional[str] = None,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
) -> "Qwen3VLConditioner":
    """Load the Qwen3-VL-4B conditioner: weights from ``model_path``, tokenizer from
    ``tokenizer_dir``, else the vendored ``configs/`` directory, else ``tokenizer_repo``."""
    qwen = load_qwen3_vl_model(model_path, dtype=dtype, device=device)
    tokenizer_dir = tokenizer_dir or QWEN25_TOKENIZER_CONFIG_DIR
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(
            f"Krea 2 tokenizer config directory not found at {tokenizer_dir}. "
            "Expected configs/qwen25_tokenizer/ with tokenizer.json and tokenizer_config.json."
        )
    tokenizer, processor = load_qwen3_vl_tokenizer(
        tokenizer_dir, max_length=max_length, overrides=QWEN3_VL_TOKENIZER_OVERRIDES
    )
    conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=max_length, select_layers=select_layers)
    return conditioner


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        qwen: "Qwen3VLForConditionalGeneration",
        tokenizer,
        processor,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
    ):
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = QWEN_VL_SYSTEM_PROMPT
        self.prompt_template_encode_suffix = QWEN_VL_PROMPT_SUFFIX
        self.prompt_template_encode_start_idx = QWEN_VL_DROP_IDX
        self.prompt_template_encode_suffix_start_idx = 5

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(self.qwen.device, non_blocking=True)
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        inputs = self.tokenizer(
            text,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
            return_tensors="pt",
        ).to(self.qwen.device, non_blocking=True)
        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
        states = self.qwen.model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

        hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
        hiddens = hiddens[:, prefix_idx:]
        mask = mask[:, prefix_idx:]

        return hiddens, mask
