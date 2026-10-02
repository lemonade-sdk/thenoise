# Anima tokenize / text-encode strategies

from typing import Any, List, Optional, Union

import torch

from thenoise.dit.anima.strategy_base import TextEncodingStrategy, TokenizeStrategy
from thenoise.utils.text_encoder import (
    QWEN3_06B_TOKENIZER_OVERRIDES,
    load_qwen3_tokenizer,
    load_t5_tokenizer,
)
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

class AnimaTokenizeStrategy(TokenizeStrategy):
    """Dual tokenization: Qwen3 for the text encoder, T5 ids as LLM-Adapter targets."""

    def __init__(
        self,
        qwen3_tokenizer=None,
        t5_tokenizer=None,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
        t5_tokenizer_path: Optional[str] = None,
    ) -> None:
        # Load tokenizers from vendored configs if not provided directly.
        if qwen3_tokenizer is None:
            qwen3_tokenizer = load_qwen3_tokenizer(overrides=QWEN3_06B_TOKENIZER_OVERRIDES)
        if t5_tokenizer is None:
            t5_tokenizer = load_t5_tokenizer(t5_tokenizer_path)

        self.qwen3_tokenizer = qwen3_tokenizer
        self.qwen3_max_length = qwen3_max_length
        self.t5_tokenizer = t5_tokenizer
        self.t5_max_length = t5_max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        qwen3_encoding = self.qwen3_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.qwen3_max_length
        )
        qwen3_input_ids = qwen3_encoding["input_ids"]
        qwen3_attn_mask = qwen3_encoding["attention_mask"]

        # T5 ids are the LLM Adapter's target input
        t5_encoding = self.t5_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.t5_max_length
        )
        t5_input_ids = t5_encoding["input_ids"]
        t5_attn_mask = t5_encoding["attention_mask"]
        return [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncodingStrategy(TextEncodingStrategy):
    """Encode Qwen3 tokens; the T5 ids pass through to the LLM Adapter."""

    def __init__(self) -> None:
        super().__init__()

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """``models``: [qwen3_text_encoder].

        ``tokens``: [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask],
        returned as [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask].
        """
        qwen3_text_encoder = models[0]
        qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = tokens

        encoder_device = qwen3_text_encoder.device

        qwen3_input_ids = qwen3_input_ids.to(encoder_device)
        qwen3_attn_mask = qwen3_attn_mask.to(encoder_device)
        outputs = qwen3_text_encoder(input_ids=qwen3_input_ids, attention_mask=qwen3_attn_mask)
        prompt_embeds = outputs.last_hidden_state
        prompt_embeds[~qwen3_attn_mask.bool()] = 0

        return [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask]
