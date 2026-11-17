"""Krea 2's conditioner — the Qwen3-VL forward the DiT reads its text from.

The DiT takes whatever the conditioner returns, cross-attending to the tokens the mask
keeps, so what these tests pin is the shape of that handover: the system prompt dropped,
padding masked, and a reference arriving as vision tokens with the modality map the
model needs beside them.
"""
from __future__ import annotations

import pytest
import torch

from conftest import tiny_qwen_config

SELECT = (1, 2)
MAX_LENGTH = 48


@pytest.fixture(scope="module")
def tiny_conditioner():
    """The conditioner over a Qwen3-VL shrunk to what a CPU forward costs nothing."""
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration

    from thenoise.dit.krea2.encoder import Qwen3VLConditioner
    from thenoise.utils.text_encoder import (
        QWEN25_TOKENIZER_CONFIG_DIR,
        QWEN3_VL_TOKENIZER_OVERRIDES,
        load_qwen3_vl_processor,
        load_qwen3_vl_tokenizer,
    )

    qwen = Qwen3VLForConditionalGeneration(tiny_qwen_config()).eval().requires_grad_(False)
    tokenizer, _ = load_qwen3_vl_tokenizer(
        QWEN25_TOKENIZER_CONFIG_DIR, max_length=MAX_LENGTH,
        overrides=QWEN3_VL_TOKENIZER_OVERRIDES,
    )
    conditioner = Qwen3VLConditioner(
        qwen, tokenizer, load_qwen3_vl_processor(tokenizer),
        max_length=MAX_LENGTH, select_layers=SELECT,
    )
    return conditioner.eval(), Image.new("RGB", (64, 64), "white")


def test_a_prompt_arrives_as_max_length_tokens_with_the_prefix_dropped(tiny_conditioner):
    """The conditioner pads to its window and cuts the system prompt back off, so the
    handover is exactly ``max_length`` tokens however long the prompt was."""
    conditioner, _ = tiny_conditioner
    with torch.no_grad():
        hiddens, mask = conditioner(["a red fox sitting in the snow"])

    assert hiddens.shape == (1, MAX_LENGTH, len(SELECT), tiny_qwen_config().text_config.hidden_size)
    assert mask.shape == hiddens.shape[:2]
    assert mask[0].sum() < MAX_LENGTH  # the padding is masked, and dropped downstream


def test_a_reference_widens_the_conditioning_and_carries_its_modality_map(tiny_conditioner):
    """The picture's placeholders are expanded before the forward, so its features land
    in the sequence — and a forward that ran without them, or with a modality map one
    token short of the appended suffix, does not get this far.
    """
    conditioner, picture = tiny_conditioner
    with torch.no_grad():
        plain, plain_mask = conditioner(["a red fox"])
        shown, shown_mask = conditioner(["a red fox"], [picture])

    assert shown.shape[2:] == plain.shape[2:]
    assert shown.shape[1] > MAX_LENGTH  # truncation is lifted, the picture has to fit
    assert shown_mask[0].sum() > plain_mask[0].sum()
    assert torch.isfinite(shown).all()


def test_a_picture_is_shown_to_one_prompt_at_a_time(tiny_conditioner):
    """``images`` belongs to the call, and the processor pairs a flat picture list with
    one sequence; a batch would need it per row."""
    conditioner, picture = tiny_conditioner

    with pytest.raises(ValueError, match="one prompt at a time"):
        conditioner(["first", "second"], [picture])
