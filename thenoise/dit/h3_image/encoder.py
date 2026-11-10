"""The H3-Image conditioner — Qwen3-VL-32B truncated to 50 layers, no final norm.

Two things set this presentation apart from every other Qwen-VL conditioner here.

**Nothing wraps the prompt.** The released model is conditioned on raw text with
vision blocks spliced in — one ``Picture n: `` label per reference, in prompt order,
then the prompt — with none of the system-turn/user-turn scaffolding the other
conditioners build. That is not a stylistic choice to tidy away: the language model
was trained on exactly this distribution, so adding a chat wrapper moves the whole
conditioning off it. It also means there is no prefix to drop, so the entire token
sequence, first marker to last, is the conditioning.

**What is returned is unnormalized.** The file is a 32B decoder cut to its first 50
layers, and the final RMSNorm went with the cut: the checkpoint has no
``...norm.weight`` at all, and the DiT's ``condition_proj`` expects raw layer output.
Hence the identity swap in :func:`_truncate_qwen3_vl` rather than a norm that
happens to be initialised to ones — the difference is visible in the output, because
an RMSNorm renormalises every token to a fixed magnitude and the prompt's magnitude
carries information the DiT reads.

The layer count is read off the file, so a release truncated somewhere else loads.

Per-token **modality tags** come back with the embeddings: the vision placeholders
inside the prompt carry the visual tag, everything else the text tag, and the DiT
modulates the two differently (see
:mod:`thenoise.dit.h3_image.models`). A tag is a property of the *prompt*, so it
travels with the conditioning rather than being recomputed per step.
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
#: them. ``<Picture n>: `` with angle brackets belongs to the video models; the image
#: release spells it without.
PICTURE_LABEL = "Picture {}: "

#: Modality tag of a DiT-modulation group: 0 is visual, 1 is text.
TAG_VISUAL, TAG_TEXT = 0, 1

#: The token id the language model substitutes every expanded vision patch with, and
#: the pad token a wholly empty prompt falls back to.
IMAGE_PAD_ID = QWEN3_VL_32B_H3_CONFIG["image_token_id"]
PAD_TOKEN_ID = QWEN3_VL_32B_H3_CONFIG["text_config"]["bos_token_id"]


def prompt_with_pictures(prompt: str, num_images: int) -> str:
    """The conditioning text: one labelled vision block per reference, then the prompt.

    An empty prompt with no references tokenizes to nothing at all, which no encoder
    runs; it is left empty here and given the pad token by
    :meth:`H3ImageTextEncoder.encode`, so the emptiness is decided once.
    """
    labels = "".join(PICTURE_LABEL.format(i + 1) + VISION_BLOCK
                     for i in range(num_images))
    return labels + prompt


def vision_token_tags(input_ids: torch.Tensor, image_pad_id: int = IMAGE_PAD_ID
                      ) -> Tensor:
    """Per-token modality tags: 0 across each vision block, 1 everywhere else.

    A block is the run of expanded image patches *and* the two markers framing it, so
    each run of ``image_pad`` is widened by one token on either side — the framing
    tokens describe the picture, not the sentence, and the DiT's visual modulation is
    what the reference model applies to them.
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

    The truncation is a property of the released file, not of the vendored config, so
    the depth is set here; the norm is replaced rather than deleted because the
    language model's forward calls it.
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
        """Encode one prompt -> ``(embeds [1, L, 5120], mask [1, L], tags [1, L])``.

        The images are the *already fitted* PIL images, in the order the prompt labels
        them. ``mask`` marks real tokens: the batch is one sequence, so it is all ones
        and the DiT's sequence length is simply ``L``.
        """
        images = [] if images is None else list(images)
        text = prompt_with_pictures(prompt, len(images))
        inputs = self.processor(text=[text], images=images or None)

        input_ids = torch.as_tensor(inputs["input_ids"]).to(self.device, torch.long)
        if input_ids.shape[-1] == 0:
            # An empty prompt with no references tokenizes to nothing, which no
            # language model runs; the pad token is the release's own fallback.
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

        # The truncated model ends in an identity, so this *is* the unnormalized output
        # of the last surviving layer.
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
    """Load the truncated Qwen3-VL-32B conditioner plus its tokenizer and processor.

    The tokenizer is the shared Qwen BPE; the release adds no tokens the image
    presentation needs, so no special-token surgery happens at load.
    """
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
