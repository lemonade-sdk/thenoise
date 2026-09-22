"""Qwen-Image 2.1 text encoder — Qwen3-VL-8B, vision tokens in, image slots out.

Qwen-Image 2.1 is conditioned by a *full* Qwen3-VL-8B: the LM **and** its vision
tower. An edit therefore feeds the reference image in twice, and both halves have to
agree on where it sits: the language model sees it as vision tokens spliced into the
prompt, and the DiT replaces exactly those positions with the reference VAE latent.

So the encoder returns, per conditioning branch, the prompt embeddings with the
system turn *and* the vision tokens removed, plus ``image_slots`` — the token index
each removed image left behind, where
:meth:`thenoise.dit.qwen_image21.models.QwenImage21Transformer2DModel.build_sequence`
splices the latent back in.

Two details are load-bearing for matching the reference implementation:

  * **The hidden state is the last layer's output BEFORE the final RMSNorm** (the
    model is tuned on that; ``hidden_states[-1]`` is already normalised on current
    transformers), so it is captured with a forward-pre-hook on the LM's final norm.
  * **The system turn is dropped and the user turn starts the conditioning**: only
    everything from the second chat-start marker onwards reaches the DiT.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from thenoise.utils.qwen_configs import QWEN3_VL_8B_INSTRUCT_CONFIG
from thenoise.utils.text_encoder import (
    QWEN25_TOKENIZER_CONFIG_DIR,
    QWEN3_VL_TOKENIZER_OVERRIDES,
    find_tokenizer_dir,
    load_qwen3_vl_model,
    load_qwen3_vl_processor,
    load_qwen3_vl_tokenizer,
)

logger = logging.getLogger(__name__)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"

#: One reference image, as the language model sees it: a single (expanded) image pad
#: between the vision markers.
VISION_BLOCK = VISION_START + IMAGE_PAD + VISION_END

#: The turn the model is conditioned with, and the system prompt that precedes it
#: (dropped before the embeddings reach the DiT).
SYSTEM_PROMPT = IM_START + "system\nComprehend and analyze the provided prompt." + IM_END + "\n"
PROMPT_SUFFIX = IM_END + "\n" + IM_START + "assistant\n"
T2I_TEMPLATE = SYSTEM_PROMPT + IM_START + "user\n{}" + PROMPT_SUFFIX


def prompt_template(prompt: str, num_images: int) -> str:
    """The chat-wrapped prompt, with one labelled vision block per reference.

    The images lead the user turn as ``<image1>``, ``<image2>``, ... each followed by its
    vision block. An empty prompt becomes a single space, so the user turn is never
    empty (which would change the tokenisation of the markers around it).
    """
    if not prompt:
        prompt = " "
    refs = " ".join(f"<image{i + 1}>{VISION_BLOCK}" for i in range(num_images))
    return (T2I_TEMPLATE.replace("{}", refs + "{}", 1) if refs else T2I_TEMPLATE).format(prompt)


def keep_mask_and_slots(
    input_ids: Tensor,
    mm_token_type_ids: Tensor,
    *,
    im_start_id: int,
) -> Tuple[Tensor, List[int]]:
    """Which prompt tokens condition the DiT, and where each image's slot lands.

    Two kinds of token are removed, exactly as the reference does:

      * everything before the SECOND ``im_start`` — the system turn (the first is
        the user turn's, so with only one marker nothing is dropped); and
      * each contiguous run of image tokens, which the DiT replaces with the
        reference latent.

    A slot is the index the removed run *used* to start at, counted in the kept
    tokens before it. Runs are identified by ``mm_token_type_ids`` (modality 1 =
    image), which is what the processor itself uses to expand them.

    ``input_ids``/``mm_token_type_ids`` are one sequence (no batch axis).
    """
    starts = (input_ids == im_start_id).nonzero().flatten()
    first_user_turn = int(starts[1]) if starts.numel() > 1 else 0

    keep = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    keep[:first_user_turn] = False

    image = mm_token_type_ids == 1
    slots: List[int] = []
    run = 0
    for i, is_image in enumerate(image.tolist() + [False]):
        if is_image and not run:
            run = i
        elif run and not is_image:
            keep[run:i] = False
            slots.append(int(keep[:run].count_nonzero()))
            run = 0
    return keep, slots


class QwenImage21TextEncoder(nn.Module):
    """Qwen3-VL-8B prompt encoder: ``(prompt, images) -> (embeddings, slots)``.

    Registered with the memory manager as any other text encoder; the
    tokenizer/processor are pure Python and stay put.
    """

    def __init__(self, qwen: nn.Module, tokenizer, processor) -> None:
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor
        self.im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
        if self.im_start_id is None:
            raise ValueError(
                "the Qwen tokenizer has no chat-start token id; the vendored "
                f"{QWEN25_TOKENIZER_CONFIG_DIR!r} tokenizer is incomplete"
            )
        self._pre_norm: Optional[Tensor] = None
        final_norm = qwen.model.language_model.norm
        final_norm.register_forward_pre_hook(self._capture_pre_norm)

    # ------------------------------------------------------------------ internals
    def _capture_pre_norm(self, _module: nn.Module, args: Tuple[Tensor, ...]) -> None:
        """Remember the LM's last layer output, i.e. the norm's *input*."""
        self._pre_norm = args[0]

    @property
    def dtype(self) -> torch.dtype:
        return next(self.qwen.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    def _to_model(self, value, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(value).to(self.device, dtype)

    # --------------------------------------------------------------------- public
    @torch.no_grad()
    def encode(
        self, prompt: str, images: Optional[Sequence] = None
    ) -> Tuple[Tensor, List[int]]:
        """Encode one prompt -> ``([1, L, hidden], slots)``.

        ``images`` are PIL images in prompt order (already at the size the VAE saw).
        The system turn and the vision tokens are removed, so ``L`` is the
        conditioning length the DiT's sequence is built from.
        """
        images = list(images or [])
        text = prompt_template(prompt, len(images))
        inputs = self.processor(text=[text], images=images or None)

        input_ids = self._to_model(inputs["input_ids"], torch.long)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": self._to_model(inputs["attention_mask"], torch.long),
            # M-RoPE needs the per-token modality map; the model refuses to guess it.
            "mm_token_type_ids": self._to_model(inputs["mm_token_type_ids"], torch.long),
            "use_cache": False,
        }
        if images:
            model_inputs["pixel_values"] = self._to_model(inputs["pixel_values"], self.dtype)
            model_inputs["image_grid_thw"] = self._to_model(inputs["image_grid_thw"], torch.long)

        # ``.model`` is the bare Qwen3VLModel (no LM head): its last layer output,
        # captured before the final RMSNorm, is the conditioning.
        self.qwen.model(**model_inputs)
        hidden = self._pre_norm
        self._pre_norm = None
        if hidden is None:
            raise RuntimeError("the text encoder's final norm never ran")

        keep, slots = keep_mask_and_slots(
            input_ids[0], model_inputs["mm_token_type_ids"][0], im_start_id=self.im_start_id
        )
        return hidden[:, keep.to(hidden.device)], slots

    def forward(self, prompt: str, images: Optional[Sequence] = None) -> Tuple[Tensor, List[int]]:
        return self.encode(prompt, images)


def load_qwen_image21_text_encoder(
    path: str,
    *,
    dtype: torch.dtype,
    device,
    tokenizer_dir: Optional[str] = None,
) -> QwenImage21TextEncoder:
    """Load the Qwen3-VL-8B conditioner + tokenizer/processor for Qwen-Image 2.1.

    ``path`` is a single safetensors file (official HF or ComfyUI layout, bf16 or
    int8-convrot). The tokenizer defaults to the vendored Qwen config directory, or
    to a ``tokenizer/`` directory next to the checkpoint.
    """
    qwen = load_qwen3_vl_model(
        path, dtype=dtype, device=device, config=QWEN3_VL_8B_INSTRUCT_CONFIG
    )
    tokenizer_dir = (
        tokenizer_dir
        or find_tokenizer_dir(path)
        or QWEN25_TOKENIZER_CONFIG_DIR
    )
    tokenizer, _ = load_qwen3_vl_tokenizer(
        tokenizer_dir,
        max_length=QWEN3_VL_8B_INSTRUCT_CONFIG["text_config"]["max_position_embeddings"],
        overrides=QWEN3_VL_TOKENIZER_OVERRIDES,
    )
    encoder = QwenImage21TextEncoder(qwen, tokenizer, load_qwen3_vl_processor(tokenizer))
    logger.info("Loaded Qwen-Image 2.1 text encoder (Qwen3-VL-8B) from %s", path)
    return encoder.eval().requires_grad_(False)


__all__ = [
    "QwenImage21TextEncoder",
    "keep_mask_and_slots",
    "load_qwen_image21_text_encoder",
    "prompt_template",
]
