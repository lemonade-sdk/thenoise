"""Krea 2 (K2) text encoder: Qwen3-VL-4B conditioner.

Returns the stacked selected hidden states (b, seq, num_select_layers, dim) plus the
attention mask; the layerwise fusion lives inside the DiT (TextFusionTransformer), so
the raw stack is what gets cached during training.

Loading follows musubi conventions (cf. qwen_image's load_qwen2_5_vl): the model config
is vendored in ``thenoise.utils.qwen_configs`` so it is built without fetching
config.json from the Hub, weights are loaded directly from a local safetensors file
(ComfyUI-style ``model.``/``visual.`` keys are accepted as well as the official HF
layout), and the model + tokenizer are loaded through ``thenoise.utils.text_encoder``.
This lets K2 share the same Qwen3-VL-4B weights a user already has for ComfyUI,
instead of requiring a separate transformers/Diffusers checkpoint.
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
    load_qwen3_vl_processor,
    load_qwen3_vl_tokenizer,
)

logger = logging.getLogger(__name__)

# Only the tokenizer is still fetched by repo id (small, HF-cached after first use).
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
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``model_path`` (safetensors),
    tokenizer from ``tokenizer_dir`` (a local directory) when given, else from the vendored
    ``configs/`` directory (so no Hub access is needed), else from ``tokenizer_repo``."""
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
    # Build a real Qwen3-VL processor (image/video capable) for the image-grounded
    # instruction encode; the text-only path still works through the same object.
    processor = load_qwen3_vl_processor(processor)
    conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=max_length, select_layers=select_layers)
    return conditioner.eval().requires_grad_(False)


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
        self.qwen = qwen.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = QWEN_VL_SYSTEM_PROMPT
        self.prompt_template_encode_suffix = QWEN_VL_PROMPT_SUFFIX
        self.prompt_template_encode_start_idx = QWEN_VL_DROP_IDX
        self.prompt_template_encode_suffix_start_idx = 5

        # Image-grounded instruction template (the Krea 2 edit semantic path), ported from
        # comfyui-krea2edit's ``Krea2EditGroundedEncode``: the reference image is fed as
        # vision tokens so the encoder *sees* the image while reading the instruction.
        self.grounded_system = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
            "texture, quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.grounded_suffix = "<|im_end|>\n<|im_start|>assistant\n"

    def _grounded_template(self, nimg: int) -> str:
        """Image-grounded instruction template (identity-edit ``fit``).

        Prefixes each vision-token group with the bare ``<|vision_start|>...`` marker.
        """
        vis = "<|vision_start|><|image_pad|><|vision_end|>" * nimg
        return self.grounded_system + vis + "{}" + self.grounded_suffix

    def _prep_image(self, image, grounding_px: int):
        """Cap the reference image fed to the vision encoder.

        ``grounding_px`` caps the longest side (0 = native); never upscales.
        """
        if not grounding_px:
            return image
        w, h = image.size
        if max(w, h) > grounding_px:
            s = grounding_px / max(w, h)
            return image.resize((round(w * s), round(h * s)))
        return image

    def _forward_grounded(
        self,
        text: list[str],
        images,
        grounding_px: int,
    ) -> tuple[Tensor, Tensor]:
        """Image-grounded encode: run the full Qwen3-VL with the reference image as vision
        tokens and tap the same selected layers. Returns ``(B, seq, 12, dim)`` + mask.
        """
        template = self._grounded_template(len(images))
        texts = [template.format(item) for item in text]
        prepped = [self._prep_image(img, grounding_px) for img in images]
        inputs = self.processor(text=texts, images=prepped, return_tensors="pt").to(
            self.qwen.device, non_blocking=True
        )
        with torch.no_grad():
            outputs = self.qwen.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=inputs["mm_token_type_ids"],
                output_hidden_states=True,
            )
            hiddens = torch.stack(
                [outputs.hidden_states[i] for i in self.select_layers], dim=2
            )
        # After the vision-token expansion every token is valid for a single sample
        # (no padding); build an all-True key-padding mask of the expanded length.
        mask = torch.ones(
            hiddens.shape[0], hiddens.shape[1], device=hiddens.device, dtype=torch.bool
        )
        return hiddens, mask

    def forward(
        self,
        text: list[str],
        images=None,
        *,
        grounding_px: int = 768,
    ) -> tuple[Tensor, Tensor]:
        """Encode prompts (text-only, or grounded on ``images``).

        ``images`` is a single reference image (or list). When given, the instruction is
        encoded together with the image as vision tokens (the Krea 2 edit semantic path);
        otherwise the fast text-only path is used (unchanged).
        """
        if images is not None:
            return self._forward_grounded(text, images, grounding_px)
        return self._forward_text(text)

    def _forward_text(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(self.qwen.device, non_blocking=True)
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        with torch.no_grad():
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
