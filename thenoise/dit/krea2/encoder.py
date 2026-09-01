"""Krea 2 (K2) text encoder: Qwen3-VL-4B conditioner.

Returns the stacked selected hidden states (b, seq, num_select_layers, dim) plus the
attention mask; the layerwise fusion lives inside the DiT (TextFusionTransformer), so
the raw stack is what gets cached during training.

Loading follows musubi conventions (cf. qwen_image's load_qwen2_5_vl): the model config
is vendored here so it is built without fetching config.json from the Hub, weights are
loaded directly from a local safetensors file (ComfyUI-style `model.`/`visual.` keys are
accepted as well as the official HF layout), and only the tokenizer is still pulled by
repo id. This lets K2 share the same Qwen3-VL-4B weights a user already has for ComfyUI,
instead of requiring a separate transformers/Diffusers checkpoint.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from thenoise.dit.quantized import replace_linears
from thenoise.utils.loader import load_text_encoder_weights

logger = logging.getLogger(__name__)


# The Qwen3-VL tokenizer config files are vendored in the package under ``configs/``
# (mirroring the ``tokenizer/`` subfolder of the official Qwen/Qwen3-VL-4B-Instruct
# repo). They carry the Qwen chat template used by the caption encoder, so the tokenizer
# loads offline with ``local_files_only=True`` and is never fetched from the Hub.
KREA2_TOKENIZER_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "tokenizer")


# Only the tokenizer is still fetched by repo id (small, HF-cached after first use).
QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Vendored copy of the Qwen3-VL-4B-Instruct config.json so the text encoder is built
# without fetching the config from the Hugging Face Hub. Qwen3-VL is natively supported by
# transformers (no auto_map / remote code), so Qwen3VLConfig.from_dict reproduces
# AutoConfig.from_pretrained exactly. Mirror upstream config.json if Qwen ever revises it.
QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def _convert_comfyui_qwen3vl_state_dict(key: str) -> str:
    """Map a ComfyUI-style (bare ``model.`` / ``visual.``) Qwen3-VL state dict key onto the HF
    ``Qwen3VLForConditionalGeneration`` layout. Official HF checkpoints already use the
    ``model.language_model.`` / ``model.visual.`` layout and pass through unchanged.
    """
    if key.startswith("model.language_model.") or key.startswith("model.visual."):
        return key
    if key.startswith("visual."):
        return "model.visual." + key[len("visual.") :]
    if key.startswith("language_model."):
        return "model." + key
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    return key


def _load_qwen3_vl_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Build Qwen3-VL-4B from the vendored config and load weights from a local safetensors."""
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)
        del model.lm_head
        replace_linears(model)

    logger.info(f"Loading Krea 2 text encoder (Qwen3-VL) weights from {model_path}")
    load_text_encoder_weights(
        model,
        model_path,
        device=device,
        dtype=dtype,
        key_map=_convert_comfyui_qwen3vl_state_dict,
    )
    if dtype is not None:
        model.to(dtype)
    return model.eval().requires_grad_(False)


def load_qwen3_vl_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_dir: Optional[str] = None,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    disable_mmap: bool = True,
) -> "Qwen3VLConditioner":
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``model_path`` (safetensors),
    tokenizer from ``tokenizer_dir`` (a local directory) when given, else from the vendored
    ``configs/tokenizer/`` directory (so no Hub access is needed), else from ``tokenizer_repo``."""
    qwen = _load_qwen3_vl_model(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    tokenizer_dir = tokenizer_dir or KREA2_TOKENIZER_CONFIG_DIR
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(
            f"Krea 2 tokenizer config directory not found at {tokenizer_dir}. "
            "Expected configs/tokenizer/ with tokenizer.json, tokenizer_config.json, "
            "vocab.json and merges.txt."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, max_length=max_length, local_files_only=True)
    processor = Qwen2TokenizerFast.from_pretrained(tokenizer_dir, max_length=max_length, local_files_only=True)
    conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=max_length, select_layers=select_layers)
    return conditioner.eval().requires_grad_(False)


def find_krea2_tokenizer_dir(text_encoder_path: str, max_depth: int = 3) -> Optional[str]:
    """Locate a local ``tokenizer/`` directory near the text encoder file.

    The downloader drops the tokenizer under the output root (``<out>/tokenizer/``)
    while the text encoder lands under ``<out>/text_encoders/``. Search ``max_depth``
    parent directories of the text encoder for a ``tokenizer/`` dir so the tokenizer is
    loaded offline when present. Returns ``None`` to fall back to the vendored
    ``configs/tokenizer/`` directory.
    """
    base = os.path.dirname(os.path.abspath(text_encoder_path))
    for _ in range(max_depth):
        cand = os.path.join(base, "tokenizer")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(base)
        if parent == base:
            break
        base = parent
    return None


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        qwen: Qwen3VLForConditionalGeneration,
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
        self.prompt_template_encode_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
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
