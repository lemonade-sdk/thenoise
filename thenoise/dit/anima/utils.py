# Anima model loading/saving utilities

import os
from typing import Dict, List, Optional, Union
import torch
from safetensors.torch import load_file
from accelerate import init_empty_weights

from thenoise.dit.anima import models as anima_models
from thenoise.utils.int8 import load_int8_if_present
from thenoise.utils.safetensors import WRAP_PREFIXES, load_dit_safetensors
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _count_anima_blocks(dit_path: str) -> int:
    """Count the number of main transformer blocks in an Anima checkpoint.

    Anima comes in variants with different block counts (the base 2.1B has 28
    blocks, the 2.9B "tune" has 40) while otherwise sharing the same
    architecture (model_channels / num_heads unchanged). Count ``blocks.{i}.``
    keys from the safetensors header -- after stripping generic wrap prefixes
    so raw (``net.``) and repackaged (``model.diffusion_model.``) checkpoints
    count identically -- rather than hardcoding a single block count.
    """
    from thenoise.utils.safetensors import MemoryEfficientSafeOpen

    indices = set()
    with MemoryEfficientSafeOpen(dit_path) as f:
        for key in f.keys():
            for prefix in WRAP_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if key.startswith("blocks."):
                try:
                    indices.add(int(key.split(".")[1]))
                except (ValueError, IndexError):
                    pass
    if not indices:
        raise ValueError(f"could not find any 'blocks.*' keys in {dit_path}; is this an Anima DiT?")
    return max(indices) + 1


def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    loading_device: Optional[Union[str, torch.device]] = None,
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> anima_models.Anima:
    """
    Load Anima model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict). if not None, model weights will be casted to this dtype.
    """
    device = torch.device(device)
    loading_device = torch.device(device) if loading_device is None else torch.device(loading_device)

    # The block count varies by checkpoint (base 2.1B = 28, 2.9B tune = 40),
    # so derive it from the checkpoint instead of hardcoding.
    num_blocks = _count_anima_blocks(dit_path)
    logger.info("Detected Anima DiT with %d transformer blocks", num_blocks)

    # We currently support fixed DiT config for Anima models
    dit_config = {
        "max_img_h": 512,
        "max_img_w": 512,
        "max_frames": 128,
        "in_channels": 16,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "model_channels": 2048,
        "concat_padding_mask": True,
        "crossattn_emb_channels": 1024,
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "num_blocks": num_blocks,
        "num_heads": 16,
        "extra_per_block_abs_pos_emb": False,
        "rope_h_extrapolation_ratio": 4.0,
        "rope_w_extrapolation_ratio": 4.0,
        "rope_t_extrapolation_ratio": 1.0,
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        "use_llm_adapter": True,
    }
    with init_empty_weights():
        model = anima_models.Anima(**dit_config)
        if dit_weight_dtype is not None:
            # Casts every init-time parameter to the target dtype. For INT8
            # checkpoints this is still correct: the int8 qweight/scale buffers
            # are created later by load_int8_if_present, so they are untouched.
            model.to(dit_weight_dtype)

    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

    if load_int8_if_present(model, dit_path, device=loading_device, dtype=dit_weight_dtype):
        return model

    # BF16 path: cast to the requested dtype and load via load_state_dict.
    sd = load_dit_safetensors(
        dit_path,
        device=loading_device,
        disable_mmap=True,
        dtype=dit_weight_dtype,
    )

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [
            k
            for k in missing
            if not any(buf_name in k for buf_name in ("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq"))
        ]
        if unexpected_missing:
            # Raise error to avoid silent failures
            raise RuntimeError(
                f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}"
            )
        missing = {}  # all missing keys were expected
    if unexpected:
        # Raise error to avoid silent failures
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Move the whole model (including buffers not present in the checkpoint, e.g. the
    # RoPE position-embedding buffers seq/dim_spatial_range/dim_temporal_range) onto the
    # loading device. Under init_empty_weights() those buffers are created on meta, and
    # load_state_dict(assign=True) only replaces keys present in the checkpoint, so they
    # would otherwise stay off-device and break rotary attention on the GPU.
    if loading_device.type != "cpu":
        model.to(loading_device)
    logger.info("Loaded DiT model from %s", dit_path)

    return model


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3 tokenizer only (without the text encoder model).

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file.
                     If a directory, loads tokenizer from it directly.
                     If a file, uses configs/qwen3_06b/ for tokenizer config.
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(__file__), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qwen3_text_encoder(
    qwen3_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    lora_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
):
    """Load Qwen3-0.6B text encoder.

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file
        dtype: Model dtype
        device: Device to load to

    Returns:
        (text_encoder_model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        # Directory with full model
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(qwen3_path, torch_dtype=dtype, local_files_only=True).model
    else:
        # Single safetensors file - use configs/qwen3_06b/ for config
        config_dir = os.path.join(os.path.dirname(__file__), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        # Load weights
        if qwen3_path.endswith(".safetensors"):
            if lora_weights is None:
                state_dict = load_file(qwen3_path, device="cpu")
            else:
                state_dict = load_safetensors_with_lora(
                    model_files=qwen3_path,
                    lora_weights_list=lora_weights,
                    lora_multipliers=lora_multipliers,
                    calc_device=device,
                    move_to_device=True,
                    dit_weight_dtype=None,
                )
        else:
            assert lora_weights is None, "LoRA weights merging is only supported for safetensors checkpoints"
            state_dict = torch.load(qwen3_path, map_location="cpu", weights_only=True)

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[len("model.") :]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded Qwen3 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens.

    Args:
        t5_tokenizer_path: Optional path to T5 tokenizer directory. If None, uses default configs.
    """
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    # Use bundled config
    config_dir = os.path.join(os.path.dirname(__file__), "configs", "t5_old")
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, "spiece.model"),
            tokenizer_file=os.path.join(config_dir, "tokenizer.json"),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json. "
        "You can download these from the google/t5-v1_1-xxl HuggingFace repository."
    )