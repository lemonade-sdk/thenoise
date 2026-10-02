# Anima model loading utilities

from typing import Optional, Union
import torch
from accelerate import init_empty_weights

from thenoise.dit.anima import models as anima_models
from thenoise.utils.loader import load_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import WRAP_PREFIXES
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _count_anima_blocks(dit_path: str) -> int:
    """Count the main transformer blocks in an Anima checkpoint.

    Anima ships variants with different depths on the same architecture, so the
    count comes from the ``blocks.{i}.`` keys of the safetensors header, with wrap
    prefixes stripped so raw and repackaged files count identically.
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
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> anima_models.Anima:
    """Load the Anima DiT, deriving its block count from the checkpoint."""
    device = torch.device(device)

    num_blocks = _count_anima_blocks(dit_path)
    logger.info("Detected Anima DiT with %d transformer blocks", num_blocks)

    # Fixed DiT config for all Anima checkpoints
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

    logger.info(f"Loading DiT model from {dit_path}, device={device}")

    load_dit(
        model,
        dit_path,
        device=device,
        dtype=dit_weight_dtype,
        key_map=qk_norm_key_map,
        expected_missing=("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq"),
    )
    logger.info("Loaded DiT model from %s", dit_path)

    return model
