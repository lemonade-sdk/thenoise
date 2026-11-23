# Anima model loading utilities

from typing import Optional, Union
import torch

from thenoise.dit.anima import models as anima_models
from thenoise.utils.loader import build_dit
from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import Header, Signature, as_header
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


#: Its LLM adapter over a Lumina trunk: the Cosmos releases share the trunk with the
#: models here, the adapter is Anima's alone.
ANIMA_SIGNATURE = Signature("Anima DiT",
                            prefixes=("llm_adapter.", "blocks.0.adaln_modulation"))


def load_anima_model(
    device: Union[str, torch.device],
    checkpoint: Union[str, Header],
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> anima_models.Anima:
    """Load the Anima DiT, deriving its depth from the checkpoint."""
    header = as_header(checkpoint)
    header.require(ANIMA_SIGNATURE)
    # Anima ships variants of different depths on one architecture, and the depth is the
    # one number no released file states.
    num_blocks = header.block_count("blocks.")
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
        "use_llm_adapter": True,
    }
    return build_dit(
        lambda: anima_models.Anima(**dit_config),
        header,
        device=device,
        dtype=dit_weight_dtype,
        key_map=qk_norm_key_map,
        expected_missing=("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq"),
        relay_stores=True,
    )
