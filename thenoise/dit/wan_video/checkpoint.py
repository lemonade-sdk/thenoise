"""Wan 2.2 transformer checkpoint loading (Comfy-Org 5B / 14B layouts)."""
from __future__ import annotations

import logging
from typing import Optional

import torch

from thenoise.dit.wan_video.model import Wan22Params, WanNativeDiT, detect_params
from thenoise.utils.loader import is_quantized_checkpoint, load_dit
from thenoise.utils.wan_weights import wan_dit_key

logger = logging.getLogger(__name__)

#: A quantized file's remaining full-precision tensors follow the quantized kernels,
#: which emit BF16; whatever dtype the export left them in is not worth trusting.
QUANTIZED_COMPUTE_DTYPE = torch.bfloat16


def load_wan_native_transformer(
    dit_path: str,
    device,
    dtype: Optional[torch.dtype] = None,
    *,
    params: Optional[Wan22Params] = None,
) -> WanNativeDiT:
    """Load a Wan transformer — plain or FP8-scaled, 5B or one 14B expert.

    The geometry comes from the file, so the caller does not have to know which
    variant it was handed, and neither does the dtype: a plain checkpoint runs at the
    precision its weights are written in, a quantized one at what its kernels emit,
    and ``dtype`` overrides both.
    """
    params = params or detect_params(dit_path)
    quantized = is_quantized_checkpoint(dit_path)
    logger.info("Loading Wan DiT (%d blocks, %d wide, %s) from %s",
                params.num_layers, params.dim, "quantized" if quantized else "plain", dit_path)
    with torch.device("meta"):
        model = WanNativeDiT(params)
    load_dit(
        model, dit_path, device=device,
        dtype=dtype or (QUANTIZED_COMPUTE_DTYPE if quantized else None),
        key_map=wan_dit_key,
    )
    return model.apply_precision().eval()
