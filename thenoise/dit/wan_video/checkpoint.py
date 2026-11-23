"""Wan 2.2 transformer checkpoint loading (Comfy-Org 5B / 14B layouts)."""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from thenoise.dit.wan_video.model import Wan22Params, WanNativeDiT, detect_params
from thenoise.utils.loader import (
    QUANTIZED_COMPUTE_DTYPE,
    build_dit,
    is_quantized_checkpoint,
)
from thenoise.utils.safetensors import Header, as_header
from thenoise.utils.wan_weights import wan_dit_key

logger = logging.getLogger(__name__)


def load_wan_native_transformer(
    checkpoint: Union[str, Header],
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
    header = as_header(checkpoint)
    params = params or detect_params(header)
    quantized = is_quantized_checkpoint(header)
    logger.info("Loading Wan DiT (%d blocks, %d wide, %s) from %s",
                params.num_layers, params.dim, "quantized" if quantized else "plain",
                header.path)
    model = build_dit(
        lambda: WanNativeDiT(params),
        header,
        device=device,
        dtype=dtype or (QUANTIZED_COMPUTE_DTYPE if quantized else None),
        key_map=wan_dit_key,
        relay_stores=True,
    )
    return model.apply_precision().eval()
