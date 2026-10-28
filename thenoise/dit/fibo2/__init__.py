"""Fibo 2.0 — a single-stream DiT that reads text through a Perceiver, not a stream.

One set of weights generates and edits; the codec it denoises is the Flux.2 AE, shared
with the Flux.2 models.
"""
from .models import (
    Fibo2Params,
    Fibo2Transformer2DModel,
    gist_count,
    scale_tokens,
    sequence_positions,
)
from .utils import detect_params, fibo2_key_map, is_fibo2, load_fibo2_dit

__all__ = [
    "Fibo2Params",
    "Fibo2Transformer2DModel",
    "detect_params",
    "fibo2_key_map",
    "gist_count",
    "is_fibo2",
    "load_fibo2_dit",
    "scale_tokens",
    "sequence_positions",
]
