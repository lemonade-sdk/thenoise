"""Dynamo/Inductor helpers for the compiled DiT blocks.

The per-block forwards are compiled with Dynamo's default (auto) dynamic mode, whose
recompile-time promotion symbolises *every* axis it can reach — which is where
Inductor's tiling analysis falls over (``CantSplit`` / symbolic ``Mul``) on the
reference-token and KV-cache paths. The models therefore declare the one axis that
genuinely varies up front.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch._dynamo import maybe_mark_dynamic

__all__ = ["RECOMPILE_LIMIT", "apply_recompile_limit", "mark_token_axis"]

#: Dynamo's own ceiling is 8 recompiles; a run meets one graph per shape it hands the
#: compiled blocks — resolution, reference count, and for a two-expert model which
#: expert is running — and passing the ceiling aborts the run rather than slowing it.
RECOMPILE_LIMIT = 64


def apply_recompile_limit() -> None:
    """Raise Dynamo's recompile ceiling to what this engine's runs need (global)."""
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT


def mark_token_axis(*tensors: Optional[torch.Tensor], dim: int = 1) -> None:
    """Declare that ``dim`` of ``tensors`` varies, for the compiled blocks.

    The token axis carries the resolution, prompt length and reference count;
    marking it gives one kernel per graph *shape* with batch, heads and hidden size
    static.

    ``maybe_mark_dynamic`` so an axis the graph specialises anyway (a broadcast
    ``[B, 1, D]`` modulation row) is specialised silently instead of raising
    ``ConstraintViolationError``.
    """
    for t in tensors:
        if t is not None and t.dim() > dim:
            maybe_mark_dynamic(t, dim)
