"""Dynamo/Inductor helpers for the compiled DiT blocks.

The per-block forwards are compiled with Dynamo's default (auto) dynamic mode,
which compiles the first call of a frame fully static and only promotes the axes
that wobbled on a recompile — and the promotion symbolises *every* axis it can
reach, which is where Inductor's tiling analysis falls over (``CantSplit`` /
symbolic ``Mul``) on the reference-token and KV-cache paths. The models therefore
declare the one axis that genuinely varies up front instead of letting Dynamo
discover it by recompiling.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch._dynamo import maybe_mark_dynamic

__all__ = ["mark_token_axis"]


def mark_token_axis(*tensors: Optional[torch.Tensor], dim: int = 1) -> None:
    """Declare that ``dim`` of ``tensors`` varies, for the compiled blocks.

    Declaring the token axis (which carries the resolution, prompt length and
    reference count) gets one kernel per graph *shape* while batch, heads and
    hidden size stay static, and avoids the wasted static-first compilation.

    ``maybe_mark_dynamic`` rather than ``mark_dynamic``: an axis the graph
    specialises anyway (a broadcast ``[B, 1, D]`` modulation row, say) is then
    specialised silently instead of raising ``ConstraintViolationError``.
    """
    for t in tensors:
        if t is not None and t.dim() > dim:
            maybe_mark_dynamic(t, dim)
