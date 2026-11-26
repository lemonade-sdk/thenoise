"""Block-sparse attention's kernel-side rules: what the kernel accepts, the operands it
reads, its arguments, and the gates a step passes before it is routed at all.

Two halves, split by when they are known. :func:`routing_unavailable` answers whether this
hardware and this trunk can route -- a fact of the load, so a model can say which attention
methods it answers before a step runs. :func:`route` answers whether *this step* should,
from the request, the schedule and the rows that must stay exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import torch
from comfy_kitchen import sol_attn, sol_attn_is_available

#: The kernel's query and key block, in tokens. Sink ranges are counted in these.
BLOCK = 64
#: The head width the fused backends are built for.
HEAD_DIM = 128
#: What the fused backends read: activations, not however a checkpoint stores weights.
DTYPES = (torch.bfloat16, torch.float16)

#: One token per query block routes individually; 256 is the kernel's maximum.
TOKEN_AUG_MAX = 256

#: Routing threshold in sigmas of the proxy row: how far a key block's mean score has to
#: stand above an independence null to be attended exactly rather than pooled. A
#: significance level, not a budget, so what it keeps is a property of the run: on a
#: 40 114-token clip it kept 25.6 % of the key blocks exact, 17.6 points of them chosen by
#: the threshold and the rest by the kernel's always-exact neighbourhood and sinks.
TAU = 1.3
#: Percents of the schedule inside which routing is used: the earliest steps stay dense,
#: being where a run decides its structure. A family may answer with its own range.
WINDOW = (0.2, 1.0)

#: Below this many tokens a call is not worth routing: dense is faster. It is a crossover
#: measurement, and it moves with head width rather than with content, so a request for
#: ``optimized`` attention is not held by it.
MIN_TOKENS = 12288


@dataclass(frozen=True)
class SolAttention:
    """One call's kernel arguments. Both sink pairs are half-open block ranges."""

    tau: float
    token_aug: int = TOKEN_AUG_MAX
    sink_blocks: Tuple[int, int] = (0, 0)
    sink_q: Tuple[int, int] = (0, 0)


def available(device) -> bool:
    """Whether a compiled sol kernel exists for this device."""
    return bool(sol_attn_is_available(torch.device(device)))


@lru_cache(maxsize=8)
def routing_unavailable(device, dtype: torch.dtype,
                        head_dim: int) -> Optional[str]:
    """Why this hardware cannot route this trunk, or ``None`` when it can.

    A fact of the load rather than of a step: a model consults it once and answers
    ``attention_methods`` from it, and every step that asks again hits the cache. The answer
    is a sentence because it is what a refused request says.
    """
    # The kind is read without building a device object, so this answers on a CPU-only
    # build too -- and the memo key stays hashable for a caller holding a path string.
    kind = getattr(device, "type", None) or str(device).split(":")[0]
    if not available(device):
        return f"no compiled sol_attn kernel for {device}"
    if head_dim != HEAD_DIM:
        return f"a {head_dim}-wide head; the fused backends are built for {HEAD_DIM}"
    if dtype not in DTYPES:
        return f"{dtype} activations; the fused backends take bf16 or fp16"
    if kind != "cuda":
        return f"{device}"
    return None


def route(*, tokens: int, sigma: float, method: str, sinks: Tuple[int, int],
          device, dtype: torch.dtype, head_dim: int,
          window: Optional[Tuple[float, float]] = None,
          dense_rows: Optional[Tuple[int, int]] = None,
          min_tokens: Optional[int] = None) -> Optional[SolAttention]:
    """This step's kernel arguments, or ``None`` to keep it dense.

    The per-step half of the decision: what the request asked for, where the schedule has
    got to, how long the sequence is, and which rows a family insists stay exact. Whether
    this hardware can route at all is :func:`routing_unavailable`, so asking costs a lookup.

    ``min_tokens`` is read at call time rather than bound as a default, so it can be lowered
    without restating the other gates.
    """
    if method == "full":
        return None
    if routing_unavailable(device, dtype, head_dim) is not None:
        return None
    floor = MIN_TOKENS if min_tokens is None else min_tokens
    # The crossover is an economy, so an explicit request walks past it.
    if method != "optimized" and tokens < floor:
        return None
    if window is not None and not window[1] <= sigma <= window[0]:
        return None
    return SolAttention(TAU, TOKEN_AUG_MAX, sinks, dense_rows or (0, 0))


@torch._dynamo.disable()
def sol_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  plan: SolAttention) -> torch.Tensor:
    """The kernel, on ``[B, T, H, D]`` operands, returning the same order.

    An island of its own because its arguments are Python block counts that dynamo would
    otherwise guard on. Letting the kernel be traced instead measured at equal speed — the
    call is
    already a ``torch.library`` custom op with a fake — so this is the simpler of two options,
    not the faster one.

    The operands arrive as a fused qkv projection's three slices, so a token's pitch is
    ``3 * H * D`` rather than ``H * D``; the kernel reads that, measured bit-identical to
    packed copies at a length that is a whole number of blocks and one that is not. Nothing
    is copied on the way in, which a dense path does not get to say about itself: see
    :func:`thenoise.utils.attention.uniform_layout`.
    """
    return sol_attn(q, k, v, tau=plan.tau, sink_blocks=list(plan.sink_blocks),
                    sink_q=list(plan.sink_q), token_aug=plan.token_aug)
