"""The generic half of block-sparse attention: the kernel's own rules, and a readout of
what it decided.

Everything here is model-agnostic: what the kernel accepts, the order it reads operands
in, the arguments it takes, and how many key blocks a given threshold actually kept. A
model contributes the two things only it can know — which rows are conditioning, and
whether its family routes at all — and the engine asks it for a plan.

The kept-fraction readout exists because ``tau`` is a significance level on a proxy row
rather than a budget: the same value keeps a different fraction of the sequence depending
on how the scores happen to be distributed, so a run tuned without that number visible is
tuned blind. It is a measurement rather than a record of a rule, and it runs only under
debug logging: once per threshold and sequence length, on the operands of a real call.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from comfy_kitchen import sol_attn, sol_attn_is_available

logger = logging.getLogger(__name__)

#: The kernel's query and key block, in tokens. Sink ranges and the kept-fraction readout
#: are counted in these.
BLOCK = 64
#: The head width the fused backends are built for. Every codec and decoder width in this
#: catalog is 64 and therefore never qualifies; trunk widths do.
HEAD_DIM = 128
#: What the fused backends read. Every weight mode this engine runs computes in one of
#: them, so this asks about activations and never about how a checkpoint stores weights.
DTYPES = (torch.bfloat16, torch.float16)

#: One token per query block routes individually; 256 is the kernel's maximum.
TOKEN_AUG_MAX = 256

#: Below this many tokens a call is not worth routing: dense is faster. The crossover moves
#: with head width — which every 128-wide trunk in this catalog shares — rather than with what
#: a model generates, so they share this number until one measures a different one. A request
#: for ``optimized`` attention is not held by it.
MIN_TOKENS = 12288


@dataclass(frozen=True)
class SolAttention:
    """One call's kernel arguments. Both sink pairs are half-open block ranges."""

    tau: float
    token_aug: int = TOKEN_AUG_MAX
    sink_blocks: Tuple[int, int] = (0, 0)
    sink_q: Tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class RoutingStats:
    """What the threshold did on the operands it was measured on."""

    tokens: int
    blocks: int
    tau: float
    #: Key blocks over the proxy threshold, averaged over query blocks and heads.
    routed: float
    #: ... and with the always-exact neighbourhood, sink keys and sink rows included.
    kept: float

    def __str__(self) -> str:
        return (f"{self.kept:.1%} of {self.blocks} key blocks exact at tau {self.tau:g} "
                f"({self.routed:.1%} by threshold) over {self.tokens} tokens")


def available(device) -> bool:
    """Whether a compiled sol kernel exists for this device."""
    return bool(sol_attn_is_available(device))


_LOGGED: set = set()
_LOG_LIMIT = 32


def claim(key) -> bool:
    """Whether this key is new, claiming it if so.

    Separate from :func:`log_once` because the kept-fraction readout is a pass over the
    operands rather than a string, and work whose only consumer is a log line should be
    skipped by the same test that skips the line.
    """
    if key in _LOGGED or len(_LOGGED) >= _LOG_LIMIT:
        return False
    _LOGGED.add(key)
    return True


def log_once(key, level: int, message: str, *args) -> None:
    """One line per distinct key per process, so a run says anything once and not 1000x.

    A run that flips between dense and sparse produces a bounded number of keys either way
    (reason, token count); the cap is only so a pathological loop cannot grow it.
    """
    if claim(key):
        logger.log(level, message, *args)


def reset_log() -> None:
    """Forget what has been reported. For tests, and for a second run in one process."""
    _LOGGED.clear()


def dense_reason(*, tokens: int, dtype: torch.dtype, head_dim: int, device,
                 min_tokens: Optional[int] = None,
                 cached: bool = False) -> Optional[str]:
    """Why this call cannot be routed, or ``None`` when it can be.

    The kernel's limits — width, dtype, device, and a length below which dense wins — plus
    one that is not the kernel's but is never worth trying: a call whose keys live in a
    cache buffer has no honest sink range to hand over, and the sinks are why a routed run of
    a conditioned model can be trusted at all.

    ``min_tokens`` defaults to :data:`MIN_TOKENS` read at call time, not bound as a default,
    so a trunk can lower it for a test without restating the rest of the gates.
    """
    floor = MIN_TOKENS if min_tokens is None else min_tokens
    if tokens < floor:
        return f"under {floor} tokens"
    if head_dim != HEAD_DIM:
        return f"head_dim {head_dim} != {HEAD_DIM}"
    if dtype not in DTYPES:
        return f"dtype {dtype}, the fused backends take bf16 or fp16"
    if device.type != "cuda":
        return f"device {device}"
    if cached:
        return "the K/V cache holds this call's keys"
    return None


def routing_stats(q: torch.Tensor, k: torch.Tensor, tau: float,
                  sink_blocks: Tuple[int, int] = (0, 0),
                  sink_q: Tuple[int, int] = (0, 0)) -> RoutingStats:
    """The kernel's own routing test, counted instead of applied.

    A proxy row per query block — its centroid against every centred key-block centroid —
    and a threshold of ``tau`` sigmas of that row under an independence null, which is the
    test the fused kernels run inside their online softmax. Costs a few GFLOP at 38 k
    tokens, so: once per shape, not once per block.

    The always-exact parts are added the way the kernel adds them — a key block either side
    of the query block, the sink key range, and every key for a sink query row — because
    what matters for tuning is the total budget, not the share the threshold chose.
    """
    batch, tokens, heads, head_dim = q.shape
    n = -(-tokens // BLOCK)
    pad = n * BLOCK - tokens
    lengths = torch.full((n,), float(BLOCK), device=q.device)
    if pad:
        lengths[-1] = float(BLOCK - pad)
        q, k = (torch.cat([t, t.new_zeros(batch, pad, heads, head_dim)], dim=1)
                for t in (q, k))

    def pool(t: torch.Tensor, mean: bool) -> torch.Tensor:
        pooled = t.float().reshape(batch, n, BLOCK, heads, head_dim).sum(2)
        return pooled / lengths.view(1, -1, 1, 1) if mean else pooled

    kc = pool(k, mean=True)                                  # [B, N, H, D] summary keys
    kc_var = (kc - kc.mean(1, keepdim=True)).pow(2).mean(1)  # [B, H, D]
    centroid = pool(q, mean=True)
    q32, kc32 = centroid.permute(0, 2, 1, 3), kc.permute(0, 2, 1, 3)
    proxy = (q32 @ (kc32 - kc32.mean(2, keepdim=True)).transpose(-1, -2)).float()
    sigma = (centroid.float().pow(2).permute(0, 2, 1, 3)
             * kc_var.unsqueeze(2)).sum(-1).clamp_min(1e-6).sqrt()
    routed = proxy > tau * sigma.unsqueeze(-1)

    idx = torch.arange(n, device=q.device)
    exact = routed | ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1)
    exact[:, :, :, sink_blocks[0]:sink_blocks[1]] = True
    exact[:, :, sink_q[0]:sink_q[1], :] = True
    return RoutingStats(tokens, n, tau,
                        float(routed.float().mean()), float(exact.float().mean()))


def _probe(q: torch.Tensor, k: torch.Tensor, plan: SolAttention) -> None:
    """Report what the threshold kept, once per ``(threshold, length)``.

    A block-pooled pass over the operands, which is a millisecond no production run owes
    anybody: the number is only interesting while somebody is investigating, so this runs when
    debug logging is on and never otherwise.
    """
    if not claim(("sol", plan.tau, q.shape[1])):
        return
    logger.debug("sol routing kept %s",
                 routing_stats(q, k, plan.tau, plan.sink_blocks, plan.sink_q))


@torch._dynamo.disable()
def sol_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  plan: SolAttention) -> torch.Tensor:
    """The kernel, on ``[B, T, H, D]`` operands, returning the same order.

    An island of its own because its arguments are Python block counts that dynamo would
    otherwise guard on, and because the readout is a side effect that tracing would bake into
    the graph. Letting the kernel be traced instead measured at equal speed — the call is
    already a ``torch.library`` custom op with a fake — so this is the simpler of two options,
    not the faster one.

    The operands arrive as a fused qkv projection's three slices, so a token's pitch is
    ``3 * H * D`` rather than ``H * D``; the kernel reads that, measured bit-identical to
    packed copies at a length that is a whole number of blocks and one that is not. Nothing
    is copied on the way in, which a dense path does not get to say about itself: see
    :func:`thenoise.utils.attention.uniform_layout`.
    """
    if logger.isEnabledFor(logging.DEBUG):
        _probe(q, k, plan)
    return sol_attn(q, k, v, tau=plan.tau, sink_blocks=list(plan.sink_blocks),
                    sink_q=list(plan.sink_q), token_aug=plan.token_aug)
