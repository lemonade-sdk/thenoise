"""Sol attention on the Wan trunk: which steps route, and what a routed step keeps exact.

The generic half — what the kernel accepts, the operand order it reads, what a threshold
kept — is in :mod:`thenoise.utils.sol_attention`. What is Wan's own is a sink range and a
window. What Wan does not need is telling, too: its self-attention carries no mask, and the
prompt reaches the trunk through cross-attention, which is three orders of magnitude too
short to route and is never offered a plan.
"""

from __future__ import annotations

import logging
from typing import Callable, Tuple

import torch

from thenoise.utils.sol_attention import (BLOCK, TOKEN_AUG_MAX, SolAttention, available,
                                          dense_reason, log_once)

#: Routing threshold in sigmas of the proxy row: a significance level rather than a budget,
#: so what it keeps is a property of the clip. See
#: :func:`thenoise.utils.sol_attention.routing_stats` for the readout of that.
TAU = 1.3
#: Percents of the schedule inside which routing is used; the steps before it stay dense,
#: being where a clip decides its structure. The reference ships 0.2; the practice in the
#: literature is 0.3. Wan's own defaults are 12 steps for T2V and 40 for the I2V pair, so
#: this is 2 and 8 steps respectively.
WINDOW = (0.2, 1.0)


def wan_sinks(tokens_per_frame: int) -> Tuple[int, int]:
    """Exact-KV key blocks: the first frame's tokens, attended exactly by every query.

    Routing happens in self-attention, whose sequence is the clip alone, so the rows worth a
    full price are the clip's beginning — the conditioned frame in an image-to-video run and
    the anchor the rest of the clip has to agree with in a text one. A frame out of ``frames``
    is also what makes this affordable: the range is a fraction of the work that shrinks as
    clips get longer, which is the direction a feature wants to be in.
    """
    return (0, -(-tokens_per_frame // BLOCK))


def window_sigmas(read: Callable[[float], float],
                  window: Tuple[float, float] = WINDOW) -> Tuple[float, float]:
    """The run's sigma at each end of ``window``, high first.

    A callback rather than a schedule tensor because Wan's schedule is analytic in its flow
    shift, so an adapter knows its own window without the sampler handing anything over.
    """
    return read(window[0]), read(window[1])


def wan_plan(*, tokens: int, tokens_per_frame: int, dtype: torch.dtype, head_dim: int,
             device, sigma: float,
             window: Tuple[float, float]) -> SolAttention | None:
    """This step's kernel arguments, or ``None`` to keep the step dense.

    Decided by the adapter, outside the compiled blocks: reading a per-step sigma inside them
    would be a dynamo guard on a value that changes every step. ``window`` is in sigmas, high
    first, because that is what a flow schedule with a shift can answer in.
    """
    def dense(reason: str) -> None:
        log_once(reason, logging.DEBUG, "sol dense at %d tokens: %s", tokens, reason)

    reason = dense_reason(tokens=tokens, dtype=dtype, head_dim=head_dim, device=device)
    if reason is None and not window[1] <= sigma <= window[0]:
        reason = f"sigma {sigma:.3g} outside the {WINDOW[0]}-{WINDOW[1]} window"
    if reason is None and not available(device):
        reason = "no compiled sol_attn kernel for this GPU"
    if reason is not None:
        dense(reason)
        return None
    sinks = wan_sinks(tokens_per_frame)
    log_once("sparse", logging.DEBUG, "sol attention: %d tokens, tau %g, sinks %s",
             tokens, TAU, sinks)
    return SolAttention(TAU, TOKEN_AUG_MAX, sinks)
