"""Sol attention on the Wan trunk: which rows a routed step keeps exact.

The kernel's rules and the gates every run passes are in
:mod:`thenoise.utils.sol_attention`. What is Wan's own is a sink range and where the window
comes from: its self-attention carries no mask, and the prompt reaches the trunk through
cross-attention, which is far too short to route and is never offered a plan.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch

from thenoise.utils.attention import DEFAULT_ATTENTION
from thenoise.utils.sol_attention import BLOCK, WINDOW, SolAttention, route


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
             device, sigma: float, window: Tuple[float, float],
             method: str = DEFAULT_ATTENTION) -> SolAttention | None:
    """This step's kernel arguments, or ``None`` to keep the step dense.

    Decided by the adapter, outside the compiled blocks: reading a per-step sigma inside them
    would be a dynamo guard on a value that changes every step. ``window`` is in sigmas, high
    first, because that is what a flow schedule with a shift can answer in.
    """
    return route(tokens=tokens, sigma=sigma, method=method,
                 sinks=wan_sinks(tokens_per_frame), device=device, dtype=dtype,
                 head_dim=head_dim, window=window)
