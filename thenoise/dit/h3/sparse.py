"""Sol attention on the MiniMax-H3 trunk: which runs route, and with what numbers.

The generic half — the kernel's rules, its operand order, what a threshold kept — is in
:mod:`thenoise.utils.sol_attention`. What is H3's own is short: the sinks, because only a
packing can say which rows are conditioning, and the family rules, because only this model
knows what routing did to its own output.

It is on whenever it applies and there is no request for it. Which route a call took follows
from the constants below, so that is debug traffic; what a run measured -- the fraction of
key blocks the kernel kept -- is the one thing reported, by
:mod:`thenoise.utils.sol_attention`. If device testing turns up a failure the answer is a
per-request knob added then, not a hatch installed now.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from thenoise.utils.sol_attention import (BLOCK, DTYPES, HEAD_DIM, TOKEN_AUG_MAX,
                                          SolAttention, available)

logger = logging.getLogger(__name__)

#: Routing threshold in sigmas of the proxy row: how far a key block's mean score has to
#: stand above an independence null to be attended exactly rather than pooled. It is a
#: significance level, not a budget, so the fraction it keeps moves with the content —
#: see :func:`thenoise.utils.sol_attention.routing_stats`.
TAU = 1.3
#: Below this the dense path is faster. The kernel names the same crossover; on gfx1151 the
#: kernel already wins at 1.3 k tokens, by 2 ms, which is not worth an opinion about.
MIN_TOKENS = 12288
#: Percents of the schedule inside which routing is used; the earliest steps stay dense,
#: where the run decides its structure. The reference defaults to 0.2; the practice in the
#: literature (SVG2, carried by later work) is 0.3, which on a 4-step run is the whole run.
WINDOW = (0.2, 1.0)
#: Stills do not route. Routed at 16,384 tokens (a 4096x4096 canvas) H3-Image degraded
#: visibly and badly, and the mechanism is consistent with that: the threshold is fitted to
#: proxy rows that look near-Gaussian, which is what video scores look like and a smooth
#: spatial field does not — on smooth operands it keeps only the always-exact diagonal and
#: stops responding to tau at all. Revisit with a kept-fraction readout from a real image
#: run, not with a different tau.
STILL_HELD_DENSE = "a still: routing measured visibly degrading at 16,384 tokens"


def sol_sinks(layout, *, audio_rows_dense: bool = True
              ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """``(exact-KV blocks, dense-query blocks)`` for a packing, :data:`BLOCK` rows a block.

    Every row before the video span is attended exactly by every query: the prompt and the
    keyframes are what a clip is conditioned on, and pooling them into one term per block is
    how a run loses them. ``audio_rows_dense`` runs the audio *query* rows dense too, which
    costs whole query blocks doing full attention and is the reference's answer to routing
    them damaging the sound.

    No video span, or a prefix that starts at row 0, and there is nothing to protect: the
    answer is no sinks rather than a refusal.
    """
    video = layout.video_span
    if video is None or video[0] <= 0:
        return (0, 0), (0, 0)
    exact = (0, -(-video[0] // BLOCK))
    audio = layout.audio_span
    if not audio_rows_dense:
        return exact, (0, 0)
    # From the audio's first row to the end of the prefix: rounding the top up keeps the
    # partial block that holds the video's own first rows, as the reference does.
    return exact, ((audio[0] // BLOCK, exact[1]) if audio else exact)


def window_sigmas(sample_sigmas: torch.Tensor,
                  window: Tuple[float, float] = WINDOW) -> Tuple[float, float]:
    """The schedule's sigma at each end of ``window``, high first.

    Percents are read as step fractions on the run's own schedule rather than as sigmas, so
    a shifted schedule (12 for H3, which parks most steps above 0.9) is protected in the
    part of itself that is early, not in a range a shift would never visit.
    """
    last = sample_sigmas.shape[0] - 1
    read = lambda percent: float(sample_sigmas[round(percent * last)])
    return read(window[0]), read(window[1])


def sol_plan(*, layout, tokens: int, dtype: torch.dtype, head_dim: int, device,
             sigma: float, sample_sigmas: Optional[torch.Tensor], cached: bool,
             held_dense: Optional[str] = None) -> Optional[SolAttention]:
    """This run's kernel arguments, or ``None`` to keep the step dense.

    The whole step takes one answer, decided outside the compiled blocks: the alternative is
    reading a per-step sigma inside them, which dynamo would guard on and recompile every
    block of every step. The first transition from dense to sparse recompiles once, which is
    the cost the window buys and is paid whether or not the window exists.

    ``held_dense`` is a family's own veto, stated as the reason a log should give.
    """
    def dense(reason: str) -> None:
        _log_once(reason, logging.DEBUG, "sol dense at %d tokens: %s", tokens, reason)

    if held_dense is not None:
        dense(held_dense)
        return None
    if tokens < MIN_TOKENS:
        dense(f"under {MIN_TOKENS} tokens")
        return None
    if head_dim != HEAD_DIM:
        dense(f"head_dim {head_dim} != {HEAD_DIM}")
        return None
    if dtype not in DTYPES:
        dense(f"dtype {dtype}, the fused backends take bf16 or fp16")
        return None
    if device.type != "cuda":
        dense(f"device {device}")
        return None
    if cached:
        # The KV-cache path attends a longer key than query out of a buffer the layout does
        # not describe, so there is no honest sink range to give the kernel.
        dense("the K/V cache holds this call's keys")
        return None
    if layout.seq_len != tokens:
        # Sinks are the whole reason a sparse run of this model is trustworthy, and a
        # layout that does not describe the call cannot say where they are.
        dense(f"layout describes {layout.seq_len} rows, the call has {tokens}")
        return None
    if sample_sigmas is None:
        dense("no schedule to place the window on")
        return None
    high, low = window_sigmas(sample_sigmas)
    if sigma > high or sigma < low:
        dense(f"sigma {sigma:.3g} outside the {WINDOW[0]}-{WINDOW[1]} window")
        return None
    if not available(device):
        dense("no compiled sol_attn kernel for this GPU")
        return None
    sinks, rows = sol_sinks(layout)
    _log_once("sparse", logging.DEBUG, "sol attention: %d tokens, tau %g, sinks %s/%s",
              tokens, TAU, sinks, rows)
    return SolAttention(TAU, TOKEN_AUG_MAX, sinks, rows)


_LOGGED = set()
_LOG_LIMIT = 32


def _log_once(key: str, level: int, message: str, *args) -> None:
    """One line per distinct key per process, so a run says anything once and not 1000x.

    A run that flips between dense and sparse every step produces a bounded number of keys
    either way (reason, token count); the cap is only so a pathological loop cannot grow it.
    """
    if key in _LOGGED or len(_LOGGED) >= _LOG_LIMIT:
        return
    _LOGGED.add(key)
    logger.log(level, message, *args)


def reset_log() -> None:
    """Forget what has been reported. For tests, and for a second run in one process."""
    _LOGGED.clear()
