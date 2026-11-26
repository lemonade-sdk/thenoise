"""Sol attention on the MiniMax-H3 trunk: which rows a routed step keeps exact.

The kernel's rules and the gates every run passes are in
:mod:`thenoise.utils.sol_attention`. What is H3's own is short: the sinks, because only a
packing can say which rows are conditioning, and the window, because a percent of the
schedule means something else on a schedule shifted by 12.

Stills do not route at all: routed, H3-Image degrades visibly, and the mechanism fits — the
threshold is fitted to proxy rows that look near-Gaussian, which a smooth spatial field does
not, so routing keeps only the always-exact diagonal and stops responding to tau. That is
why the image model does not list ``optimized`` and never builds a plan, rather than vetoing
one here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from thenoise.utils.attention import DEFAULT_ATTENTION
from thenoise.utils.sol_attention import BLOCK, WINDOW, SolAttention, route


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
             sigma: float, sample_sigmas: Optional[torch.Tensor],
             method: str = DEFAULT_ATTENTION) -> Optional[SolAttention]:
    """This run's kernel arguments, or ``None`` to keep the step dense.

    The whole step takes one answer, decided outside the compiled blocks: a per-step sigma
    read inside them is a dynamo guard, and every block recompiles every step.
    """
    if sample_sigmas is None:
        return None
    if layout.seq_len != tokens:
        # The sinks come from the packing, and one that does not describe this call cannot
        # say where they are. A pooled prompt row is what this feature must never do.
        return None
    sinks, rows = sol_sinks(layout)
    return route(tokens=tokens, sigma=sigma, method=method, sinks=sinks, dense_rows=rows,
                 device=device, dtype=dtype, head_dim=head_dim,
                 window=window_sigmas(sample_sigmas))
