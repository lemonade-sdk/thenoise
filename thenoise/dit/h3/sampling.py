"""The H3 denoising schedules.

H3-Image walks the sigma grid its output head bank was distilled on, so that grid is the
shape of the trained model rather than a tuning knob and the ladder runs over its *index*
(``samplers.sigma_ladder``). The flow shift never applies there: it would move the ladder
off the points the heads were distilled for.

The audiovisual model has two schedules at once. The run steps on the video's, and the
audio stream reads the same base grid under its own, smaller shift: a sigma is one point
of a shared underlying grid re-expressed under another shift, which is what lets a single
solver carry both streams. See :func:`carry_ratio`.
"""
from __future__ import annotations

from typing import Sequence

import torch

from thenoise.samplers import sigma_at
from thenoise.utils.math import time_shift

#: The audio stream's flow shift, where the video's is a pref. Their ratio scales the
#: carry, so nothing else may restate it.
AUDIO_SHIFT = 3.0


def percent_to_sigma(grid: Sequence[float], percent: float) -> float:
    """Denoising progress (0 = start, 1 = clean) as a sigma on the grid."""
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    return sigma_at(grid, 1.0 - percent)


def get_sigmas(steps: int, *, shift: float) -> list[float]:
    """``steps`` sigmas of a shifted schedule, descending, ending on ``0.0``.

    The base grid walked uniformly and then shifted, so the steps bunch where the shift
    puts the model's working range.
    """
    return [time_shift(1.0 - i / steps, shift) for i in range(steps)] + [0.0]


def reshift(sigma: float, *, from_shift: float, to_shift: float) -> float:
    """A sigma read on one stream's schedule, on another's.

    Back to the base grid the schedules share, then out through the other shift.
    """
    return time_shift(sigma / (from_shift + sigma * (1.0 - from_shift)), to_shift)


def audio_sigma(sigma: float, *, shift: float) -> float:
    """Where the audio stream is while the video stream is at ``sigma``.

    Always the further along of the two: the audio runs on the smaller shift.
    """
    return reshift(sigma, from_shift=shift, to_shift=AUDIO_SHIFT)


def shift_ratio(shift: float) -> float:
    """How far the video shift overshoots the audio's."""
    return shift / AUDIO_SHIFT


def carry_ratio(sigma: float, *, shift: float) -> float:
    """What the audio slice of a run's pack is scaled by, against the stream's own latent.

    A solver steps one schedule, so the audio has to travel the video's: while the stream
    is at :func:`audio_sigma`, the pack holds ``carry_ratio · own``. For these two shift
    maps that is ``σv / σa == S − (S−1)·σv``: linear in the video sigma, 1 at pure noise —
    so the starting audio needs no scaling — and ``S`` at the clean end, which is why a
    finished audio slice is divided by :func:`shift_ratio` before anything decodes it.
    """
    s = shift_ratio(shift)
    return s - (s - 1.0) * sigma


def carry(latent: torch.Tensor, sigma: float, *, shift: float) -> torch.Tensor:
    """A run's carried audio slice as the stream's own latent, which is what the model reads."""
    return latent / carry_ratio(sigma, shift=shift)


def audio_velocity(velocity: torch.Tensor, own: torch.Tensor, *, sigma: float,
                   shift: float) -> torch.Tensor:
    """A predicted audio velocity, in the carried variable the solver steps.

    ``own`` is the latent the model read, the stream's own. The term that is not the
    velocity is the derivative of the carry itself, and both coefficients are constant in
    the base grid for these shift maps — ``d carry_ratio/d σv == 1 − S`` and
    ``carry_ratio · d σa/d σv == 1 + (S − 1)·σa`` — which is what makes this exact rather
    than an approximation.
    """
    s = shift_ratio(shift)
    return ((1.0 + (s - 1.0) * audio_sigma(sigma, shift=shift)) * velocity
            + (1.0 - s) * own)


__all__ = [
    "AUDIO_SHIFT",
    "audio_sigma",
    "audio_velocity",
    "carry",
    "carry_ratio",
    "get_sigmas",
    "percent_to_sigma",
    "reshift",
    "shift_ratio",
    "time_shift",
]
