"""Shifted flow schedules and shared Euler/UniPC integration."""
from __future__ import annotations

import math
from typing import Callable

import torch

from thenoise.utils.math import generalized_time_shift
from .uni_pc import UniPCStepper


def flow_sigmas(steps: int, shift: float) -> list[float]:
    if steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("steps and flow shift must be positive")
    return [generalized_time_shift(1 - i / steps, math.log(shift), 1.0)
            for i in range(steps)] + [0.0]


def sample_flow(x: torch.Tensor, sigmas: list[float],
                predict: Callable[[torch.Tensor, float], torch.Tensor], *,
                sampler: str, progress: Callable[[int, int], None],
                after_step: Callable[[torch.Tensor], torch.Tensor] | None = None,
                cancel: Callable[[], None] | None = None) -> torch.Tensor:
    """Integrate a flow velocity function; model-specific conditioning stays with the model."""
    if sampler not in ("euler", "unipc"):
        raise ValueError(f"unknown flow sampler: {sampler}")
    solver = UniPCStepper() if sampler == "unipc" else None
    for i, (sigma, next_sigma) in enumerate(zip(sigmas, sigmas[1:])):
        if cancel:
            cancel()
        if solver is None:
            x = x + (next_sigma - sigma) * predict(x, sigma)
        else:
            x = solver.step(x, sigma, next_sigma, predict)
        if after_step:
            x = after_step(x)
        progress(i + 1, len(sigmas) - 1)
    return x
