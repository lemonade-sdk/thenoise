"""Denoising solvers (samplers) for the diffusion pipeline.

A sampler owns one denoising pass over a schedule and its solver-specific state.
Higher-order solvers may evaluate ``denoise_step`` more than once per step.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import Sampler, Step
from .euler import EulerSampler
from .uni_pc import UniPCSampler
from .er_sde import ErSdeSampler

SAMPLERS: Dict[str, Type[Sampler]] = {
    "euler": EulerSampler,
    "er_sde": ErSdeSampler,
    "unipc": UniPCSampler,
}


def create_sampler(name: str, model) -> Sampler:
    """Instantiate the named sampler bound to ``model``."""
    cls = SAMPLERS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown sampler: {name!r} (choose {', '.join(sorted(SAMPLERS))})"
        )
    return cls(model)


__all__ = ["SAMPLERS", "Sampler", "Step", "create_sampler"]
