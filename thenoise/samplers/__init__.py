"""Denoising solvers (samplers) for the diffusion pipeline.

A sampler owns one denoising pass over a schedule and its solver-specific state.
Higher-order solvers may evaluate ``denoise_step`` more than once per step.
"""
from __future__ import annotations

from typing import Dict, Type

import torch

from .base import Sampler, Step
from .euler import EulerSampler
from .uni_pc import UniPCSampler
from .er_sde import ErSdeSampler

SAMPLERS: Dict[str, Type[Sampler]] = {
    "euler": EulerSampler,
    "er_sde": ErSdeSampler,
    "unipc": UniPCSampler,
}


def check_sampler(name: str) -> None:
    """Refuse a name no solver in this registry answers to, naming the ones there are."""
    if name not in SAMPLERS:
        raise ValueError(
            f"unknown sampler: {name!r} (choose {', '.join(sorted(SAMPLERS))})"
        )


def create_sampler(name: str, model) -> Sampler:
    """Instantiate the named sampler bound to ``model``."""
    check_sampler(name)
    return SAMPLERS[name](model)


def sigma_steps(grid, device, dtype) -> list[Step]:
    """A descending sigma grid (ending on 0) -> one ``Step`` per grid gap.

    The step's tensors are in the model's own device and dtype because ``denoise_step``
    consumes ``Step.t`` the way it consumes its own ``schedule()`` output.
    """
    ts = torch.tensor(list(grid), device=device, dtype=dtype)
    return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(len(grid) - 1)]


__all__ = ["SAMPLERS", "Sampler", "Step", "check_sampler", "create_sampler",
           "sigma_steps"]
