"""Denoising solvers (samplers) for the diffusion pipeline.

A sampler owns one denoising pass over a schedule, calling the model's
``denoise_step`` exactly once per schedule step. Solver-specific state (e.g. ER-SDE's
higher-order terms, noise scaler, RNG) lives here rather than on the model, so the
model base class stays free of any particular integration scheme.

Each sampler is bound to a model instance (via ``Sampler(model)``) and duck-types the
methods it needs — ``denoise_step`` and (for ER-SDE) ``percent_to_sigma`` — so the
samplers never import the model classes.
"""
from __future__ import annotations

import logging
from typing import Dict, Type

logger = logging.getLogger(__name__)

from .base import Sampler, Step
from .euler import EulerSampler
from .er_sde import ErSdeSampler

#: name -> sampler class. New solvers register here.
SAMPLERS: Dict[str, Type[Sampler]] = {
    "euler": EulerSampler,
    "er_sde": ErSdeSampler,
}


def create_sampler(name: str, model) -> Sampler:
    """Instantiate the named sampler bound to ``model``.

    A model may declare ``SUPPORTED_SAMPLERS`` (list of names). If the requested
    ``name`` is a known sampler but not in that list, warn and fall back to the
    model's ``SAMPLER`` default instead of producing a wrong result. Most models
    do not declare it, so any registered sampler is usable.

    Raises ``ValueError`` for an unknown sampler name.
    """
    supported = getattr(model, "SUPPORTED_SAMPLERS", None)
    if (
        supported is not None
        and name in SAMPLERS
        and name not in supported
    ):
        logger.warning(
            "%s sampler is not supported by %s; using %s instead",
            name,
            model.name,
            model.SAMPLER,
        )
        name = model.SAMPLER
    cls = SAMPLERS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown sampler: {name!r} (choose {', '.join(sorted(SAMPLERS))})"
        )
    return cls(model)


__all__ = ["SAMPLERS", "Sampler", "Step", "create_sampler"]
