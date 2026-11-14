"""Common sampler API."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List

import torch

from thenoise.utils.device import synchronize_device

if TYPE_CHECKING:
    from thenoise.models.base import Conditioning, DiffusionModel


@dataclass
class Step:
    """One denoising step: timestep ``t`` and Euler step size ``delta``."""

    t: torch.Tensor
    delta: torch.Tensor


class Sampler(ABC):
    """Base class for a denoising solver, bound to a model instance."""

    def __init__(self, model: "DiffusionModel"):
        self.model = model

    def sync(self, x: torch.Tensor) -> None:
        """Wait for the step just taken, once per step, at the end of every solver loop.

        A loop that lets the host run ahead measures the submission of kernels rather
        than their execution, so the progress a caller reports drifts from what the device
        has finished — and asks the allocator for the next step's activations before this
        step's are released.
        """
        synchronize_device(x.device)

    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        schedule: List[Step],
        cond: "Conditioning",
        guidance_scale: float,
        seed: int,
        desc: str = "sampling",
        step_callback: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run one denoising pass over ``schedule``, returning the denoised latent.

        ``x`` is the model-internal latent coming from ``prepare_latent``; the return
        value goes to ``finalize_latent``. Implementations hand ``schedule`` to
        ``model.begin_sampling`` first, so a model that needs the whole grid sees the
        one that is actually being run.
        """
