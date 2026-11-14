"""Euler sampler: first-order integration of the flow ODE."""
from __future__ import annotations

from typing import Callable, List

import torch
from tqdm import tqdm

from .base import Sampler, Step


class EulerSampler(Sampler):
    def sample(
        self,
        x: torch.Tensor,
        schedule: List[Step],
        cond,
        guidance_scale: float,
        seed: int,
        desc: str = "sampling",
        step_callback: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        self.model.begin_sampling(schedule)
        for i, step in tqdm(enumerate(schedule), total=len(schedule), desc=desc):
            v = self.model.denoise_step(x, step.t, cond, guidance_scale, i)
            x = x.float() - step.delta * v.float()
            x = x.to(dtype)
            if step_callback is not None:
                x = step_callback(x, i + 1, len(schedule))
            self.sync(x)
        return x
