"""Euler sampler: first-order integration of the flow ODE."""
from __future__ import annotations

from typing import List

import torch
from tqdm import tqdm

from thenoise.utils.device import synchronize_device
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
    ) -> torch.Tensor:
        dtype = x.dtype
        for i, step in tqdm(enumerate(schedule), total=len(schedule), desc=desc):
            v = self.model.denoise_step(x, step.t, cond, guidance_scale, i)
            x = x.float() - step.delta * v.float()
            x = x.to(dtype)
            # Keep the per-step timing accurate.
            synchronize_device(x.device)
        return x
