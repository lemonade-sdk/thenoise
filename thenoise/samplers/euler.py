"""Euler sampler: first-order integration of a flow ODE with CFG."""
from __future__ import annotations

from typing import List

import torch
from tqdm import tqdm

from thenoise.utils.device import synchronize_device
from .base import Sampler, Step


class EulerSampler(Sampler):
    """Euler integration of the flow ODE with CFG.

    ``x <- x - delta * velocity``, one ``denoise_step`` per schedule step. Integration
    runs in fp32 (precise, cheap) and is cast back to the latent dtype each step.
    """

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
            # Synchronize to get accurate timing, on whatever device we're on.
            synchronize_device(x.device)
        return x
