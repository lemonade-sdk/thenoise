"""UniPC multistep solver for flow-matching models (image or video)."""
from __future__ import annotations

from collections import deque
import math
from typing import Callable

import torch
from tqdm import tqdm

from .base import Sampler, Step


def _log_snr(sigma: float) -> float:
    return math.log((1 - sigma) / sigma)


def _coefficients(previous: list[float], next_sigma: float):
    sigma0 = previous[-1]
    h = _log_snr(next_sigma) - _log_snr(sigma0)
    hh = -h
    bh = hh
    phi1 = math.expm1(hh)
    phi = phi1 / hh - 1
    rks = [(_log_snr(s) - _log_snr(sigma0)) / h for s in previous[-2::-1]][:2]
    order = len(rks) + 1
    rks.append(1.0)
    rows, rhs = [], []
    fac = 1
    for i in range(order):
        rows.append([rk ** i for rk in rks])
        rhs.append(phi * fac / bh)
        fac *= i + 2
        phi = phi / hh - 1 / fac
    matrix = torch.tensor(rows, dtype=torch.float64)
    target = torch.tensor(rhs, dtype=torch.float64)
    corrector = torch.linalg.solve(matrix, target).tolist() if order > 1 else [0.5]
    predictor = ([0.5] if order == 2 else
                 torch.linalg.solve(matrix[:-1, :-1], target[:-1]).tolist()) if order > 1 else []
    return phi1, bh, predictor, corrector


class UniPCStepper:
    """Stateful flow solver; accepts any latent rank and a velocity callback."""

    def __init__(self):
        self.history: deque[tuple[float, torch.Tensor]] = deque(maxlen=3)
        self.previous_x0: torch.Tensor | None = None

    def step(self, x: torch.Tensor, sigma: float, next_sigma: float,
             velocity: Callable[[torch.Tensor, float], torch.Tensor]) -> torch.Tensor:
        x0 = self.previous_x0
        if x0 is None:
            x0 = x - sigma * velocity(x, sigma)
        if sigma == 1.0 or next_sigma == 0.0 or not self.history:
            out = x + (next_sigma - sigma) * ((x - x0) / sigma)
            self.previous_x0 = None
        else:
            previous = [(s, p) for s, p in self.history if 0 < s < 1]
            if not previous:
                out = x + (next_sigma - sigma) * ((x - x0) / sigma)
                self.previous_x0 = None
            else:
                old = [s for s, _ in previous] + [sigma]
                phi1, bh, rho_p, rho_c = _coefficients(old, next_sigma)
                base = next_sigma / sigma * x - (1 - next_sigma) * phi1 * x0
                deltas = []
                h = _log_snr(next_sigma) - _log_snr(sigma)
                for old_sigma, old_x0 in reversed(previous[-2:]):
                    rk = (_log_snr(old_sigma) - _log_snr(sigma)) / h
                    deltas.append((old_x0 - x0) / rk)
                pred = base
                for rho, delta in zip(rho_p, deltas):
                    pred = pred - (1 - next_sigma) * bh * rho * delta
                pred_x0 = pred - next_sigma * velocity(pred, next_sigma)
                correction = rho_c[-1] * (pred_x0 - x0)
                for rho, delta in zip(rho_c[:-1], deltas):
                    correction = correction + rho * delta
                out = base - (1 - next_sigma) * bh * correction
                self.previous_x0 = pred_x0
        self.history.append((sigma, x0))
        return out


class UniPCSampler(Sampler):
    """The same solver exposed to the regular image-model sampler registry."""

    def sample(self, x, schedule: list[Step], cond, guidance_scale: float,
               seed: int, desc: str = "sampling",
               step_callback: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None) -> torch.Tensor:
        if not schedule:
            return x
        self.model.begin_sampling(schedule)
        solver = UniPCStepper()
        dtype = x.dtype
        for i, step in tqdm(enumerate(schedule), total=len(schedule), desc=desc):
            sigma = float(step.t)
            next_sigma = float(step.t - step.delta)
            def velocity(y, t):
                return self.model.denoise_step(
                    y.to(dtype), torch.as_tensor(t, device=y.device, dtype=dtype),
                    cond, guidance_scale, i,
                ).float()
            x = solver.step(x.float(), sigma, max(0.0, next_sigma), velocity).to(dtype)
            if step_callback is not None:
                x = step_callback(x, i + 1, len(schedule))
        return x
