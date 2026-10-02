"""ER-SDE sampler: higher-order stochastic flow solver (CONST-style).

One ``denoise_step`` per schedule step, plus second-order corrections and
stochastic noise. ``denoised`` is the x0 prediction ``x - sigma * v`` rebuilt
from the model's velocity.
"""
from __future__ import annotations

from typing import List

import torch
from tqdm import tqdm

from .base import Sampler, Step


class ErSdeSampler(Sampler):
    def sample(
        self,
        x: torch.Tensor,
        schedule: List[Step],
        cond,
        guidance_scale: float,
        seed: int,
        desc: str = "sampling",
    ) -> torch.Tensor:
        model = self.model
        dtype = x.dtype
        s_noise = 1.0

        def noise_scaler(t):
            return t * (torch.exp(t ** 0.3) + 10.0)

        sigmas = torch.tensor(
            [step.t for step in schedule] + [0.0],
            device=x.device,
            dtype=torch.float32,
        )
        if sigmas[0].item() >= 1.0:
            # sigma / (1 - sigma) must stay finite.
            sigmas[0] = model.percent_to_sigma(1e-4)

        half_log_snrs = -torch.log(sigmas / (1.0 - sigmas))
        er_lambdas = (-half_log_snrs).exp()

        generator = torch.Generator(device=x.device).manual_seed(seed)
        num_points = 200.0
        point_indice = torch.arange(0, num_points, dtype=torch.float32, device=x.device)

        old_denoised = None
        old_denoised_d = None
        for i in tqdm(range(len(sigmas) - 1), desc=desc):
            sigma_i = sigmas[i]
            v = model.denoise_step(x, sigma_i.to(dtype), cond, guidance_scale, i)
            xf = x.float()
            denoised = xf - sigma_i * v.float()

            if sigmas[i + 1] == 0:
                x = denoised.to(dtype)
            else:
                er_lambda_s = er_lambdas[i]
                er_lambda_t = er_lambdas[i + 1]
                alpha_s = sigmas[i] / er_lambda_s
                alpha_t = sigmas[i + 1] / er_lambda_t
                r_alpha = alpha_t / alpha_s
                r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

                # Stage 1 (Euler).
                xf = r_alpha * r * xf + alpha_t * (1.0 - r) * denoised

                stage_used = min(3, i + 1)
                if stage_used >= 2:
                    dt = er_lambda_t - er_lambda_s
                    lambda_step_size = -dt / num_points
                    lambda_pos = er_lambda_t + point_indice * lambda_step_size
                    scaled_pos = noise_scaler(lambda_pos)
                    s = torch.sum(1.0 / scaled_pos) * lambda_step_size
                    denoised_d = (denoised - old_denoised) / (
                        er_lambda_s - er_lambdas[i - 1]
                    )
                    xf = xf + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                    if stage_used >= 3:
                        s_u = torch.sum(
                            (lambda_pos - er_lambda_s) / scaled_pos
                        ) * lambda_step_size
                        denoised_u = (denoised_d - old_denoised_d) / (
                            (er_lambda_s - er_lambdas[i - 2]) / 2
                        )
                        xf = xf + alpha_t * (
                            (dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)
                        ) * denoised_u
                    old_denoised_d = denoised_d

                if s_noise > 0:
                    noise = torch.randn_like(xf, generator=generator)
                    noise_term = (
                        er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2
                    ).sqrt().nan_to_num(nan=0.0)
                    xf = xf + alpha_t * noise * s_noise * noise_term

                x = xf.to(dtype)

            old_denoised = denoised

        return x
