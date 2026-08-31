# SDXL discrete sampler helpers.
#
# SDXL is an epsilon-prediction (noise) model sampled with a discrete DDIM/euler
# schedule. The beta schedule is ComfyUI's "linear" (= diffusers
# "scaled_linear"): ``betas = (linspace(sqrt(0.00085), sqrt(0.012), 1000))**2``.
#
# For each discrete timestep index ``t`` (0..999) the noise level is
# ``sigma(t) = sqrt((1 - alphas_cumprod[t]) / alphas_cumprod[t])``. The euler
# update ``x -= (sigma_i - sigma_{i+1}) * eps`` reproduces the discrete
# euler/DDIM integration when ``denoise_step`` returns the predicted noise and
# the initial latent is scaled by the largest sigma.

import math

import numpy as np
import torch

#: SDXL beta schedule bounds (ComfyUI "linear" / diffusers "scaled_linear").
LINEAR_START = 0.00085
LINEAR_END = 0.012
NUM_TIMESTEPS = 1000


def get_alphas_cumprod(device=None, dtype=torch.float64) -> torch.Tensor:
    """``alphas_cumprod`` over 1000 discrete timesteps, index 0 = clean."""
    betas = torch.linspace(
        math.sqrt(LINEAR_START), math.sqrt(LINEAR_END), NUM_TIMESTEPS, dtype=dtype
    ) ** 2
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).to(device)


def rescale_zero_terminal_snr_alphas_cumprod(
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Zero-terminal-SNR rescaling of an ``alphas_cumprod`` grid (ComfyUI zsnr).

    Some anime SDXL checkpoints (e.g. noobai) are trained with a zero-terminal-SNR
    schedule: the noisiest timestep is shifted so the final ``alphas_cumprod``
    lands near zero instead of ``~1e-4``. Sampling such a model on the plain
    linear schedule produces garbage regardless of CFG/steps. Mirrors ComfyUI's
    ``rescale_zero_terminal_snr_sigmas`` (``model_sampling.py``) but operates on
    the ``alphas_cumprod`` grid directly, which ``sigma``/``_sigma_at`` consume.
    """
    sqrt_abar = alphas_cumprod.sqrt()
    sqrt_abar_0 = sqrt_abar[0].clone()
    sqrt_abar_T = sqrt_abar[-1].clone()
    # Shift so the last timestep is zero, then scale the first back to its old value.
    sqrt_abar = sqrt_abar - sqrt_abar_T
    sqrt_abar = sqrt_abar * sqrt_abar_0 / (sqrt_abar_0 - sqrt_abar_T)
    abar = sqrt_abar**2
    abar[-1] = 4.8973451890853435e-08
    return abar


def sigma(t: int, alphas_cumprod: torch.Tensor = None) -> float:
    """Noise level at discrete timestep ``t`` (0 = clean, 999 = noisy).

    ``alphas_cumprod`` is an optional precomputed grid; when omitted the full
    1000-element table is rebuilt, so callers that walk several timesteps should
    pass it in (see ``get_sigmas`` / ``sigmas_for_timesteps``).
    """
    abar = float((alphas_cumprod if alphas_cumprod is not None else get_alphas_cumprod())[t])
    return float(((1.0 - abar) / abar) ** 0.5)


def discrete_timesteps(steps: int) -> list[int]:
    """Discrete timestep indices, noise->clean, for ``steps`` euler steps."""
    timesteps = np.linspace(0, NUM_TIMESTEPS - 1, steps).round().astype(int)
    return timesteps[::-1].tolist()


def sigmas_for_timesteps(ts: list[int], alphas_cumprod=None) -> list[float]:
    """Sigma grid (noise->clean, trailing 0) for a timestep-index list.

    Computes the ``alphas_cumprod`` table once and reuses it across all steps,
    avoiding a fresh 1000-element cumprod per step. ``alphas_cumprod`` may be a
    precomputed grid (e.g. a zsnr-rescaled one) for callers that walk several
    timesteps.
    """
    abar = alphas_cumprod if alphas_cumprod is not None else get_alphas_cumprod()
    return [sigma(t, abar) for t in ts] + [0.0]


def get_sigmas(steps: int, alphas_cumprod=None) -> list[float]:
    """Sigma grid, noise->clean, with a trailing 0 (mirrors diffusers Euler)."""
    return sigmas_for_timesteps(discrete_timesteps(steps), alphas_cumprod)
