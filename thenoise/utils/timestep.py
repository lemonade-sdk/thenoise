"""Shared sinusoidal timestep embedding for the DiT models."""
from __future__ import annotations

import math

import torch


def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
    time_factor: float = 1000.0,
) -> torch.Tensor:
    """Sinusoidal timestep features ``[cos, sin]``, computed in fp32.

    ``t`` is the flow timestep in ``[0, 1]``; ``time_factor`` (default 1000)
    scales it before the sinusoid. Leading dims are preserved, so ``(B,) ->
    (B, dim)`` and ``(B, T) -> (B, T, dim)``. Odd ``dim`` is zero-padded to
    ``dim``, and the result is cast to ``t``'s dtype.
    """
    t_f = t.float() * time_factor
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t_f.device) / half
    )
    args = t_f[..., None] * freqs
    features = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        features = torch.cat([features, torch.zeros_like(features[..., :1])], dim=-1)
    if torch.is_floating_point(t):
        features = features.to(t.dtype)
    return features


__all__ = ["timestep_embedding"]
