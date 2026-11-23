from __future__ import annotations

import math

import torch
from torch import Tensor


def fourier_frequencies(num_freqs: int, *, include_pi: bool, device=None) -> Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32, device=device)
    return freqs * math.pi if include_pi else freqs


def fourier_features(coords: Tensor, num_freqs: int, *, include_pi: bool,
                     include_input: bool = True) -> Tensor:
    # Angles in fp32: the top doubling is 128, and sin of a large angle is noise in fp16.
    angles = (coords[..., None].float()
              * fourier_frequencies(num_freqs, include_pi=include_pi,
                                    device=coords.device)).flatten(-2)
    sin, cos = angles.sin().to(coords.dtype), angles.cos().to(coords.dtype)
    parts = (coords, sin, cos) if include_input else (sin, cos)
    return torch.cat(parts, dim=-1)


def fourier_dim(dim: int, num_freqs: int, *, include_input: bool = True) -> int:
    return dim * (2 * num_freqs + (1 if include_input else 0))


def sinusoidal_positions(embed_dim: int, positions: Tensor) -> Tensor:
  # fp32, not the release's float64: the widest step is ~1e-4 and it lands on bf16.
    steps = torch.arange(embed_dim // 2, dtype=torch.float32, device=positions.device)
    omega = 1.0 / 10000.0 ** (steps / (embed_dim // 2))
    angles = positions.reshape(-1).to(torch.float32)[:, None] * omega
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


__all__ = ["fourier_dim", "fourier_features", "fourier_frequencies",
           "sinusoidal_positions"]
