from __future__ import annotations

import math

import torch
from torch import Tensor


def fourier_frequencies(num_freqs: int, *, include_pi: bool = True,
                        device=None) -> Tensor:
    """``[num_freqs]`` doublings ``2**i``, optionally every one scaled by pi."""
    freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32, device=device)
    return freqs * math.pi if include_pi else freqs


def fourier_features(coords: Tensor, num_freqs: int, *, include_input: bool = True,
                     include_pi: bool = True) -> Tensor:
    """``[..., dim]`` -> ``[..., dim * (2 * num_freqs + 1)]``: coords, then sines, then cosines."""
    # Angles in fp32: 128 * pi is where an fp16 frequency stops being exact.
    angles = (coords[..., None].float()
              * fourier_frequencies(num_freqs, include_pi=include_pi,
                                    device=coords.device)).flatten(-2)
    sin, cos = angles.sin().to(coords.dtype), angles.cos().to(coords.dtype)
    parts = (coords, sin, cos) if include_input else (sin, cos)
    return torch.cat(parts, dim=-1)


def fourier_dim(dim: int, num_freqs: int, *, include_input: bool = True) -> int:
    """Width of :func:`fourier_features` for ``dim`` coordinates."""
    return dim * (2 * num_freqs + (1 if include_input else 0))


__all__ = ["fourier_dim", "fourier_features", "fourier_frequencies"]
