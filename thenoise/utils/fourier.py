from __future__ import annotations

import math

import torch
from torch import Tensor


def fourier_frequencies(num_freqs: int, *, include_pi: bool, device=None) -> Tensor:
    """``[num_freqs]`` doublings ``2**i``, optionally every one scaled by pi."""
    freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32, device=device)
    return freqs * math.pi if include_pi else freqs


def fourier_features(coords: Tensor, num_freqs: int, *, include_pi: bool,
                     include_input: bool = True) -> Tensor:
    """``[..., dim]`` -> ``[..., dim * (2 * num_freqs + 1)]``: coords, then sines, then cosines."""
    # Angles in fp32: the top doubling is 128, and sin of a large angle is noise in fp16.
    angles = (coords[..., None].float()
              * fourier_frequencies(num_freqs, include_pi=include_pi,
                                    device=coords.device)).flatten(-2)
    sin, cos = angles.sin().to(coords.dtype), angles.cos().to(coords.dtype)
    parts = (coords, sin, cos) if include_input else (sin, cos)
    return torch.cat(parts, dim=-1)


def fourier_dim(dim: int, num_freqs: int, *, include_input: bool = True) -> int:
    """Width of :func:`fourier_features` for ``dim`` coordinates."""
    return dim * (2 * num_freqs + (1 if include_input else 0))


def sinusoidal_positions(embed_dim: int, positions: Tensor) -> Tensor:
    """``[M]`` positions -> ``[M, embed_dim]``: sines then cosines of ``pos / 10000**(2i/D)``.

    Unlike :func:`fourier_features` the input is not part of the output, because what is
    being identified is the position itself -- Hunyuan3D's multi-view conditioner marks
    each view of a picture this way and adds the result to that view's tokens. Device comes
    from ``positions``, the only place it is known.
    """
    # fp32 rather than the release's float64: the widest step is still ~1e-4 and the result
    # is added to bf16 tokens, and fp64 is not something to spend on an inference path.
    steps = torch.arange(embed_dim // 2, dtype=torch.float32, device=positions.device)
    omega = 1.0 / 10000.0 ** (steps / (embed_dim // 2))
    angles = positions.reshape(-1).to(torch.float32)[:, None] * omega
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


__all__ = ["fourier_dim", "fourier_features", "fourier_frequencies",
           "sinusoidal_positions"]
