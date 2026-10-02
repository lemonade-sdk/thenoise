"""Film grain: luminance-only, spatially-correlated noise."""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Gaussian 5x5 kernel (sigma ~ 1.0).
_G = [
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
]
_G_SUM = sum(row for row in _G for row in row)


def film_grain(
    pixels: torch.Tensor,
    *,
    strength: float = 0.03,
    seed: int | None = None,
) -> torch.Tensor:
    """Add film-grain-like noise to the luminance of ``[C, H, W]`` pixels.

    Gaussian noise, spatially correlated by a 5x5 blur, then added equally to
    every RGB channel — a uniform channel delta is a pure luminance shift.
    Typical ``strength`` is 0.01 (barely visible) to 0.08 (pronounced).
    """
    c, h, w = pixels.shape
    rgb = pixels[: min(3, c)]

    if strength == 0.0:
        return pixels

    noise = torch.empty(1, 1, h, w, dtype=pixels.dtype, device=pixels.device)
    if seed is not None:
        gen = torch.Generator(device=pixels.device)
        gen.manual_seed(seed)
        noise.normal_(generator=gen)
    else:
        noise.normal_()

    kernel = torch.tensor(_G, dtype=pixels.dtype, device=pixels.device) / _G_SUM
    kernel = kernel.view(1, 1, 5, 5)

    noise = F.pad(noise, (2, 2, 2, 2), mode="replicate")
    noise = F.conv2d(noise, kernel)

    grain = noise * strength

    grained_rgb = rgb.unsqueeze(0) + grain  # broadcasts over C

    if c > 3:
        return torch.cat([grained_rgb.squeeze(0), pixels[3:]], dim=0)
    return grained_rgb.squeeze(0)


__all__ = ["film_grain"]
