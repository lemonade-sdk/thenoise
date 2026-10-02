"""Nyquist notch: removes the 2px checkerboard grid artifact."""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Binomial kernel modulated by (-1)^n, so it isolates the Nyquist (2px) band.
_B = [-1.0, 6.0, -15.0, 20.0, -15.0, 6.0, -1.0]


def nyquist_notch(pixels: torch.Tensor) -> torch.Tensor:
    """Remove the 2px grid artifact from ``[C, H, W]`` pixels.

    Subtracting the x and y passes removes the grid but subtracts the 2D term
    twice, hence it is added back.
    """
    c, _h, _w = pixels.shape
    rgb = pixels[: min(3, c)]
    if rgb.shape[0] == 0:
        return pixels

    k = torch.tensor(_B, dtype=pixels.dtype, device=pixels.device) / 64.0
    kx = k.view(1, 1, 1, 7).expand(3, 1, 1, 7).contiguous()
    ky = k.view(1, 1, 7, 1).expand(3, 1, 7, 1).contiguous()
    kxy = (k[:, None] * k[None, :]).view(1, 1, 7, 7).expand(3, 1, 7, 7).contiguous()

    x = F.pad(rgb.unsqueeze(0), (3, 3, 3, 3), mode="replicate")

    bx = F.conv2d(x, kx, groups=3)[:, :, 3:-3, :]
    by = F.conv2d(x, ky, groups=3)[:, :, :, 3:-3]
    bxy = F.conv2d(x, kxy, groups=3)

    notched = rgb.unsqueeze(0) - bx - by + bxy

    if c > 3:
        return torch.cat([notched.squeeze(0), pixels[3:]], dim=0)
    return notched.squeeze(0)


__all__ = ["nyquist_notch"]
