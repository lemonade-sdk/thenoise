"""Tensor post-processing filters for decoded pixel tensors.

Nyquist Notch that removes 2px grid artifacts (the
alternating checkerboard pattern a VAE decoder can leave behind).

These functions operate on the fp32 GPU pixel tensor ``[C, H, W]`` produced by
``DiffusionModel.decode`` (values in ``[-1, 1]``) and return the same shape.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Nyquist Notch — removes 2px grid artifacts. Place before deconvolution.
# b = [-1, +6, -15, +20, -15, +6, -1] / 64  (binomial * (-1)^n, +1 at center)
_B = [-1.0, 6.0, -15.0, 20.0, -15.0, 6.0, -1.0]


def nyquist_notch(pixels: torch.Tensor) -> torch.Tensor:
    """Apply the Nyquist Notch filter to ``[C, H, W]`` pixels (in place-safe).

    Port of the GLSL shader:

        Bx   = convolution of center with K along x
        By   = convolution of center with K along y
        Bxy  = separable 2D convolution with K ⊗ K
        out  = center - Bx - By + Bxy

    where ``K = B / 64``. ``B`` is the binomial kernel ``[1, 6, 15, 20, 15, 6, 1]``
    modulated by ``(-1)^n``, so it isolates the Nyquist-frequency (2px) component;
    subtracting it (and re-adding the 2D term to avoid double-counting) removes
    the grid artifact while leaving the rest of the image intact.

    Only the first three (RGB) channels are filtered; any extra channels pass
    through unchanged.
    """
    c, _h, _w = pixels.shape
    rgb = pixels[: min(3, c)]
    if rgb.shape[0] == 0:
        return pixels

    k = torch.tensor(_B, dtype=pixels.dtype, device=pixels.device) / 64.0
    # 1x7 / 7x1 / 7x7 kernels, replicated per channel for grouped convolution.
    kx = k.view(1, 1, 1, 7).expand(3, 1, 1, 7).contiguous()
    ky = k.view(1, 1, 7, 1).expand(3, 1, 7, 1).contiguous()
    kxy = (k[:, None] * k[None, :]).view(1, 1, 7, 7).expand(3, 1, 7, 7).contiguous()

    # Clamp-to-edge sampling: pad with edge replication so border texels sample the edge value instead of zero.
    x = F.pad(rgb.unsqueeze(0), (3, 3, 3, 3), mode="replicate")  # [1, 3, H+6, W+6]

    bx = F.conv2d(x, kx, groups=3)[:, :, 3:-3, :]   # 1x7: trim the H pad
    by = F.conv2d(x, ky, groups=3)[:, :, :, 3:-3]   # 7x1: trim the W pad
    bxy = F.conv2d(x, kxy, groups=3)                # 7x7: already [1, 3, H, W]

    notched = rgb.unsqueeze(0) - bx - by + bxy  # [1, 3, H, W]

    if c > 3:
        return torch.cat([notched.squeeze(0), pixels[3:]], dim=0)
    return notched.squeeze(0)


__all__ = ["nyquist_notch"]
