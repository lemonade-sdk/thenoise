"""RCAS — Robust Contrast Adaptive Sharpening.

A 5-tap cross filter whose sharpening lobe adapts to the local contrast of each
pixel, with minimal halo artifacts.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LOBE_MAX = -0.1875

_EPS = 1e-6


def rcas(pixels: torch.Tensor, *, strength: float = 0.5) -> torch.Tensor:
    """Sharpen ``[C, H, W]`` pixels in ``[-1, 1]`` with RCAS.

    The local min/max of the 5-tap cross gives a per-pixel ``lobe``: stronger in
    low-contrast regions, weaker at high-contrast edges, where it goes to 0 to
    keep the output from clipping. Typical ``strength`` is 0.3 to 0.8.
    """
    c, h, w = pixels.shape
    rgb = pixels[:3 if c >= 3 else c]

    if strength == 0.0:
        return pixels

    p = F.pad(rgb, (1, 1, 1, 1), mode="replicate")

    n = p[:, 0:h, 1:w + 1]
    s = p[:, 2:h + 2, 1:w + 1]
    w_ = p[:, 1:h + 1, 0:w]
    e = p[:, 1:h + 1, 2:w + 2]
    center = rgb

    mn = _min(n, s, w_, e)
    mx = _max(n, s, w_, e)

    lo = (mn + 1.0) * 0.5
    hi = (mx + 1.0) * 0.5
    mid = (center + 1.0) * 0.5

    hit_min = torch.minimum(lo, mid) / (hi * 4.0).clamp(min=_EPS)
    hit_max = (1.0 - torch.maximum(hi, mid)) / ((lo * 4.0 - 4.0).clamp(max=-_EPS))

    lobe = torch.max(-hit_min, hit_max).max(dim=0, keepdim=True).values

    lobe = (lobe * strength).clamp(_LOBE_MAX, 0.0)

    norm = (lobe * 4.0 + 1.0).reciprocal()
    neighbors = (n + s + w_ + e) * lobe + center
    sharpened = (neighbors * norm).clamp(-1.0, 1.0)

    if c > 3:
        return torch.cat([sharpened, pixels[3:]], dim=0)
    return sharpened


def _min(*tensors: torch.Tensor) -> torch.Tensor:
    out = tensors[0]
    for t in tensors[1:]:
        out = torch.min(out, t)
    return out


def _max(*tensors: torch.Tensor) -> torch.Tensor:
    out = tensors[0]
    for t in tensors[1:]:
        out = torch.max(out, t)
    return out


__all__ = ["rcas"]
