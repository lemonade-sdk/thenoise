"""Shared token position-id builders for the RoPE input side.

The DiT models each build a ``[num_tokens, n_axes]`` coordinate tensor that feeds
``matrix_rope`` (via ``RopeCache``). The grid math is identical across models —
row-major ``meshgrid("ij")`` over per-axis ``arange`` — and the only differences
are the two conventions that are easy to get subtly wrong:

  * a per-axis ``start`` offset (Z-Image's image grid begins at ``cap_len + 1``),
  * a per-axis centered offset ``r - ceil(size / 2)`` (Qwen-Image's h/w axes).

``grid_from_axes`` is the primitive (used when an axis has custom values, e.g.
Flux.2's video ``t`` coordinate); ``grid_positions`` is the convenience wrapper
that builds the axes from sizes/start/centered.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def grid_from_axes(axes: list[Tensor]) -> Tensor:
    """``[prod(len(axes)), len(axes)]`` row-major grid from precomputed 1D axes."""
    grids = torch.meshgrid(axes, indexing="ij")
    return torch.stack(grids, dim=-1).reshape(-1, len(axes))


def grid_positions(
    sizes: list[int],
    *,
    start: Optional[list[int]] = None,
    centered: Optional[list[bool]] = None,
    dtype=torch.float32,
    device=None,
) -> Tensor:
    """``[prod(sizes), len(sizes)]`` row-major grid of per-axis coordinates.

    ``start[i]`` shifts axis ``i`` to ``[start[i], start[i] + sizes[i])``.
    ``centered[i]`` applies the ``r - ceil(sizes[i] / 2)`` convention (Qwen-Image
    uses it on the h/w axes only). ``dtype`` is preserved so callers keep their
    exact coordinate type (Z-Image uses ``int32``, others ``float32``).
    """
    axes = []
    for i, size in enumerate(sizes):
        x0 = 0 if start is None else start[i]
        axis = torch.arange(x0, x0 + size, dtype=dtype, device=device)
        if centered is not None and centered[i]:
            axis = axis - math.ceil(size / 2)
        axes.append(axis)
    return grid_from_axes(axes)


def broadcast_positions(
    seq_len: int,
    n_axes: int,
    *,
    offset: int = 0,
    dtype=torch.float32,
    device=None,
) -> Tensor:
    """``[seq_len, n_axes]`` with every axis equal to ``offset + arange(seq_len)``.

    Qwen-Image's text stream uses a single advancing index broadcast across all
    three axes (``pos_freqs[max_vid_index + j]``).
    """
    k = torch.arange(offset, offset + seq_len, dtype=dtype, device=device)
    return k[:, None].expand(seq_len, n_axes)


__all__ = ["grid_from_axes", "grid_positions", "broadcast_positions"]
