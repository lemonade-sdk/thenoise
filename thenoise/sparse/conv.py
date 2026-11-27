from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .maps import flat_keys
from .tensor import SparseTensor


class SubmanifoldConv3d(nn.Module):
    """A convolution over the support only: a tap on an empty cell reads zero, and a plain
    convolution would fill the volume out from one voxel in three layers.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = ((kernel_size,) * 3 if isinstance(kernel_size, int)
                            else tuple(int(v) for v in kernel_size))
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size,
                                               device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype)) \
            if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_channels * self.weight.shape[2] ** 3)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(submanifold_conv(
            x.feats, x.maps.neighbor(self.kernel_size, self.dilation), self.weight, self.bias))


def submanifold_conv(feats: torch.Tensor, neighbor: torch.Tensor,
                     weight: torch.Tensor, bias: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
    """One GEMM per kernel tap over the rows the map names; ``-1`` reads the spare zero row."""
    in_channels = feats.shape[1]
    if weight.shape[1] != in_channels:
        raise ValueError(f"weight takes {weight.shape[1]} channels, lattice carries {in_channels}")
    taps = weight.permute(2, 3, 4, 1, 0).reshape(-1, in_channels, weight.shape[0])
    if taps.shape[0] != neighbor.shape[1]:
        raise ValueError(f"weight has {taps.shape[0]} taps, map has {neighbor.shape[1]}")
    empty = torch.cat([feats, feats.new_zeros((1, in_channels))])
    out = feats.new_zeros((feats.shape[0], weight.shape[0]))
    if bias is not None:
        out += bias
    for tap in range(taps.shape[0]):
        index = neighbor[:, tap].long()
        out += empty[torch.where(index >= 0, index, feats.shape[0])] @ taps[tap]
    return out


class SubmanifoldConvNeXtBlock3d(nn.Module):
    """Conv, normalise, activate, conv, with the input added back."""

    def __init__(self, channels: int, dilation: int = 1, groups: int = 32,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.conv1 = SubmanifoldConv3d(channels, channels, 3, dilation=dilation, bias=False)
        self.conv2 = SubmanifoldConv3d(channels, channels, 3, dilation=dilation)
        self.norm = nn.GroupNorm(groups, channels, eps=eps) if channels % groups == 0 \
            else nn.GroupNorm(1, channels, eps=eps)
        self.act = nn.SiLU()

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv1(x)
        out = out.replace(self.act(self.norm(out.feats)))
        return out.replace(x.feats + self.conv2(out).feats)


def sparse_mean_pool3d(x: SparseTensor, size: int = 2
                       ) -> Tuple[SparseTensor, torch.Tensor]:
    """Coarsen by ``size``, averaging the voxels in each new cell.

    The second answer maps every old voxel to its new row, which is what the nearest
    upsample and any residual across the downsample walk back along.
    """
    if size < 1:
        raise ValueError(f"a pool size below 1 is not a lattice: {size}")
    batch, width, height, depth = x.shape
    coords = x.coords.clone()
    coords[:, 1] = torch.div(coords[:, 1], size, rounding_mode="floor")
    coords[:, 2] = torch.div(coords[:, 2], size, rounding_mode="floor")
    coords[:, 3] = torch.div(coords[:, 3], size, rounding_mode="floor")
    grid = (batch, -(-width // size), -(-height // size), -(-depth // size))
    keys = flat_keys(coords, grid)
    unique, inverse = torch.unique(keys, return_inverse=True)
    feats = x.feats.new_zeros((unique.numel(), x.feats.shape[1]))
    feats.index_add_(0, inverse, x.feats)
    counts = feats.new_zeros((unique.numel(), 1))
    counts.index_add_(0, inverse, feats.new_ones((len(x), 1)))
    first = torch.full((unique.numel(),), len(x), dtype=torch.long, device=x.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(x), device=x.device), reduce="amin",
                          include_self=False)
    return SparseTensor(feats / counts, coords[first], grid), inverse


def sparse_upsample3d_nearest(feats: torch.Tensor, pool_index: torch.Tensor) -> torch.Tensor:
    """Every child takes its parent's features, along the index the pool handed back."""
    return feats[pool_index.long()]


__all__ = ["SubmanifoldConv3d", "SubmanifoldConvNeXtBlock3d", "sparse_mean_pool3d",
           "sparse_upsample3d_nearest", "submanifold_conv"]
