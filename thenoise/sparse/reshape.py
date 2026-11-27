from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .conv import SubmanifoldConv3d
from .maps import flat_keys
from .tensor import SparseTensor


def child_offsets(factor: int, device: torch.device) -> torch.Tensor:
    """``[factor**3, 3]`` child offsets, x fastest: ``dx + dy*factor + dz*factor**2``.

    That is how these checkpoints order a cell's children; the tidier ``dx*factor**2 + ...``
    would permute every subdivision head and upsampled feature with no shape to complain.
    """
    return torch.tensor([(i, j, k) for k in range(factor) for j in range(factor)
                         for i in range(factor)], device=device, dtype=torch.int32)


def pixel_unshuffle_3d(dense: torch.Tensor, factor: int) -> torch.Tensor:
    """``[B, C, W, H, D]`` to ``[B, C*factor**3, W/f, H/f, D/f]``, child-major in channels."""
    batch, channels, width, height, depth = dense.shape
    if min(width, height, depth) % factor:
        raise ValueError(f"a shuffle of {factor} needs the grid divisible by {factor}, "
                         f"got {(width, height, depth)}")
    return dense.reshape(batch, channels, width // factor, factor, height // factor, factor,
                         depth // factor, factor).permute(0, 1, 7, 5, 3, 2, 4, 6).reshape(
        batch, channels * factor ** 3, width // factor, height // factor, depth // factor)


def pixel_shuffle_3d(dense: torch.Tensor, factor: int) -> torch.Tensor:
    """The inverse of :func:`pixel_unshuffle_3d`: channels back into finer cells."""
    batch, channels, width, height, depth = dense.shape
    if channels % factor ** 3:
        raise ValueError(f"{channels} channels do not split into {factor ** 3} cells")
    return dense.reshape(batch, channels // factor ** 3, factor, factor, factor, width,
                         height, depth).permute(0, 1, 5, 4, 6, 3, 7, 2).reshape(
        batch, channels // factor ** 3, width * factor, height * factor, depth * factor)


class SparseChannel2Spatial(nn.Module):
    """Every voxel becomes ``factor**3`` of them, splitting its channels between them.

    ``subdivision`` says which children are born: ``[M]`` keeps or drops a whole voxel,
    ``[M, factor**3]`` — what a subdivision head predicts — keeps the cells it marks.
    """

    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError(f"a subdivision of {factor} is not a refinement")
        self.factor = factor

    def plan(self, x: SparseTensor, subdivision: Optional[torch.Tensor] = None
             ) -> Tuple[SparseTensor, torch.Tensor]:
        """The finer lattice, and the parent split row each of its cells came from.

        An index into ``x.feats.reshape(M, factor**3, -1)``: a residual climbing through the
        refinement gathers along it rather than finding its parent by coordinate again.
        """
        factor = self.factor
        voxels, channels = x.feats.shape
        if channels % factor ** 3:
            raise ValueError(f"{channels} channels do not split into {factor ** 3} cells")
        per_cell = channels // factor ** 3
        offsets = child_offsets(factor, x.device)

        if subdivision is None:
            child, parent = torch.meshgrid(torch.arange(factor ** 3, device=x.device),
                                           torch.arange(voxels, device=x.device),
                                           indexing="ij")
            child, parent = child.reshape(-1), parent.reshape(-1)
        else:
            keep = subdivision.reshape(voxels, factor ** 3)
            both = keep.nonzero(as_tuple=False)
            parent, child = both[:, 0], both[:, 1]

        coords = x.coords[parent]
        coords[:, 1:] = coords[:, 1:] * factor + offsets[child]
        grid = (x.shape[0], x.shape[1] * factor, x.shape[2] * factor, x.shape[3] * factor)
        source = parent * factor ** 3 + child
        feats = x.feats.reshape(voxels * factor ** 3, per_cell)[source]
        return SparseTensor(feats, coords, grid), source

    def forward(self, x: SparseTensor, subdivision: Optional[torch.Tensor] = None
                ) -> SparseTensor:
        return self.plan(x, subdivision)[0]


class SparseSpatial2Channel(nn.Module):
    """``factor**3`` children become their parent, their channels concatenated.

    A parent missing a child is refused rather than filled in with an invented zero.
    """

    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: SparseTensor) -> SparseTensor:
        factor = self.factor
        parent = x.coords.clone()
        parent[:, 1:] = torch.div(parent[:, 1:], factor, rounding_mode="floor")
        place = ((x.coords[:, 1] - parent[:, 1] * factor)
                 + (x.coords[:, 2] - parent[:, 2] * factor) * factor
                 + (x.coords[:, 3] - parent[:, 3] * factor) * factor * factor).long()

        keys = flat_keys(parent, x.shape)
        unique, inverse = torch.unique(keys, return_inverse=True)
        counts = torch.zeros(unique.numel(), dtype=torch.long, device=x.device)
        counts.index_add_(0, inverse, torch.ones_like(inverse))
        if bool((counts != factor ** 3).any()):
            raise ValueError(f"{int((counts != factor ** 3).sum())} of {unique.numel()} "
                             f"parents do not have {factor ** 3} children")

        grid = (x.shape[0], -(-x.shape[1] // factor), -(-x.shape[2] // factor),
                -(-x.shape[3] // factor))
        feats = x.feats.new_zeros((unique.numel(), factor ** 3, x.feats.shape[1]))
        feats[inverse, place] = x.feats
        first = torch.full((unique.numel(),), len(x), dtype=torch.long, device=x.device)
        first.scatter_reduce_(0, inverse, torch.arange(len(x), device=x.device),
                              reduce="amin", include_self=False)
        return SparseTensor(feats.reshape(unique.numel(), -1), parent[first], grid)


__all__ = ["SparseChannel2Spatial", "SparseSpatial2Channel",
           "child_offsets", "pixel_shuffle_3d", "pixel_unshuffle_3d"]
