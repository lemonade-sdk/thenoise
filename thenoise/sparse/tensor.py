from __future__ import annotations

from typing import Optional, Sequence

import torch

from .maps import MapCache


class SparseTensor:
    """``feats [M, C]`` on ``coords [M, 4]`` of ``(batch, x, y, z)``.

    ``coords`` is never mutated: operations replace ``feats``, which is what lets the maps
    be shared down a stack of layers.
    """

    def __init__(self, feats: torch.Tensor, coords: torch.Tensor,
                 shape: Sequence[int]) -> None:
        if feats.ndim != 2:
            raise ValueError(f"features are [M, C], got {tuple(feats.shape)}")
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"coordinates are [M, 4], got {tuple(coords.shape)}")
        if feats.shape[0] != coords.shape[0]:
            raise ValueError(f"{feats.shape[0]} features on {coords.shape[0]} voxels")
        if coords.device != feats.device:
            raise ValueError(f"coordinates are on {coords.device} and features on "
                             f"{feats.device}; a lattice lives on one device")
        if len(shape) != 4:
            raise ValueError(f"shape is (batch, width, height, depth), got {tuple(shape)}")
        self.feats = feats
        self.coords = coords if coords.dtype == torch.int32 else coords.to(torch.int32)
        self.shape = tuple(int(v) for v in shape)
        self.maps = MapCache(self.coords, self.shape)

    def __len__(self) -> int:
        return self.feats.shape[0]

    @property
    def device(self) -> torch.device:
        return self.feats.device

    @property
    def dtype(self) -> torch.dtype:
        return self.feats.dtype

    def replace(self, feats: torch.Tensor) -> "SparseTensor":
        """The same voxels carrying new features, and the same maps."""
        if feats.shape[0] != len(self):
            raise ValueError(f"{feats.shape[0]} features for {len(self)} voxels")
        out = object.__new__(SparseTensor)
        out.feats, out.coords, out.shape, out.maps = feats, self.coords, self.shape, self.maps
        return out

    def subset(self, selection: torch.Tensor) -> "SparseTensor":
        """The voxels a mask or an index keeps: a new lattice, so new maps."""
        return SparseTensor(self.feats[selection], self.coords[selection], self.shape)

    def concat(self, other: "SparseTensor") -> "SparseTensor":
        """One lattice holding both voxel sets; repeated coordinates stay repeated."""
        if self.shape != other.shape:
            raise ValueError(f"cannot join grid {self.shape} with {other.shape}")
        return SparseTensor(torch.cat([self.feats, other.feats]),
                            torch.cat([self.coords, other.coords]), self.shape)

    def to_dense(self) -> torch.Tensor:
        """``[batch, C, W, H, D]`` with each voxel scattered into its cell, rest zero."""
        batch, width, height, depth = self.shape
        cells = self.feats.new_zeros((batch, width, height, depth, self.feats.shape[1]))
        cells[self.coords[:, 0].long(), self.coords[:, 1].long(), self.coords[:, 2].long(),
              self.coords[:, 3].long()] = self.feats
        return cells.permute(0, 4, 1, 2, 3).contiguous()

    @classmethod
    def from_dense(cls, dense: torch.Tensor, support: Optional[torch.Tensor] = None
                   ) -> "SparseTensor":
        """The cells of a ``[batch, C, W, H, D]`` tensor that ``support`` marks, or all of them.

        Occupancy is decided apart from the features, so it arrives as a parameter rather
        than being read off a magnitude.
        """
        if dense.ndim != 5:
            raise ValueError(f"a dense lattice is [batch, C, W, H, D], got {tuple(dense.shape)}")
        batch, channels, width, height, depth = dense.shape
        cells = dense.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        if support is None:
            index = torch.arange(cells.shape[0], device=dense.device)
        else:
            index = support.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        coords = torch.stack((index // (width * height * depth),
                              (index // (height * depth)) % width,
                              (index // depth) % height,
                              index % depth), dim=1)
        return cls(cells[index], coords, (batch, width, height, depth))

    def __repr__(self) -> str:
        return (f"SparseTensor({len(self)} voxels, {self.feats.shape[1]} channels, "
                f"grid {self.shape}, {self.dtype})")


__all__ = ["SparseTensor"]
