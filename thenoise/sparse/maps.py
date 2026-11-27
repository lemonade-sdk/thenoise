from __future__ import annotations

from typing import Sequence, Tuple

import torch

#: Bounds the candidates and keys a lookup holds at once.
CHUNK = 1 << 20


def strides(shape: Sequence[int]) -> Tuple[int, int, int, int]:
    """The flat-key strides of ``(batch, width, height, depth)``."""
    if len(shape) != 4:
        raise ValueError(f"shape is (batch, width, height, depth), got {tuple(shape)}")
    _, width, height, depth = shape
    return width * height * depth, height * depth, depth, 1


def flat_keys(coords: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """The volume's flat index per voxel: injective over coordinates inside the grid."""
    batch_stride, x_stride, y_stride, z_stride = strides(shape)
    return (coords[:, 0].long() * batch_stride + coords[:, 1].long() * x_stride
            + coords[:, 2].long() * y_stride + coords[:, 3].long() * z_stride)


class HashMap:
    """Sorted-array membership: a value for a known key, ``-1`` for anything else."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.numel() != values.numel():
            raise ValueError(f"{keys.numel()} keys for {values.numel()} values")
        order = torch.argsort(keys)
        self.keys = keys[order]
        self.values = values[order]

    def lookup(self, query: torch.Tensor) -> torch.Tensor:
        """``query``'s values, ``-1`` where the key is absent."""
        out = torch.empty(query.numel(), dtype=torch.int32, device=query.device)
        flat = query.reshape(-1)
        n = self.keys.numel()
        for begin in range(0, flat.numel(), CHUNK):
            part = flat[begin:begin + CHUNK]
            position = torch.searchsorted(self.keys, part)
            hit = position < n  # past the end is a miss; clamp so the read below is legal
            position = torch.where(hit, position, torch.zeros_like(position))
            found = hit & (self.keys[position] == part)
            out[begin:begin + part.numel()] = torch.where(found, self.values[position], -1)
        return out.view(query.shape)


def kernel_offsets(kernel: Sequence[int], dilation: Sequence[int],
                   device: torch.device) -> torch.Tensor:
    """``[V, 3]`` tap offsets, ordered as ``(t, r, s)`` in C order to match a weight's taps."""
    t, r, s = kernel
    dt, dr, ds = dilation
    return torch.tensor([[(i - t // 2) * dt, (j - r // 2) * dr, (k - s // 2) * ds]
                         for i in range(t) for j in range(r) for k in range(s)],
                        device=device, dtype=torch.long)


class MapCache:
    """Neighbour maps for one coordinate set, built once per kernel and dilation."""

    def __init__(self, coords: torch.Tensor, shape: Sequence[int]) -> None:
        self.coords = coords
        self.shape = tuple(shape)
        self.keys = flat_keys(coords, self.shape)
        self.hash = HashMap(self.keys, torch.arange(self.keys.numel(), dtype=torch.int32,
                                                   device=coords.device))
        self._neighbors: dict[Tuple[Tuple[int, ...], Tuple[int, ...]], torch.Tensor] = {}

    def neighbor(self, kernel: Sequence[int] = (3, 3, 3), dilation: int = 1) -> torch.Tensor:
        """``[M, V]`` row indices of each voxel's taps, ``-1`` where there is none."""
        taps = tuple(int(v) for v in kernel)
        dil = (dilation,) * 3 if isinstance(dilation, int) else tuple(int(v) for v in dilation)
        found = self._neighbors.get((taps, dil))
        if found is None:
            found = self._build(taps, dil)
            self._neighbors[(taps, dil)] = found
        return found

    def _build(self, taps: Sequence[int], dilation: Sequence[int]) -> torch.Tensor:
        _, width, height, depth = self.shape
        batch_stride, x_stride, y_stride, z_stride = strides(self.shape)
        offsets = kernel_offsets(taps, dilation, self.coords.device)
        rows = max(CHUNK // (offsets.shape[0] * 4), 1)
        miss = torch.tensor(-1, dtype=torch.int32, device=self.coords.device)
        out = torch.empty((self.keys.numel(), offsets.shape[0]), dtype=torch.int32,
                          device=self.coords.device)
        for begin in range(0, self.keys.numel(), rows):
            part = self.coords[begin:begin + rows]
            cell = part[:, None, 1:].long() + offsets
            keys = (part[:, None, 0].long() * batch_stride + cell[..., 0] * x_stride
                    + cell[..., 1] * y_stride + cell[..., 2] * z_stride)
            # A tap off the edge has the flat key of the next line, or of the next batch:
            # out of the grid is a miss however the lookup answers it.
            inside = ((cell[..., 0] >= 0) & (cell[..., 0] < width)
                      & (cell[..., 1] >= 0) & (cell[..., 1] < height)
                      & (cell[..., 2] >= 0) & (cell[..., 2] < depth))
            out[begin:begin + part.shape[0]] = torch.where(
                inside, self.hash.lookup(keys), miss)
        return out


__all__ = ["CHUNK", "HashMap", "MapCache", "flat_keys", "kernel_offsets", "strides"]
