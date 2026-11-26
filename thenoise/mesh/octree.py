from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Tuple

import torch
from torch import Tensor

CORNERS = torch.tensor(list(product((0, 1), repeat=3)), dtype=torch.long)
MIN_LEVEL_RESOLUTION = 63
CHUNK = 8192


SdfFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class SdfVolume:
    """Sampled SDF: ``values``/``sampled`` are ``[R+1, R+1, R+1]`` corners, ``cells`` the ``[R, R, R]`` holding the level."""

    values: Tensor
    cells: Tensor
    sampled: Tensor
    resolution: int
    bounds: float
    level: float
    queries: int

    def extracted_cells(self) -> Tensor:
        """Indices of the cells holding the surface."""
        return self.cells.nonzero()

    def span(self) -> str:
        """The sampled field's range, which says whether it ever reached the cut level."""
        seen = self.values[self.sampled]
        return "nothing sampled" if seen.numel() == 0 else f"{seen.min():.4f}..{seen.max():.4f}"

    def corner_positions(self, cells: Tensor) -> Tensor:
        """``[K, 3]`` cell indices -> ``[K, 8, 3]`` points in ``[-bounds, bounds]``."""
        index = cells[:, None, :] + CORNERS.to(cells.device)
        return self.corner_coordinates(index)

    def corner_coordinates(self, index: Tensor) -> Tensor:
        """Corner grid indices -> points in ``[-bounds, bounds]``."""
        return -self.bounds + 2.0 * self.bounds * index.to(torch.float32) / self.resolution


def octree_levels(resolution: int, start: int = MIN_LEVEL_RESOLUTION) -> Tuple[int, ...]:
    """Resolutions sampled coarsest first; each must divide the next, so a coarse corner is also a fine corner."""
    if resolution < 1:
        raise ValueError(f"octree resolution must be positive, got {resolution}")
    if start < 1:
        raise ValueError(f"octree start must be positive, got {start}")
    coarse = resolution
    while coarse % 2 == 0:
        coarse //= 2
    while coarse < start and coarse < resolution:
        coarse *= 2
    levels = []
    while coarse < resolution:
        levels.append(coarse)
        coarse *= 2
    levels.append(resolution)
    return tuple(levels)


def _cells_of(size: int, device) -> Tensor:
    axis = torch.arange(size, device=device)
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)


def _corner_grid_index(index: Tensor, size: int) -> Tensor:
    """One integer per corner of a ``size`` grid, so a 1D unique finds the duplicates."""
    stride = size + 1
    return (index[:, 0] * stride + index[:, 1]) * stride + index[:, 2]


def _unpack(grid: Tensor, stride: int) -> Tensor:
    plane = stride * stride
    return torch.stack((grid // plane, (grid // stride) % stride, grid % stride), dim=-1)


def _halo(cells: Tensor, crossing: Tensor, size: int, device) -> Tensor:
    """Which of ``cells`` touch a cell that holds the level, as a mask the same length."""
    held = torch.zeros((size,) * 3, dtype=torch.float32, device=device)
    held[cells[crossing, 0], cells[crossing, 1], cells[crossing, 2]] = 1.0
    halo = torch.nn.functional.max_pool3d(held[None, None], 3, stride=1, padding=1)
    return halo[0, 0, cells[:, 0], cells[:, 1], cells[:, 2]] > 0


def _evaluate(sdf: SdfFn, points: Tensor, chunk: int) -> Tensor:
    if points.shape[0] <= chunk:
        return sdf(points)
    return torch.cat([sdf(part) for part in points.split(chunk)])


@torch.no_grad()
def sample_sdf(sdf: SdfFn, resolution: int, *, device, bounds: float = 1.0,
               level: float = 0.0, chunk: int = CHUNK,
               start: int = MIN_LEVEL_RESOLUTION, band: float = 0.0) -> SdfVolume:
    """Narrow-band octree sampling of ``sdf`` (``[M, 3] -> [M]``), costing surface not volume."""
    if bounds <= 0:
        raise ValueError(f"bounds must be positive, got {bounds}")
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    levels = octree_levels(resolution, start)
    grid = torch.zeros((resolution + 1,) * 3, dtype=torch.float32, device=device)
    sampled = torch.zeros((resolution + 1,) * 3, dtype=torch.bool, device=device)
    inside = torch.zeros((resolution,) * 3, dtype=torch.bool, device=device)

    cells = _cells_of(levels[0], device)
    queried = 0
    holding = None
    for size in levels:
        step = resolution // size
        index = (cells[:, None, :] + CORNERS.to(device)).reshape(-1, 3)
        stride = size + 1
        unique, inverse = torch.unique(_corner_grid_index(index, size), return_inverse=True)
        positions = -bounds + 2.0 * bounds * _unpack(unique, stride).to(torch.float32) / size
        values = _evaluate(sdf, positions, chunk).to(grid.dtype)
        queried += positions.shape[0]
        corners = _unpack(unique, stride) * step
        grid[corners[:, 0], corners[:, 1], corners[:, 2]] = values
        sampled[corners[:, 0], corners[:, 1], corners[:, 2]] = True
        around = values[inverse.view(-1).view(cells.shape[0], 8)]
        lo, hi = around.amin(dim=1), around.amax(dim=1)
        crossing = (lo <= level) & (hi >= level)
        # ``band`` reads magnitude too: a saturated field hides a surface no corner straddles.
        held = (crossing if band <= 0.0
                else crossing | ((around - level).abs().amin(dim=1) < band))
        if size == levels[-1]:
            holding = cells[crossing]
            break
        cells = (cells[_halo(cells, held, size, device), None, :] * 2
                 + CORNERS.to(device)).reshape(-1, 3)
        if cells.shape[0] == 0:
            break

    if holding is not None and holding.shape[0]:
        inside[holding[:, 0], holding[:, 1], holding[:, 2]] = True

    return SdfVolume(
        values=grid,
        cells=inside,
        sampled=sampled,
        resolution=resolution,
        bounds=bounds,
        level=level,
        queries=queried,
    )
