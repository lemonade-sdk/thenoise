from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from thenoise.sparse import HashMap, SparseTensor, flat_keys

from .extract import Mesh, empty_mesh

#: The two ways to cut a quad, as corner indices below. The first draws the 0–2 diagonal, the
#: second the 1–3.
SPLITS = torch.tensor([[0, 1, 2, 0, 2, 3],
                       [0, 1, 3, 1, 2, 3]], dtype=torch.long)


def corner_ring(device: torch.device) -> Tensor:
    """``[3, 4, 3]`` — the voxels around an edge along each axis, in winding order.

    An edge along ``a`` is shared by four voxels: none away along ``a``, a unit square in the
    other two. Those are ``(a+1, a+2)`` so all three axes wind alike; reorder one and the
    sheets meet at a ridge inside out, quietly until the mesh is lit.
    """
    ring = torch.zeros(3, 4, 3, dtype=torch.int32)
    square = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.int32)
    for axis in range(3):
        around = [(axis + 1) % 3, (axis + 2) % 3]
        ring[axis, :, around[0]] = square[:, 0]
        ring[axis, :, around[1]] = square[:, 1]
    return ring.to(device)


def _quads(here: Tensor, crossed: Tensor, keys: HashMap, grid: Tensor, ring: Tensor) -> Tensor:
    """``[Q, 4]`` voxel rows per crossing that owns a whole quad."""
    rows, axes = crossed.nonzero(as_tuple=True)
    if rows.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=here.device)
    cells = here[rows, 1:][:, None, :] + ring[axes]
    # A cell one step past the edge carries the flat key of the next line, or of the next
    # batch: outside the grid is no neighbour however the lookup answers it.
    inside = ((cells >= 0) & (cells < grid)).all(dim=-1)
    found = keys.lookup(flat_keys(torch.cat((torch.zeros_like(cells[..., :1]), cells), dim=-1)
                                  .reshape(-1, 4), (1, *grid.tolist()))).reshape(-1, 4)
    return found[(inside & (found >= 0)).all(dim=-1)].long()


def _cut(quad: Tensor, positions: Tensor, weight: Optional[Tensor]) -> Tensor:
    """Each quad's two triangles, along the diagonal its weight prefers."""
    splits = SPLITS.to(quad.device)
    if weight is None:
        corners = positions[quad]
        aligned = []
        for split in (splits[0], splits[1]):
            faced = corners[:, split].reshape(-1, 3, 3)
            span = torch.linalg.cross(faced[:, 1] - faced[:, 0], faced[:, 2] - faced[:, 0])
            pair = span.reshape(-1, 2, 3)
            aligned.append((pair[:, 0] * pair[:, 1]).sum(dim=-1).abs())
        # The honest diagonal is the one whose two triangles face each other least.
        take = aligned[1] > aligned[0]
    else:
        w = weight[quad]
        take = w[:, 1] * w[:, 3] >= w[:, 0] * w[:, 2]
    return torch.where(take[:, None], quad[:, splits[1]],
                       quad[:, splits[0]]).reshape(-1, 3)


def dual_grid_mesh(lattice: SparseTensor, vertex: Tensor, crossed: Tensor,
                   split: Optional[Tensor] = None, *, bounds: float = 0.5) -> List[Mesh]:
    """The surface a lattice describes, one mesh per batch entry.

    A voxel crossed along an axis owns the quad around the edge at its own corner, joining the
    dual vertices of the four voxels there. A position is the cell index plus the in-cell
    offset read out for it, scaled into ``[-bounds, bounds]``.

    A mesh's vertex *is* a voxel, numbered from zero within its own batch entry, so what a
    voxel carries — its material — gathers by the same index. That keeps vertices no face
    uses; pruning them belongs to whoever holds those features too.
    """
    coords, shape = lattice.coords, lattice.shape
    if vertex.shape != (len(lattice), 3):
        raise ValueError(f"a dual vertex per voxel is [M, 3], got {tuple(vertex.shape)}")
    if crossed.shape != (len(lattice), 3):
        raise ValueError(f"three crossings per voxel are [M, 3], got {tuple(crossed.shape)}")
    if split is not None:
        split = split.reshape(-1)
        if split.shape[0] != len(lattice):
            raise ValueError(f"{split.shape[0]} split weights on {len(lattice)} voxels")

    device = coords.device
    ring, grid = corner_ring(device), torch.tensor(shape[1:], dtype=torch.int32,
                                                   device=device)
    step = torch.tensor([2.0 * bounds / float(v) for v in shape[1:]], dtype=torch.float32,
                        device=device)
    meshes: List[Mesh] = []
    for entry in coords[:, 0].unique():
        rows = torch.where(coords[:, 0] == entry)[0]
        here = coords[rows].clone()
        here[:, 0] = 0
        keys = HashMap(flat_keys(here, (1, *shape[1:])),
                       torch.arange(len(rows), dtype=torch.int32, device=device))
        quad = _quads(here, crossed[rows], keys, grid, ring)
        if quad.shape[0] == 0:
            meshes.append(empty_mesh(device))
            continue
        positions = (-bounds) + (here[:, 1:].float() + vertex[rows].float()) * step
        meshes.append(Mesh(vertices=positions,
                           faces=_cut(quad, positions,
                                      None if split is None else split[rows])))
    return meshes


__all__ = ["dual_grid_mesh"]
