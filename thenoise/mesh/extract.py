from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .octree import CORNERS, SdfVolume

#: The Kuhn split as CORNERS indices (4i + 2j + k); neighbouring cubes share every face diagonal.
TETRAHEDRA = torch.tensor([[0, 4, 6, 7], [0, 4, 5, 7], [0, 2, 6, 7],
                           [0, 2, 3, 7], [0, 1, 5, 7], [0, 1, 3, 7]], dtype=torch.long)

#: Triangles per tetrahedron vertex pattern, as tetrahedron edges, wound outward; complements reversed.
_PATTERNS = {
    0b0001: (((0, 1), (0, 2), (0, 3)),),
    0b0010: (((0, 1), (1, 3), (1, 2)),),
    0b0100: (((1, 2), (2, 3), (0, 2)),),
    0b1000: (((0, 3), (2, 3), (1, 3)),),
    0b0011: (((0, 2), (1, 3), (1, 2)), ((0, 2), (0, 3), (1, 3))),
    0b0101: (((0, 1), (1, 2), (2, 3)), ((0, 1), (2, 3), (0, 3))),
    0b1001: (((0, 1), (0, 2), (2, 3)), ((0, 1), (2, 3), (1, 3))),
}


def _triangle_table() -> Tensor:
    """The patterns above, padded to two triangles of three edges each, missing ones -1."""
    table = torch.full((16, 2, 3, 2), -1, dtype=torch.long)
    for code, patterns in _PATTERNS.items():
        for slot, pattern in enumerate(patterns):
            edges = torch.tensor(pattern, dtype=torch.long)
            table[code, slot] = edges
            table[15 - code, slot] = edges.flip(0)
    return table


def _corner_coordinates() -> Tensor:
    """The eight cube corners as positions, matching ``CORNERS`` order."""
    return torch.tensor([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)],
                        dtype=torch.float32)


def _orientation() -> Tensor:
    """+1 per tetrahedron keeping the cube's orientation, -1 per reversal, which the Kuhn split alternates."""
    corners = _corner_coordinates()[TETRAHEDRA]
    steps = torch.stack((corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0],
                         corners[:, 3] - corners[:, 0]), dim=1)
    return torch.sign(torch.linalg.det(steps)).to(torch.long)


ORIENTATION = _orientation()

_TET_TRIANGLES = _triangle_table()

#: The pattern table twice: index 0 as derived, index 1 for the flipped tetrahedra.
_TET_TABLES = torch.stack((_TET_TRIANGLES, _TET_TRIANGLES.flip(-2)))

_FLIP = (ORIENTATION < 0).to(torch.long)


@dataclass(frozen=True)
class Mesh:
    """A triangle soup: ``[V, 3]`` positions and ``[F, 3]`` indices into them."""

    vertices: Tensor
    faces: Tensor

    @property
    def is_empty(self) -> bool:
        return self.faces.shape[0] == 0

    def cpu(self) -> "Mesh":
        """The mesh detached and on the host, which is where every writer starts."""
        return Mesh(self.vertices.detach().cpu(), self.faces.detach().cpu())

    def scaled(self, factor: float) -> "Mesh":
        """Positions about the origin; unscaled is the same mesh, not a copy."""
        if factor == 1.0:
            return self
        return Mesh(self.vertices * factor, self.faces)

    def signed_volume(self) -> Tensor:
        """The volume the faces enclose, positive when they face outward."""
        triangles = self.vertices[self.faces]
        return (triangles[:, 0]
                * torch.linalg.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0


def empty_mesh(device=None) -> Mesh:
    return Mesh(torch.zeros((0, 3), device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device))


def _unravel(ids: Tensor, stride: int) -> Tensor:
    plane = stride * stride
    return torch.stack((ids // plane, (ids // stride) % stride, ids % stride), dim=-1)


def _edge_points(volume: SdfVolume, keys: Tensor, plane: int) -> Tensor:
    """One point per packed corner pair, interpolated, always low grid id first so both cells sharing an edge agree bit for bit."""
    stride = volume.resolution + 1
    flat = volume.values.reshape(-1)
    lo, hi = keys // plane, keys % plane
    low, high = flat[lo], flat[hi]
    step = high - low
    crossing = step.abs() >= 1e-12
    # A flat edge has no crossing to locate, so give it the midpoint rather than a NaN.
    ratio = torch.where(crossing, (volume.level - low) / torch.where(crossing, step,
                                                                     torch.ones_like(step)),
                        torch.full_like(step, 0.5))
    ends = volume.corner_coordinates(
        torch.stack((_unravel(lo, stride), _unravel(hi, stride)), dim=1))
    return ends[:, 0, :] + ratio.unsqueeze(-1) * (ends[:, 1, :] - ends[:, 0, :])


@torch.no_grad()
def extract_surface(volume: SdfVolume) -> Mesh:
    """Tetrahedral marching cubes over the cells ``volume`` kept."""
    cells = volume.extracted_cells()
    if cells.shape[0] == 0:
        return empty_mesh(cells.device)

    stride = volume.resolution + 1
    plane = stride ** 3
    index = cells[:, None, :] + CORNERS.to(cells.device)
    corner_id = (index[..., 0] * stride + index[..., 1]) * stride + index[..., 2]
    inside = volume.values.reshape(-1)[corner_id] < volume.level
    tetrahedra = TETRAHEDRA.to(cells.device)
    tables = _TET_TABLES.to(cells.device)
    bits = (1 << torch.arange(4, device=cells.device))
    codes = (inside[:, tetrahedra].to(torch.long) * bits).sum(-1)

    cut: list[Tensor] = []
    for tet in range(tetrahedra.shape[0]):
        code = codes[:, tet]
        keep = (code != 0) & (code != 15)
        if not bool(keep.any()):
            continue
        plan = tables[_FLIP[tet].item()][code[keep]]
        ids = corner_id[keep][:, tetrahedra[tet]]
        pairs = torch.gather(ids, 1, plan.clamp(min=0).reshape(ids.shape[0], -1))
        triangles = (plan >= 0).all(-1).all(-1)
        cut.append(pairs.view(-1, 3, 2)[triangles.reshape(-1)])

    pairs = torch.cat(cut).reshape(-1, 2)
    unique, inverse = torch.unique(pairs.min(-1).values * plane + pairs.max(-1).values,
                                   return_inverse=True)
    faces = inverse.view(-1, 3)
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) \
        & (faces[:, 0] != faces[:, 2])
    faces = faces[distinct]

    used = torch.unique(faces)
    lookup = torch.full((unique.shape[0],), -1, dtype=torch.long, device=faces.device)
    lookup[used] = torch.arange(used.shape[0], device=faces.device)
    points = _edge_points(volume, unique, plane)
    return Mesh(vertices=points[used], faces=lookup[faces])


__all__ = ["ORIENTATION", "TETRAHEDRA", "Mesh", "empty_mesh", "extract_surface"]
