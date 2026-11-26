from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .octree import CORNERS, SdfVolume

NORMAL_CREASE_DEGREES = 40.0
#: Thinnest face a writer draws; 1 is an equilateral and 0 a line, so this is a third of a degree.
SLIVER_QUALITY = 0.01

_CREASE_LIMIT = math.cos(math.radians(NORMAL_CREASE_DEGREES))

#: The Kuhn split as CORNERS indices (4i + 2j + k); neighbouring cubes share every face diagonal.
TETRAHEDRA = torch.tensor([[0, 4, 6, 7], [0, 4, 5, 7], [0, 2, 6, 7],
                           [0, 2, 3, 7], [0, 1, 5, 7], [0, 1, 3, 7]], dtype=torch.long)

#: Triangles per vertex pattern, as tetrahedron edges wound outward; complements reversed.
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
    """+1 per tetrahedron keeping the cube's orientation, -1 per reversal."""
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

    def corners(self) -> Tensor:
        """Every face as its three positions."""
        return self.vertices[self.faces]

    def face_spans(self) -> Tensor:
        """The cross product of every face: its direction, with twice its area as length."""
        corners = self.corners()
        return torch.linalg.cross(corners[:, 1] - corners[:, 0],
                                  corners[:, 2] - corners[:, 0])

    def face_normals(self) -> Tensor:
        """The unit normal of every face; a faceless one points nowhere."""
        span = self.face_spans()
        return span / span.norm(dim=-1, keepdim=True).clamp(min=1e-20)

    def face_areas(self) -> Tensor:
        """Area of every triangle, in the mesh's own units."""
        return self.face_spans().norm(dim=-1) * 0.5

    def face_quality(self) -> Tensor:
        """One for an equilateral, zero for a line: a ratio of lengths, so size is never asked."""
        corners = self.corners()
        there = corners[:, 1] - corners[:, 0]
        on = corners[:, 2] - corners[:, 1]
        squares = (there.square().sum(dim=-1) + on.square().sum(dim=-1)
                   + (there + on).square().sum(dim=-1))
        twice_area = torch.linalg.cross(there, on).norm(dim=-1)
        return 2 * math.sqrt(3.0) * twice_area / squares.clamp(min=1e-30)

    def select(self, keep_vertices: Tensor, keep_faces: Tensor) -> "Mesh":
        """The kept faces, renumbered into the kept positions."""
        lookup = torch.full((self.vertices.shape[0],), -1, dtype=self.faces.dtype,
                            device=self.faces.device)
        lookup[keep_vertices] = torch.arange(int(keep_vertices.sum()),
                                             device=self.faces.device)
        return Mesh(self.vertices[keep_vertices], lookup[self.faces[keep_faces]])

    def signed_volume(self) -> Tensor:
        """The volume the faces enclose, positive when they face outward."""
        triangles = self.corners()
        return (triangles[:, 0]
                * torch.linalg.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0

    def without_degenerate(self) -> "Mesh":
        """The mesh without faces no reader can draw: the faceless and the needles, by shape."""
        keep_faces = self.face_quality() > SLIVER_QUALITY
        used = torch.zeros(self.vertices.shape[0], dtype=torch.bool,
                           device=self.faces.device)
        used[self.faces[keep_faces]] = True
        if bool(keep_faces.all()):
            return self
        return self.select(used, keep_faces)

    def smooth_normals(self) -> Tensor:
        """The way every point faces: neighbours within NORMAL_CREASE_DEGREES of the widest."""
        span = self.face_spans()
        worth = span.norm(dim=-1)
        unit = span / worth.clamp(min=1e-20).unsqueeze(-1)
        corner = self.faces.reshape(-1)
        # Ranks rise with area, so an amax over ranks leaves each point its widest face.
        lightest_first = torch.argsort(worth.repeat_interleave(3))
        widest = torch.full((self.vertices.shape[0],), -1, dtype=torch.long,
                            device=self.faces.device)
        widest.scatter_reduce_(0, corner[lightest_first],
                               torch.arange(lightest_first.shape[0], device=corner.device),
                               reduce="amax", include_self=False)
        seed = unit[lightest_first[widest.clamp(min=0)] // 3]
        inside = (unit.unsqueeze(1) * seed[self.faces]).sum(dim=-1) >= _CREASE_LIMIT
        weight = (worth.unsqueeze(1) * inside.to(worth.dtype)).unsqueeze(-1)
        faced = torch.zeros_like(self.vertices)
        faced.index_add_(0, corner, (unit.unsqueeze(1) * weight).reshape(-1, 3))
        faced = torch.where(faced.square().sum(dim=-1, keepdim=True) > 0.0, faced, seed)
        return faced / faced.norm(dim=-1, keepdim=True).clamp(min=1e-20)


def empty_mesh(device=None) -> Mesh:
    return Mesh(torch.zeros((0, 3), device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device))


def _unravel(ids: Tensor, stride: int) -> Tensor:
    plane = stride * stride
    return torch.stack((ids // plane, (ids // stride) % stride, ids % stride), dim=-1)


def _edge_points(volume: SdfVolume, keys: Tensor, plane: int) -> Tensor:
    """One point per packed corner pair, low grid id first, so both cells sharing an edge agree."""
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
