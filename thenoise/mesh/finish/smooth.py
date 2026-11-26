from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable

import torch
from torch import Tensor

from ..extract import Mesh

CREASE_DEGREES = 50.0
STEP = 0.5
MAX_ROUNDS = 16


@dataclass(frozen=True)
class _Neighbourhood:
    """Both ends of every edge worth averaging across, and the weights every round reads."""

    src: Tensor
    dst: Tensor
    mix: Tensor
    still: Tensor


def relax_surface(mesh: Mesh, strength: float, check: Callable[[], None]) -> Mesh:
    """``mesh`` eased toward its crease-free neighbourhood, ``strength`` many rounds."""
    if strength <= 0.0 or mesh.is_empty:
        return mesh

    rounds = min(MAX_ROUNDS, max(1, ceil(strength * MAX_ROUNDS)))
    near = _neighbours(mesh, CREASE_DEGREES)

    points = mesh.vertices.clone()
    total = torch.zeros_like(points)
    for _round in range(rounds):
        check()
        total.zero_()
        total.index_add_(0, near.dst, points[near.src])
        points = points * near.still + total * near.mix

    return Mesh(points, mesh.faces)


def _neighbours(mesh: Mesh, crease: float) -> _Neighbourhood:
    """The ends of every edge whose two faces meet within ``crease`` degrees."""
    faces, points = mesh.faces, mesh.vertices
    pair = torch.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    pair = pair.permute(1, 0, 2).reshape(-1, 2)
    owner = torch.arange(faces.shape[0], device=faces.device).repeat_interleave(3)
    span = points.shape[0]
    edges, at = torch.unique(pair.min(dim=1).values * span + pair.max(dim=1).values,
                             return_inverse=True)

    order = torch.argsort(at, stable=True)
    grouped = at[order]
    counts = torch.bincount(grouped, minlength=edges.shape[0])
    place = (torch.arange(grouped.shape[0], device=at.device)
             - (counts.cumsum(0) - counts)[grouped])
    # An edge shared by more than two faces keeps the first two, which a cut does not produce.
    two = place < 2
    at_edge = torch.full((edges.shape[0], 2), -1, dtype=torch.long, device=at.device)
    at_edge[grouped[two], place[two]] = owner[order][two]

    unit = mesh.face_normals()
    both_sides = at_edge[:, 1] >= 0
    turn = torch.rad2deg(torch.acos((unit[at_edge[:, 0]] * unit[at_edge[:, 1]])
                                    .sum(dim=-1).clamp(-1.0, 1.0)))
    worth = both_sides & (turn <= crease)

    low, high = edges // span, edges % span
    src = torch.cat((low[worth], high[worth]))
    dst = torch.cat((high[worth], low[worth]))
    ones = torch.ones(dst.shape[0], dtype=points.dtype, device=points.device)
    degree = torch.zeros(span, dtype=points.dtype, device=points.device)
    degree.index_add_(0, dst, ones)

    ends = torch.cat((low, high))
    sharp = torch.zeros(span, dtype=points.dtype, device=points.device)
    sharp.index_add_(0, ends, torch.where(torch.cat((worth, worth)), 0.0, 1.0)
                     .to(points.dtype))
    free = torch.where(degree > 0.0, 1.0, 0.0) * torch.where(sharp > 0.0, 0.0, 1.0)
    return _Neighbourhood(src=src, dst=dst,
                          mix=(free * STEP / degree.clamp(min=1.0)).unsqueeze(1),
                          still=(1.0 - free * STEP).unsqueeze(1))


__all__ = ["MAX_ROUNDS", "relax_surface"]
