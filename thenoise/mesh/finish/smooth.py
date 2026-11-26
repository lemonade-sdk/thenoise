from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
from typing import Callable

import torch
from torch import Tensor

from ..extract import Mesh
from ..memory import room

logger = logging.getLogger(__name__)

CREASE_DEGREES = 50.0
STEP = 0.5
MAX_ROUNDS = 16
SMOOTH_BYTES_PER_TRIANGLE = 420


@dataclass(frozen=True)
class _Neighbourhood:
    """Both ends of every edge worth averaging across, and the weights every round reads."""

    src: Tensor
    dst: Tensor
    mix: Tensor
    still: Tensor
    degree: Tensor
    moving: Tensor
    crossed: int
    creases: int
    held: int


def relax_surface(mesh: Mesh, strength: float, check: Callable[[], None]) -> Mesh:
    """``mesh`` eased toward its crease-free neighbourhood, ``strength`` many rounds."""
    if strength <= 0.0 or mesh.is_empty:
        return mesh

    triangles = mesh.faces.shape[0]
    room(mesh.vertices.device, triangles * SMOOTH_BYTES_PER_TRIANGLE,
         f"smoothing {triangles} triangles", "lower the level of detail")
    rounds = min(MAX_ROUNDS, max(1, ceil(strength * MAX_ROUNDS)))
    near = _neighbours(mesh, CREASE_DEGREES)
    before = float(mesh.signed_volume())
    rough = _wobble(mesh.vertices, near)

    points = mesh.vertices.clone()
    total = torch.zeros_like(points)
    for _round in range(rounds):
        check()
        total.zero_()
        total.index_add_(0, near.dst, points[near.src])
        points = points * near.still + total * near.mix

    after = Mesh(points, mesh.faces)
    smoother = _wobble(points, near)
    logger.info("smooth: %d of %d edges averaged, %d points held at a crease, %d rounds, "
                "wobble %.2g -> %.2g (%s), %s",
                near.crossed, near.crossed + near.creases, near.held, rounds, rough, smoother,
                _cut(rough, smoother), _change(before, float(after.signed_volume())))
    return after


def _wobble(points: Tensor, near: _Neighbourhood) -> float:
    """How far a point sits from the average of its neighbours, over the points a round may
    move: held points disagree with their neighbours on purpose and would hide the measure."""
    total = torch.zeros_like(points).index_add_(0, near.dst, points[near.src])
    away = (points - total / near.degree.clamp(min=1.0)).norm(dim=-1)
    return float(away[near.moving].mean()) if bool(near.moving.any()) else 0.0


def _cut(before: float, after: float) -> str:
    return "flat" if not before else f"{100 * (after / before - 1):+.0f}%"


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
    two = place < 2
    at_edge = torch.full((edges.shape[0], 2), -1, dtype=torch.long, device=at.device)
    at_edge[grouped[two], place[two]] = owner[order][two]

    triad = points[faces]
    normal = torch.linalg.cross(triad[:, 1] - triad[:, 0], triad[:, 2] - triad[:, 0])
    unit = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-12)
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
                          still=(1.0 - free * STEP).unsqueeze(1), degree=degree.unsqueeze(1),
                          moving=free > 0.0,
                          crossed=int(worth.sum()), creases=int((~worth).sum()),
                          held=int(((sharp > 0.0) & (degree > 0.0)).sum()))


def _change(before: float, after: float) -> str:
    """How much of the shape the smoothing ate, which is what decides whether it needs more."""
    return "volume unchanged" if not before else f"volume {100 * (after / before - 1):+.1f}%"


__all__ = ["MAX_ROUNDS", "relax_surface"]
