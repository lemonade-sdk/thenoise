"""Paint measured control fields: depth is normalised per image and mapped to grey,
a skeleton is drawn at its painted size rather than stretched.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple
import torch
from PIL import Image, ImageDraw

from .base import scale_field, uint8_to_pil

#: Limb colours in edge order — the ViTPose demo palette the ControlNets were trained on.
COCO_POSE_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 100), (255, 0, 0), (255, 50, 0), (255, 200, 0), (255, 255, 0),
    (150, 255, 0), (100, 255, 0), (0, 255, 0), (255, 150, 0), (0, 255, 255),
    (0, 150, 255), (0, 100, 255), (0, 50, 255), (255, 0, 255), (255, 0, 150),
    (200, 0, 255), (255, 0, 200), (0, 255, 100), (0, 200, 255), (255, 255, 0),
    (200, 255, 0), (100, 0, 255),
)

#: Canvas colours: SD's openpose used green, the newer ControlNets black.
CANVASES = {"black": (0, 0, 0), "green": (36, 255, 12)}

#: COCO-18 derives the neck at the shoulder midpoint; these are its label names.
_NECK_PARENTS = ("L_Shoulder", "R_Shoulder")
_HEAD = "Nose"


def normalize(field: torch.Tensor) -> torch.Tensor:
    """A ``[1, H, W]`` fp32 field -> ``[0, 1]``; a flat field stays flat."""
    lo, hi = field.min(), field.max()
    if hi > lo:
        return (field - lo) / (hi - lo)
    return field.clamp(0.0, 1.0)


def depth_to_pil(
    field: torch.Tensor, invert: bool = False, colormap: str = "gray"
) -> Image.Image:
    """A ``[1, H, W]`` fp32 relative-depth field -> an 8-bit image at the same size."""
    x = normalize(field)
    if invert:
        x = 1.0 - x
    if colormap == "gray":
        return uint8_to_pil(scale_field(x))
    raise ValueError(f"colormap {colormap!r} is not available yet")


def skeleton_edges(
    edges: Sequence[Sequence[int]], labels: dict
) -> Tuple[List[Tuple[int, int]], Tuple[int, int] | None]:
    """Widen a checkpoint's edges to COCO-18: the shoulders meet at a derived neck and
    the head runs through it; a skeleton with no shoulders is drawn as declared."""
    drawn = [(int(a), int(b)) for a, b in edges]
    by_name = {name: int(index) for index, name in labels.items()}
    if not all(name in by_name for name in (*_NECK_PARENTS, _HEAD)):
        return drawn, None
    left, right = (by_name[name] for name in _NECK_PARENTS)
    neck = max(by_name.values()) + 1
    with_neck: List[Tuple[int, int]] = []
    for a, b in drawn:
        if {a, b} == {left, right}:
            with_neck += [(a, neck), (neck, b)]  # the torso runs through the neck
        else:
            with_neck.append((a, b))
    with_neck.append((neck, by_name[_HEAD]))
    return with_neck, (left, right)


def skeleton_to_pil(
    size: Tuple[int, int],
    people: Iterable[Any],
    edges: Sequence[Sequence[int]],
    labels: dict,
    line_width: int = 4,
    radius: int = 4,
    background: str = "black",
) -> Image.Image:
    """Draw the people as a skeleton on a ``size`` canvas. Normalised keypoints are
    scaled here, and a limb is drawn only when both its joints survived."""
    canvas = Image.new("RGB", size, CANVASES[background])
    skeleton, neck_parents = skeleton_edges(edges, labels)
    pen = ImageDraw.Draw(canvas)
    for person in people:
        keypoints, scores = _with_neck(person.keypoints, person.scores, neck_parents)
        confidence = scores.tolist()
        # Normalised keypoints -> canvas pixels: the last pixel's centre is at size - 1.
        points = [
            (float(x) * (size[0] - 1), float(y) * (size[1] - 1))
            for x, y in keypoints.tolist()
        ]
        for limb, (start, end) in enumerate(skeleton):
            if confidence[start] <= 0 or confidence[end] <= 0:
                continue
            pen.line(
                [points[start], points[end]],
                fill=COCO_POSE_PALETTE[limb % len(COCO_POSE_PALETTE)],
                width=line_width,
            )
        for joint, point in enumerate(points):
            if confidence[joint] <= 0:
                continue
            x, y = point
            pen.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=COCO_POSE_PALETTE[joint % len(COCO_POSE_PALETTE)],
            )
    return canvas


def _with_neck(
    keypoints: torch.Tensor, scores: torch.Tensor, neck_parents: Tuple[int, int] | None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append the COCO-18 neck at the shoulder midpoint. Scored with the weaker
    shoulder, so a culled shoulder drops the neck and the limbs through it."""
    if neck_parents is None:
        return keypoints, scores
    left, right = neck_parents
    neck = ((keypoints[left] + keypoints[right]) * 0.5).unsqueeze(0)
    weaker_shoulder = scores[[left, right]].min()
    return (
        torch.cat([keypoints, neck]),
        torch.cat([scores, weaker_shoulder.reshape(1)]),
    )


__all__ = [
    "CANVASES",
    "COCO_POSE_PALETTE",
    "depth_to_pil",
    "normalize",
    "skeleton_edges",
    "skeleton_to_pil",
]
