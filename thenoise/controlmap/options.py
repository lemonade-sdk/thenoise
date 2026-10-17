"""Every control-map parameter, declared once — flags, request schema and UI form.

The knob plumbing (``option``, its validation, its schema and its flags) is shared
with the post-decode chain and lives in :mod:`thenoise.utils.options`; what is
declared here is what a control-map request can ask of a kind.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import ClassVar, Dict

from thenoise.utils.options import (
    add_flags,
    option,
    options_schema,
    validate,
)

#: Names a list of the ``/controlmaps`` payload a field takes its choices from.
DETECTOR_POOL = "detectors"

#: The pose canvases; :data:`thenoise.controlmap.render.CANVASES` holds their colours.
CANVAS_NAMES = ("black", "green")

#: The palettes a depth map can be painted with; ``turbo`` is the vendored lookup table.
COLORMAPS = ("gray", "turbo")


def add_option_args(parser: argparse.ArgumentParser, kind: str, cls: type) -> None:
    """One ``--<kind>-<option>`` flag per option; a flag nobody passes stays unset."""
    add_flags(parser, cls, prefix=kind, dest_prefix=f"{kind}_", suppress_unset=True)


@dataclass
class ControlMapOptions:
    """What a request asks of a kind; fields are validated here, not in each kind."""

    def __post_init__(self) -> None:
        validate(self)


@dataclass
class CannyOptions(ControlMapOptions):
    """Edge thresholds (Sobel magnitude, 8-bit units) and rendered line thickness."""

    kind: ClassVar[str] = "canny"

    low: float = option(
        100.0, label="Low edge strength", ge=0, step=1,
        help="low threshold, in Sobel magnitude units")
    high: float = option(
        200.0, label="High edge strength", ge=0, step=1,
        help="high threshold; weak edges connected to a strong one are kept")
    thickness: int = option(
        1, label="Line width", ge=1, step=1, help="rendered edge thickness in pixels")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.low > self.high:
            raise ValueError(
                f"low threshold {self.low} exceeds high threshold {self.high}"
            )


@dataclass
class LineartOptions(ControlMapOptions):
    """Line intensity threshold (0-1) and rendered line thickness."""

    kind: ClassVar[str] = "lineart"

    threshold: float = option(
        0.5, label="Line strength", ge=0, le=1, step=0.05,
        help="line intensity threshold, on the per-image normalised edge strength")
    thickness: int = option(
        1, label="Line width", ge=1, step=1, help="line thickness in pixels")


@dataclass
class SoftedgeOptions(ControlMapOptions):
    """The soft edge cutoff (0-1) and the smoothing blur (sigma, 0 = none)."""

    kind: ClassVar[str] = "softedge"

    threshold: float = option(
        0.2, label="Edge strength", ge=0, le=1, step=0.05,
        help="soft edge cutoff: where the gradient starts to count")
    blur: float = option(
        1.5, label="Blur", ge=0, step=0.5,
        help="smoothing sigma in pixels, 0 leaves an edge crisp")


@dataclass
class DepthOptions(ControlMapOptions):
    """How the relative-depth field is painted."""

    kind: ClassVar[str] = "depth"

    invert: bool = option(False, label="Invert (near = dark)", help="swap near/far")
    colormap: str = option(
        "gray", label="Colormap", choices=COLORMAPS,
        help="paint the depth field as grayscale or as the turbo rainbow")


@dataclass
class PoseOptions(ControlMapOptions):
    """Which person detector to crop with, and how the skeleton is painted."""

    kind: ClassVar[str] = "pose"

    detector: str = option(
        "", label="Person detector", pool=DETECTOR_POOL,
        help="person detector checkpoint to crop with (default: the first in the pool)")
    threshold: float = option(
        0.5, label="Person confidence", ge=0, le=1, step=0.05,
        help="minimum detector score for a person to be in the map at all")
    kp_threshold: float = option(
        0.3, label="Joint confidence", ge=0, le=1, step=0.05,
        help="minimum confidence for a joint to be drawn")
    line_width: int = option(
        4, label="Bone thickness", ge=1, step=1, help="skeleton line width in pixels")
    radius: int = option(4, label="Joint radius", ge=1, step=1, help="keypoint dot radius")
    background: str = option(
        "black", label="Background", choices=CANVAS_NAMES,
        help="canvas colour; green is the original openpose one")


#: Control-map kind -> the option dataclass its request carries.
OPTIONS_BY_KIND: Dict[str, type] = {
    cls.kind: cls
    for cls in (CannyOptions, DepthOptions, PoseOptions, LineartOptions, SoftedgeOptions)
}

__all__ = [
    "CANVAS_NAMES",
    "COLORMAPS",
    "CannyOptions",
    "DETECTOR_POOL",
    "DepthOptions",
    "LineartOptions",
    "OPTIONS_BY_KIND",
    "PoseOptions",
    "ControlMapOptions",
    "SoftedgeOptions",
    "add_option_args",
    "option",
    "options_schema",
    "validate",
]
