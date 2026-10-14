"""Every control-map parameter, declared once — flags, request schema and UI form."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Optional, Sequence, get_type_hints

#: Names a list of the ``/controlmaps`` payload a field takes its choices from.
DETECTOR_POOL = "detectors"

#: The pose canvases; :data:`thenoise.controlmap.render.CANVASES` holds their colours.
CANVAS_NAMES = ("black", "green")

#: The palettes a depth map can be painted with; ``turbo`` is the vendored lookup table.
COLORMAPS = ("gray", "turbo")

#: The JSON type the web UI builds each field's control from.
_JSON_TYPE = {bool: "boolean", int: "integer", float: "number", str: "string"}

#: ``dataclasses`` metadata key -> the word the UI and the schema use for it.
_WIRE_KEYS = (("ge", "min"), ("le", "max"), ("step", "step"), ("choices", "choices"),
              ("pool", "pool"))


def option(
    default,
    *,
    label: str,
    help: str = "",
    ge: Optional[float] = None,
    le: Optional[float] = None,
    step: Optional[float] = None,
    choices: Optional[Sequence[str]] = None,
    pool: Optional[str] = None,
):
    """One tunable: its default, the words shown for it, and what it accepts."""
    meta: Dict[str, Any] = {"label": label}
    if help:
        meta["help"] = help
    for key, value in (("ge", ge), ("le", le), ("step", step), ("choices", choices),
                       ("pool", pool)):
        if value is not None:
            meta[key] = value
    return field(default=default, metadata=meta)


def validate(options: "ControlMapOptions") -> None:
    """Reject a value outside ``ge``/``le``, or a choice that is not offered."""
    kind = getattr(options, "kind", "")
    for entry in fields(options):
        meta, value = entry.metadata, getattr(options, entry.name)
        low, high = meta.get("ge"), meta.get("le")
        if (low is not None and value < low) or (high is not None and value > high):
            bound = (
                f">= {low}" if high is None
                else f"<= {high}" if low is None
                else f"in [{low}, {high}]"
            )
            raise ValueError(f"{entry.name} must be {bound}, got {value}")
        choices = meta.get("choices")
        if choices is not None and value not in choices:
            named = f"{kind} " if kind else ""
            raise ValueError(
                f"unknown {named}{entry.name} {value!r}; expected one of {tuple(choices)}"
            )


def options_schema(cls: type) -> List[Dict[str, Any]]:
    """One kind's options as a form description: label, control, default, bounds."""
    hints = get_type_hints(cls)
    described = []
    for entry in fields(cls):
        meta = entry.metadata
        item: Dict[str, Any] = {
            "name": entry.name,
            "label": meta.get("label", entry.name),
            "type": _JSON_TYPE[hints[entry.name]],
            "default": entry.default,
        }
        for key, wire in _WIRE_KEYS:
            if key in meta:
                item[wire] = tuple(meta[key]) if key == "choices" else meta[key]
        described.append(item)
    return described


def add_option_args(parser: argparse.ArgumentParser, kind: str, cls: type) -> None:
    """One ``--<kind>-<field>`` flag per option; a flag nobody passes stays unset."""
    hints = get_type_hints(cls)
    for entry in fields(cls):
        meta = entry.metadata
        flag = f"--{kind}-{entry.name.replace('_', '-')}"
        help = f"{meta.get('help', '')} (default: {entry.default})".strip()
        if hints[entry.name] is bool:
            parser.add_argument(
                flag, dest=f"{kind}_{entry.name}", action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS, help=help,
            )
            continue
        kwargs: Dict[str, Any] = {
            "dest": f"{kind}_{entry.name}", "type": hints[entry.name],
            "default": argparse.SUPPRESS, "help": help,
        }
        if meta.get("choices"):
            kwargs["choices"] = list(meta["choices"])
        parser.add_argument(flag, **kwargs)


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
