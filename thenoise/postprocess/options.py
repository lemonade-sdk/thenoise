"""Every post-decode knob, declared once — flag, wire field, PNG record, UI control
and place in the chain.

**Field order is chain order.** The filters run top to bottom over the pixels, so
where a knob sits in this dataclass decides what it acts on. ``filter`` groups knobs
that one filter consumes together (they are read as a set, so a filter that wants
several knobs applies them in one pass rather than four); ``stage`` says when it
runs, because one filter has to see the raw decode.

Every knob's default is *off*, and that is what makes the chain free when it is not
asked for: a filter whose knobs all still hold their default is skipped, so a request
that wants nothing gets the very tensor it was handed back.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple

from thenoise.utils.options import option, option_names, validate

#: Right after the VAE decode, at the native decoded resolution.
DECODE = "decode"
#: On the pixels that get delivered, after any pixel upscaling and resize.
FINAL = "final"


@dataclass
class PostprocessOptions:
    """What a request asks of the post-decode chain."""

    qwen_vae_enhance: bool = option(
        False, label="Reduce grid pattern", hint="enable when needed",
        help="apply the Nyquist Notch post filter to decoded pixels "
             "(removes 2px grid artifacts)",
        filter="nyquist_notch", stage=DECODE)

    # Colour before detail: sharpening should see the colours the image ships with,
    # and out-of-gamut chroma is clamped here rather than sharpened into clipping.
    saturation: float = option(
        0.0, label="Saturation", ge=-1.0, le=1.0, step=0.05,
        filter="color", stage=FINAL,
        help="colour saturation around the neutral 0.0: -1.0 greyscales, 1.0 doubles")
    vibrance: float = option(
        0.0, label="Vibrance", ge=-1.0, le=1.0, step=0.05,
        filter="color", stage=FINAL,
        help="saturation that lifts washed-out pixels and leaves already vivid ones alone")
    temperature: float = option(
        0.0, label="Temperature", ge=-1.0, le=1.0, step=0.05,
        filter="color", stage=FINAL,
        help="white balance around the neutral 0.0: -1.0 cools to blue, 1.0 warms to amber")
    tint: float = option(
        0.0, label="Tint", ge=-1.0, le=1.0, step=0.05,
        filter="color", stage=FINAL,
        help="white balance across the green-magenta axis: -1.0 greens, 1.0 magentas")

    # Sharpen before graining: the grain is texture on top of the final detail, and
    # a sharpen pass over noise would only amplify it.
    sharpening: float = option(
        0.0, label="Sharpening", ge=0.0, le=1.0, step=0.05,
        filter="rcas", stage=FINAL,
        help="RCAS sharpening strength 0.0–1.0 (contrast-adaptive)")

    film_grain: float = option(
        0.0, label="Film grain", ge=0.0, le=10.0, step=0.1, ui_max=1.0,
        filter="film_grain", stage=FINAL,
        help="film-grain strength 0.0–10.0. Reasonable values are <= 1.0")

    def __post_init__(self) -> None:
        validate(self)


#: Every knob name, in chain order.
POSTPROCESS_NAMES = option_names(PostprocessOptions)

#: The default of every knob, which is by definition its "off" value.
DEFAULTS: Dict[str, Any] = {
    name: getattr(PostprocessOptions, name) for name in POSTPROCESS_NAMES
}


@dataclass(frozen=True)
class FilterSpec:
    """One filter in the chain: the knobs it reads, and the stage it runs in."""

    name: str
    options: Tuple[str, ...]
    stage: str


def _declared_chain() -> Tuple[FilterSpec, ...]:
    """The filters in chain order, taking each one's knobs from the declaration."""
    chain, declared = [], set()
    for entry in fields(PostprocessOptions):
        meta = entry.metadata
        for key in ("filter", "stage"):
            if key not in meta:
                raise ValueError(f"{entry.name} declares no {key}")
        name, stage = meta["filter"], meta["stage"]
        if name in declared and chain[-1][0] != name:
            # A filter takes all its knobs at once, so splitting them across another
            # filter's would ask it for the half it is not handed.
            raise ValueError(
                f"{name} knobs must be declared together: {entry.name} splits them"
            )
        declared.add(name)
        if chain and chain[-1][0] == name:
            chain[-1] = (name, chain[-1][1] + (entry.name,), stage)
        else:
            chain.append((name, (entry.name,), stage))
    return tuple(FilterSpec(name, options, stage) for name, options, stage in chain)


CHAIN = _declared_chain()


def filter_specs(stage: str) -> Tuple[FilterSpec, ...]:
    """The filters of one stage, in chain order."""
    return tuple(spec for spec in CHAIN if spec.stage == stage)


def flat_options(holder: Any) -> Dict[str, Any]:
    """The knobs read off anything that carries them flat — a request, an argparse
    namespace, a wire model — as ``GenerateRequest`` keyword arguments.
    """
    missing = [name for name in POSTPROCESS_NAMES if not hasattr(holder, name)]
    if missing:
        raise ValueError(f"{type(holder).__name__} carries no post-process field "
                         f"{', '.join(missing)}")
    return {name: getattr(holder, name) for name in POSTPROCESS_NAMES}


def options_from(holder: Any) -> PostprocessOptions:
    """The knobs of anything carrying them flat, as the chain wants them."""
    return PostprocessOptions(**flat_options(holder))


__all__ = [
    "CHAIN",
    "DEFAULTS",
    "DECODE",
    "FINAL",
    "POSTPROCESS_NAMES",
    "FilterSpec",
    "PostprocessOptions",
    "filter_specs",
    "flat_options",
    "options_from",
]
