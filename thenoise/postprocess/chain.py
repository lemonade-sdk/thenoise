"""Applying the declared chain: one stage's filters, in chain order, over a pixel tensor.

The knobs, their bounds and the order they run in are declared in
:class:`~thenoise.postprocess.options.PostprocessOptions`; this module only holds what
each declared filter name *does*, and the loop that runs a stage.
"""
from __future__ import annotations

from typing import Callable, Dict

from .options import DEFAULTS, FINAL, filter_specs

# An adapter takes the knobs its filter declared, by name, and returns pixels. The
# imports live inside the adapters on purpose: the CLI builds its flags from the
# declaration, and parsing a command line must not have to import torch.


def _nyquist(pixels, **_switch):
    """The notch filter has nothing to tune: its knob is a switch."""
    from .nyquist import nyquist_notch

    return nyquist_notch(pixels)


def _color(pixels, *, saturation: float, vibrance: float, temperature: float, tint: float):
    """One pass over the pixels for the four colour knobs, not four passes."""
    from .color import adjust_color

    return adjust_color(
        pixels,
        saturation=saturation,
        vibrance=vibrance,
        temperature=temperature,
        tint=tint,
    )


def _rcas(pixels, *, sharpening: float):
    from .rcas import rcas

    return rcas(pixels, strength=sharpening)


def _film_grain(pixels, *, film_grain: float):
    """The request asks for grain on a 0-10 scale; the filter works in 0-1."""
    from .film_grain import film_grain as add_grain

    return add_grain(pixels, strength=film_grain / 10.0)


#: Chain filter name -> its adapter.
FILTERS: Dict[str, Callable] = {
    "nyquist_notch": _nyquist,
    "color": _color,
    "rcas": _rcas,
    "film_grain": _film_grain,
}


def apply_stage(pixels, options, stage: str = FINAL):
    """Run one stage of the chain over ``[C, H, W]`` fp32 pixels in ``[-1, 1]``.

    A filter whose knobs all still hold their default is skipped, so a request that
    asks for nothing returns the very tensor it was handed — the chain costs nothing
    when it is not asked for anything.
    """
    for spec in filter_specs(stage):
        values = {name: getattr(options, name) for name in spec.options}
        if all(values[name] == DEFAULTS[name] for name in values):
            continue
        pixels = FILTERS[spec.name](pixels, **values)
    return pixels


__all__ = ["FILTERS", "apply_stage"]
