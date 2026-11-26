from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Dict

from thenoise.utils.device import release_device_memory
from thenoise.utils.options import apply_chain

from .options import DEFAULTS, FINISH, filter_specs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ask:
    """What a filter is asked with, beyond its knobs: how to stop, and where it was painted."""

    check: Callable[[], None]
    spots: tuple[tuple[float, float, float, float], ...] = ()


def _floaters(mesh, ask: Ask, *, floaters: float):
    from .cleanup import drop_islands

    return drop_islands(mesh, floaters)


def _simplify(mesh, ask: Ask, *, triangles: float):
    from .simplify import reduce_to

    return reduce_to(mesh, triangles)


def _smooth(mesh, ask: Ask, *, smooth: float):
    from .smooth import relax_surface

    return relax_surface(mesh, smooth, ask.check, ask.spots)

FILTERS: Dict[str, Callable] = {"floaters": _floaters, "simplify": _simplify,
                                "smooth": _smooth}

def _reported(spec, ask: Ask):
    """A filter that says what it was handed, and what it gave back."""
    def run(carrier, **knobs):
        before = carrier.faces.shape[0]
        logger.info("finishing %s over %d triangles", spec.name, before)
        started = time.monotonic()
        kept = FILTERS[spec.name](carrier, ask, **knobs)
        logger.info("finishing %s: %d -> %d triangles in %.1fs", spec.name, before,
                    kept.faces.shape[0], time.monotonic() - started)
        return kept

    return run


def finish_mesh(mesh, options, check: Callable[[], None], progress: Callable[..., None],
                spots: Sequence[Sequence[float]] = ()):
    """What the request asked of the cut, in declared order; the mesh itself if nothing."""
    device = mesh.vertices.device
    host = mesh if device.type == "cpu" else mesh.cpu()   # the device carries the weights
    if device.type != "cpu":
        # The reduction wants ~330 B per triangle of RAM the stage's cache is already holding.
        release_device_memory(device)
    stages = filter_specs(FINISH)
    ask = Ask(check=check, spots=tuple(tuple(one) for one in spots))
    return apply_chain(host, options, stages,
                       {spec.name: _reported(spec, ask) for spec in stages}, DEFAULTS,
                       check=check,
                       on_filter=lambda _spec: progress(FINISH))


__all__ = ["Ask", "FILTERS", "finish_mesh"]
