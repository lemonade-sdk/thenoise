from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from thenoise.utils.options import apply_chain

from .options import DEFAULTS, FINISH, filter_specs


@dataclass(frozen=True)
class Ask:
    """What a filter is asked with, beyond its knobs: how to stop."""
    check: Callable[[], None]


def _floaters(mesh, ask: Ask, *, floaters: float):
    from .cleanup import drop_islands
    return drop_islands(mesh, floaters)


def _simplify(mesh, ask: Ask, *, triangles: float):
    from .simplify import reduce_to
    return reduce_to(mesh, triangles)


def _smooth(mesh, ask: Ask, *, smooth: float):
    from .smooth import relax_surface
    return relax_surface(mesh, smooth, ask.check)

FILTERS: Dict[str, Callable] = {"floaters": _floaters, "simplify": _simplify,
                                "smooth": _smooth}

def _asked(spec, ask: Ask):
    """A filter with the run's stop signal already in hand."""
    def run(carrier, **knobs):
        return FILTERS[spec.name](carrier, ask, **knobs)

    return run


def finish_mesh(mesh, options, check: Callable[[], None], progress: Callable[..., None]):
    """What the request asked of the cut, in declared order; the mesh itself if nothing.

    Every filter works where the mesh already is. The reducer is the one that needs the
    host, and it takes it for itself, so nothing here is handed over wholesale - which is
    what left the piece rule and the relaxation running on the CPU beside a free GPU.
    """
    stages = filter_specs(FINISH)
    ask = Ask(check=check)
    return apply_chain(mesh, options, stages,
                       {spec.name: _asked(spec, ask) for spec in stages}, DEFAULTS,
                       check=check,
                       on_filter=lambda _spec: progress(FINISH))


__all__ = ["Ask", "FILTERS", "finish_mesh"]
