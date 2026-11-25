from __future__ import annotations

from typing import Callable, Dict

from thenoise.utils.options import apply_chain

from .options import DEFAULTS, FINISH, filter_specs


def _floaters(mesh, *, floaters: float):
    from .cleanup import drop_islands

    return drop_islands(mesh, floaters)


def _simplify(mesh, *, triangles: float):
    from .simplify import reduce_to

    return reduce_to(mesh, triangles)


FILTERS: Dict[str, Callable] = {"floaters": _floaters, "simplify": _simplify}


def finish_mesh(mesh, options, check: Callable[[], None],
                progress: Callable[..., None]):
    """What the request asked of the cut, in declared order; the mesh itself if nothing."""
    return apply_chain(mesh, options, filter_specs(FINISH), FILTERS, DEFAULTS,
                       check=check,
                       on_filter=lambda _spec: progress("finishing"))


__all__ = ["FILTERS", "finish_mesh"]
