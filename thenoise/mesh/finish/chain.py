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
    """What the request asked of the cut, in declared order; the mesh itself if nothing.

    The chain runs on the host and returns a host mesh: the reduction is a numpy library and the
    filter is gathers, so none of it wants the compute device, which still carries the weights and
    the allocator's cache of the stage that made the cut. Writers start on the host anyway.
    """
    host = mesh if mesh.vertices.device.type == "cpu" else mesh.cpu()
    return apply_chain(host, options, filter_specs(FINISH), FILTERS, DEFAULTS,
                       check=check,
                       on_filter=lambda _spec: progress("finishing"))


__all__ = ["FILTERS", "finish_mesh"]
