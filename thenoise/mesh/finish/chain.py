from __future__ import annotations

import logging
import time
from typing import Callable, Dict

from thenoise.utils.device import release_device_memory
from thenoise.utils.options import apply_chain

from .options import DEFAULTS, FINISH, filter_specs

logger = logging.getLogger(__name__)


def _floaters(mesh, *, floaters: float):
    from .cleanup import drop_islands

    return drop_islands(mesh, floaters)


def _simplify(mesh, *, triangles: float):
    from .simplify import reduce_to

    return reduce_to(mesh, triangles)


FILTERS: Dict[str, Callable] = {"floaters": _floaters, "simplify": _simplify}


def _reported(spec):
    """A filter that says what it was handed, and what it gave back."""
    def run(carrier, **knobs):
        before = carrier.faces.shape[0]
        logger.info("finishing %s over %d triangles", spec.name, before)
        started = time.monotonic()
        kept = FILTERS[spec.name](carrier, **knobs)
        logger.info("finishing %s: %d -> %d triangles in %.1fs", spec.name, before,
                    kept.faces.shape[0], time.monotonic() - started)
        return kept

    return run


def finish_mesh(mesh, options, check: Callable[[], None],
                progress: Callable[..., None]):
    """What the request asked of the cut, in declared order; the mesh itself if nothing.

    Runs on the host: none of it wants the compute device, which still carries the stage's weights.
    """
    device = mesh.vertices.device
    host = mesh if device.type == "cpu" else mesh.cpu()
    if device.type != "cpu":
        # The reduction wants ~330 B per triangle of RAM the stage's cache is already holding.
        release_device_memory(device)
    stages = filter_specs(FINISH)
    return apply_chain(host, options, stages,
                       {spec.name: _reported(spec) for spec in stages}, DEFAULTS,
                       check=check,
                       on_filter=lambda spec: progress(f"finishing: {spec.name}"))


__all__ = ["FILTERS", "finish_mesh"]
