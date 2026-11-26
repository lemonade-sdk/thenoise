from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

from thenoise.utils.pipeline_cache import PipelineCache

logger = logging.getLogger(__name__)


class Stages:
    """The stage boundary of one controller: residency, retried work, cache slot."""

    def __init__(self, model, cache: PipelineCache):
        self._model = model
        self._cache = cache

    def run(self, work: Callable[..., Any], *args, resident: Sequence[str], pixels: int,
            cache: Optional[str] = None, key: Any = None,
            prepare: Optional[Callable[[], None]] = None, **kwargs) -> Any:
        """``work(*args)`` with ``resident`` held, cached under ``cache`` when named."""
        if cache is not None and self._cache.hit(cache, key):
            logger.info("%s stage reused: nothing recomputed", cache)
            return self._cache.get(cache)
        with self._model.memory.stage(*resident, pixels=pixels) as stage:
            if prepare is not None:
                prepare()
            value = stage.run(work, *args, **kwargs)
        if cache is not None:
            self._cache.store(cache, key, value)
        return value


__all__ = ["Stages"]
