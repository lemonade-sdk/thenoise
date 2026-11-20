"""One progress bar per pipeline stage."""
from __future__ import annotations

from collections.abc import Iterable

from tqdm import tqdm


class StageBars:
    """A ``progress(stage, step, of)`` sink drawing one bar per stage, ``silent`` excepted."""

    def __init__(self, silent: Iterable[str] = ()) -> None:
        self._silent = set(silent)
        self._bars: dict[str, tqdm] = {}

    def __call__(self, stage: str, step: int, of: int) -> None:
        if stage in self._silent:
            return
        bar = self._bars.get(stage)
        if bar is None:
            bar = self._bars[stage] = tqdm(total=of, desc=stage, leave=False)
        bar.update(step - bar.n)
        if of and step >= of:
            bar.close()

    def close(self) -> None:
        for bar in self._bars.values():
            bar.close()


__all__ = ["StageBars"]
