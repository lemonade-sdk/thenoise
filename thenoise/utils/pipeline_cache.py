"""One slot per named stage; writing one clears every stage downstream of it."""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Iterable, Sequence, Tuple

if TYPE_CHECKING:
    from PIL import Image


def content_digest(images: Iterable["Image.Image"]) -> Tuple[str, ...]:
    """Each image's decoded pixels, hashed: a re-saved picture is still one reference."""
    return tuple(hashlib.md5(image.tobytes()).hexdigest() for image in images)


class _CacheSlot:
    """Single-entry cache slot with explicit release on overwrite."""

    __slots__ = ("_key", "_value")

    def __init__(self) -> None:
        self._key: Any = None
        self._value: Any = None

    def is_hit(self, key: Any) -> bool:
        return self._key == key

    @property
    def value(self) -> Any:
        return self._value

    def store(self, key: Any, value: Any) -> None:
        self._key = key
        self._value = value

    def clear(self) -> None:
        self._key = None
        self._value = None


class PipelineCache:
    """One slot per named stage, each clearing everything after it."""

    def __init__(self, stages: Sequence[str] = ("reference", "prompt", "sampling",
                                                "decode")) -> None:
        self._slots = {name: _CacheSlot() for name in stages}
        self._order = tuple(stages)

    def hit(self, stage: str, key: Any) -> bool:
        """Whether ``stage`` holds exactly ``key``; an empty slot never hits."""
        return key is not None and self._slot(stage).is_hit(key)

    def get(self, stage: str) -> Any:
        return self._slot(stage).value

    def store(self, stage: str, key: Any, value: Any) -> None:
        # Released before the next stage allocates, so two copies never coexist.
        slot = self._slot(stage)
        slot.store(key, value)
        for later in self._order[self._order.index(stage) + 1:]:
            self._slots[later].clear()

    def clear_all(self) -> None:
        for slot in self._slots.values():
            slot.clear()

    def _slot(self, stage: str) -> _CacheSlot:
        try:
            return self._slots[stage]
        except KeyError:
            raise ValueError(f"no cache stage {stage!r} "
                             f"(known: {', '.join(self._order)})") from None


__all__ = ["PipelineCache", "content_digest"]
