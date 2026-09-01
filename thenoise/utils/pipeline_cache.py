"""Single-entry pipeline cache with cascade invalidation and immediate release.

Each stage holds at most one (key, value) pair. When a stage is written to
with a new key, the old value is explicitly released *and* all downstream
stages are cleared immediately. This avoids keeping two large tensor copies
in memory during GC windows.

Stage dependency graph (upstream -> downstream):

    reference  ->  prompt  ->  sampling  ->  decode

``reference`` caches the encoded input image (edit path only). A miss at
``reference`` clears ``prompt``, ``sampling`` and ``decode``.
A miss at ``prompt`` clears ``sampling`` and ``decode``.
A miss at ``sampling`` clears ``decode``.
A miss at ``decode`` clears nothing downstream.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple


class _CacheSlot:
    """Single-entry cache slot with explicit release on overwrite."""

    __slots__ = ("_key", "_value")

    def __init__(self) -> None:
        self._key: Any = None
        self._value: Any = None

    @property
    def key(self) -> Any:
        return self._key

    @property
    def value(self) -> Any:
        return self._value

    def is_hit(self, key: Any) -> bool:
        """Return True if the slot's key matches the given key."""
        return self._key == key

    def store(self, key: Any, value: Any) -> None:
        """Store a new (key, value), releasing the old value immediately."""
        self._key = key
        self._value = value

    def clear(self) -> None:
        """Release the cached value and reset the key."""
        self._key = None
        self._value = None


class PipelineCache:
    """Four-stage pipeline cache with cascade invalidation.

    When any stage is invalidated (key mismatch), the old value is released
    immediately and all downstream stages are cleared. This prevents holding
    two copies of large tensors in memory simultaneously.

    The optional ``reference`` stage (edit path only) sits upstream of prompt.
    """

    __slots__ = ("_reference", "_prompt", "_sampling", "_decode")

    def __init__(self) -> None:
        self._reference = _CacheSlot()
        self._prompt = _CacheSlot()
        self._sampling = _CacheSlot()
        self._decode = _CacheSlot()

    # --------------------------------------------------------------- reference

    @property
    def reference_key(self) -> Any:
        return self._reference.key

    def reference_hit(self, key: Tuple) -> bool:
        return self._reference.is_hit(key)

    def reference_get(self) -> Any:
        return self._reference.value

    def reference_store(self, key: Tuple, value: Any) -> None:
        """Store the encoded reference latent, cascading invalidation downstream."""
        self._reference.store(key, value)
        self._prompt.clear()
        self._sampling.clear()
        self._decode.clear()

    # ------------------------------------------------------------------ prompt

    @property
    def prompt_key(self) -> Any:
        return self._prompt.key

    def prompt_hit(self, key: Tuple) -> bool:
        return self._prompt.is_hit(key)

    def prompt_get(self) -> Any:
        return self._prompt.value

    def prompt_store(self, key: Tuple, value: Any) -> None:
        """Store prompt result, cascading invalidation downstream."""
        self._prompt.store(key, value)
        self._sampling.clear()
        self._decode.clear()

    # ------------------------------------------------------------------ sampling

    @property
    def sampling_key(self) -> Any:
        return self._sampling.key

    def sampling_hit(self, key: Tuple) -> bool:
        return self._sampling.is_hit(key)

    def sampling_get(self) -> Any:
        return self._sampling.value

    def sampling_store(self, key: Tuple, value: Any) -> None:
        """Store sampling result, cascading invalidation downstream."""
        self._sampling.store(key, value)
        self._decode.clear()

    # ------------------------------------------------------------------ decode

    @property
    def decode_key(self) -> Any:
        return self._decode.key

    def decode_hit(self, key: Tuple) -> bool:
        return self._decode.is_hit(key)

    def decode_get(self) -> Any:
        return self._decode.value

    def decode_store(self, key: Tuple, value: Any) -> None:
        """Store decode result. No downstream stages to invalidate."""
        self._decode.store(key, value)

    # ------------------------------------------------------------------ bulk

    def clear_all(self) -> None:
        """Release every cached value and reset all keys."""
        self._reference.clear()
        self._prompt.clear()
        self._sampling.clear()
        self._decode.clear()
