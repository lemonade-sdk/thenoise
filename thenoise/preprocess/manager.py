"""What can be extracted, and one resident preprocessor at a time: model-free,
device-bound, and last-used stays loaded, like ``PixelUpscalerManager``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from thenoise.utils.device import release_device_memory
from .base import PreprocessOptions, Preprocessor
from .canny import CannyPreprocessor

#: Control-map kind -> the implementations it can choose between. A kind with no entry
#: here is offered by no endpoint, CLI listing or UI dropdown.
REGISTRY: Dict[str, List[Type[Preprocessor]]] = {
    CannyPreprocessor.kind: [CannyPreprocessor]
}


class PreprocessorManager:
    """Availability and residency for the preprocessor pool."""

    def __init__(self, device: Any):
        self.device = device
        self._resident: Optional[Preprocessor] = None

    def available(self) -> Dict[str, List[str]]:
        """The kinds on offer, each with the names of the implementations it has."""
        return {
            kind: [cls.name for cls in classes] for kind, classes in self._kinds()
        }

    def defaults(self) -> Dict[str, str]:
        """Each kind's default implementation: the first one registered."""
        return {kind: classes[0].name for kind, classes in self._kinds() if classes}

    def options_for(self, kind: str, **values: Any) -> PreprocessOptions:
        """The option dataclass of ``kind`` filled from the caller's fields, so a
        flag-driven caller never has to name a kind's options class."""
        return self._implementations(kind)[0].OPTIONS(**values)

    def get(self, kind: str, name: str = "") -> Preprocessor:
        """The resident preprocessor for ``kind``, loading it when the resident differs.
        An empty ``name`` is the kind's default; the caller holds the inference lock."""
        classes = self._implementations(kind)
        chosen = next((cls for cls in classes if not name or cls.name == name), None)
        if chosen is None:
            raise ValueError(
                f"no {kind} preprocessor named {name!r}; "
                f"available: {', '.join(cls.name for cls in classes)}"
            )
        if self._resident is not None and self._resident.name != chosen.name:
            self.unload()
        if self._resident is None:
            self._resident = chosen()
            self._resident.load(self.device)
        return self._resident

    def _kinds(self):
        """The registry in a stable order, as ``kind -> implementation classes``."""
        return sorted(REGISTRY.items())

    def _implementations(self, kind: str) -> List[Type[Preprocessor]]:
        """The classes behind ``kind``, or a ValueError naming what is available."""
        classes = REGISTRY.get(kind)
        if not classes:
            raise ValueError(
                f"no preprocessor for {kind!r}; available: {', '.join(sorted(REGISTRY))}"
            )
        return classes

    def unload(self) -> None:
        """Drop the resident preprocessor; a no-op when nothing is loaded."""
        if self._resident is None:
            return
        self._resident.unload()
        self._resident = None
        release_device_memory(self.device)


__all__ = ["REGISTRY", "PreprocessorManager"]
