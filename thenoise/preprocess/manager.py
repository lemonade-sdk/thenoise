"""What can be extracted, and one resident preprocessor at a time: model-free,
device-bound, and last-used stays loaded, like ``PixelUpscalerManager``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Type

from thenoise.utils.device import release_device_memory
from .base import PreprocessOptions, Preprocessor
from .canny import CannyPreprocessor
from .depth import DepthPreprocessor

#: Control-map kind -> the weights-free implementations it always has. Kinds whose
#: implementations are checkpoints on disk come from ``--preprocessor-dir`` discovery.
REGISTRY: Dict[str, List[Type[Preprocessor]]] = {
    CannyPreprocessor.kind: [CannyPreprocessor]
}

#: A checkpoint's ``model_type`` -> the kind it implements, for discovery by reading
#: ``config.json`` rather than trusting a directory name. In v5 Depth-Anything-V2 is
#: merged into the ``depth_anything`` type.
_MODEL_TYPE_KINDS = {"depth_anything": "depth"}

#: The class behind each discoverable kind.
_KIND_CLASSES = {DepthPreprocessor.kind: DepthPreprocessor}


def _kind_of_config(path: str) -> Optional[str]:
    """The kind a checkpoint directory's ``config.json`` implements, or ``None``."""
    with open(path, encoding="utf-8") as f:
        model_type = json.load(f).get("model_type", "")
    return _MODEL_TYPE_KINDS.get(model_type)


class PreprocessorManager:
    """Availability and residency for the preprocessor pool."""

    def __init__(self, device: Any, preprocessor_dir: str = ""):
        self.device = device
        self.preprocessor_dir = preprocessor_dir
        self._resident: Optional[Preprocessor] = None
        self._paths = self._discover()

    def available(self) -> Dict[str, List[str]]:
        """The kinds on offer, each with the names of the implementations it has."""
        result = {}
        for kind in self._known_kinds():
            entries = self._implementations(kind)
            if entries:
                result[kind] = [name for name, _, _ in entries]
        return result

    def defaults(self) -> Dict[str, str]:
        """Each kind's default implementation: the first one on offer."""
        return {kind: names[0] for kind, names in self.available().items() if names}

    def options_for(self, kind: str, **values: Any) -> PreprocessOptions:
        """The option dataclass of ``kind`` filled from the caller's fields, so a
        flag-driven caller never has to name a kind's options class. A known kind
        answers even before its weights are on disk, so the CLI can parse its flags."""
        return self._option_class(kind).OPTIONS(**values)

    def get(self, kind: str, name: str = "") -> Preprocessor:
        """The resident preprocessor for ``kind``, loading it when the resident differs.
        An empty ``name`` is the kind's default; the caller holds the inference lock."""
        entries = self._implementations(kind)
        if not entries:
            raise ValueError(
                f"no preprocessor for {kind!r}; "
                f"available: {', '.join(self.available())}"
            )
        chosen = next((entry for entry in entries if not name or entry[0] == name), None)
        if chosen is None:
            raise ValueError(
                f"no {kind} preprocessor named {name!r}; "
                f"available: {', '.join(entry[0] for entry in entries)}"
            )
        name, cls, path = chosen
        if self._resident is not None and self._resident.name != name:
            self.unload()
        if self._resident is None:
            preprocessor = cls()
            preprocessor.load(self.device, path)
            self._resident = preprocessor
        return self._resident

    def _implementations(self, kind: str) -> List[Tuple[str, Type[Preprocessor], Optional[str]]]:
        """``(name, class, weights path)`` for each implementation of ``kind`` that is
        on offer, in a stable order: the weights-free registry first, then discovered."""
        entries = [(cls.name, cls, None) for cls in REGISTRY.get(kind, ())]
        for name, path in self._paths.get(kind, {}).items():
            entries.append((name, _KIND_CLASSES[kind], path))
        return entries

    def _option_class(self, kind: str) -> Type[Preprocessor]:
        """The class behind ``kind`` (registry or discoverable), or a ValueError for
        a kind the engine does not know."""
        classes = REGISTRY.get(kind)
        if classes:
            return classes[0]
        cls = _KIND_CLASSES.get(kind)
        if cls:
            return cls
        raise ValueError(
            f"no preprocessor for {kind!r}; "
            f"available: {', '.join(self.available())}"
        )

    def _known_kinds(self) -> List[str]:
        """Every kind the engine knows, whether or not a checkpoint is on disk."""
        return sorted(set(REGISTRY) | set(_KIND_CLASSES))

    def _discover(self) -> Dict[str, Dict[str, str]]:
        """``kind -> name -> directory``, from the ``config.json`` of each subdirectory
        of ``--preprocessor-dir``. A kind with no entry here is simply not on offer."""
        paths: Dict[str, Dict[str, str]] = {}
        if not self.preprocessor_dir:
            return paths
        for entry in sorted(os.listdir(self.preprocessor_dir)):
            directory = os.path.join(self.preprocessor_dir, entry)
            config = os.path.join(directory, "config.json")
            if not os.path.isfile(config):
                continue
            kind = _kind_of_config(config)
            if kind:
                paths.setdefault(kind, {})[entry] = directory
        return paths

    def unload(self) -> None:
        """Drop the resident preprocessor; a no-op when nothing is loaded."""
        if self._resident is None:
            return
        self._resident.unload()
        self._resident = None
        release_device_memory(self.device)


__all__ = ["PreprocessorManager", "REGISTRY"]
