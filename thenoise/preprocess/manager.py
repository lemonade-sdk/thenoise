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
from .lineart import LineartPreprocessor
from .pose import PosePreprocessor
from .softedge import SoftedgePreprocessor

#: Control-map kind -> the weights-free implementations it always has; the rest come from discovery.
REGISTRY: Dict[str, List[Type[Preprocessor]]] = {
    CannyPreprocessor.kind: [CannyPreprocessor],
    LineartPreprocessor.kind: [LineartPreprocessor],
    SoftedgePreprocessor.kind: [SoftedgePreprocessor],
}

#: A checkpoint's ``model_type`` -> what it is for, by reading ``config.json`` rather
#: than trusting a directory name. A person detector is a pose companion, not a kind.
_MODEL_TYPE_KINDS = {
    "depth_anything": "depth",
    "vitpose": "pose",
    "rt_detr": "detector",
}

#: The class behind each discoverable kind.
_KIND_CLASSES = {
    DepthPreprocessor.kind: DepthPreprocessor,
    PosePreprocessor.kind: PosePreprocessor,
}

#: The companion checkpoints a kind needs, as ``kind -> (load keyword, discovered kind)``.
_RESOURCES = {PosePreprocessor.kind: ("detectors", "detector")}


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
            if entries and not self._missing(kind):
                result[kind] = [name for name, _, _ in entries]
        return result

    def missing(self) -> Dict[str, str]:
        """Each kind the engine knows that cannot run yet, with what would complete it."""
        return {
            kind: f"{reason}; run '{self._option_class(kind).DOWNLOAD}' to fetch it"
            for kind in self._known_kinds()
            if (reason := self._missing(kind))
        }

    def detectors(self) -> List[str]:
        """The person detectors on disk, which a ``pose`` request picks from."""
        return sorted(self._paths.get("detector", {}))

    def defaults(self) -> Dict[str, str]:
        """Each kind's default implementation: the first one on offer."""
        return {kind: names[0] for kind, names in self.available().items() if names}

    def options_for(self, kind: str, **values: Any) -> PreprocessOptions:
        """The option dataclass of ``kind`` filled from the caller's fields."""
        return self._option_class(kind).OPTIONS(**values)

    def get(self, kind: str, name: str = "") -> Preprocessor:
        """The resident preprocessor for ``kind``, loading it when the resident differs.
        An empty ``name`` is the kind's default; the caller holds the inference lock."""
        reason = self._missing(kind)
        if reason:
            raise ValueError(
                f"the {kind} preprocessor {reason}; run "
                f"'{self._option_class(kind).DOWNLOAD}' to fetch it"
            )
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
            preprocessor.load(self.device, path, **self._resources(kind))
            self._resident = preprocessor
        return self._resident

    def _implementations(self, kind: str) -> List[Tuple[str, Type[Preprocessor], Optional[str]]]:
        """``(name, class, weights path)`` per implementation of ``kind`` on offer."""
        entries = [(cls.name, cls, None) for cls in REGISTRY.get(kind, ())]
        for name, path in self._paths.get(kind, {}).items():
            entries.append((name, _KIND_CLASSES[kind], path))
        return entries

    def _option_class(self, kind: str) -> Type[Preprocessor]:
        """The class behind ``kind`` (registry or discoverable), or a ValueError for an unknown kind."""
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

    def _missing(self, kind: str) -> str:
        """What ``kind`` still needs before it can run, or ``""`` when it can: a
        discoverable one needs its checkpoint, and pose needs a person detector too."""
        if kind in _KIND_CLASSES and not self._implementations(kind):
            return "has no checkpoint in the preprocessor directory"
        if kind == PosePreprocessor.kind and not self.detectors():
            return "needs a person detector (RT-DETR) beside its keypoint model"
        return ""

    def _resources(self, kind: str) -> Dict[str, Any]:
        """The companion checkpoints ``kind`` loads beside its own weights."""
        companion = _RESOURCES.get(kind)
        if not companion:
            return {}
        keyword, discovered = companion
        return {keyword: dict(self._paths.get(discovered, {}))}

    def _known_kinds(self) -> List[str]:
        """Every kind the engine knows, whether or not a checkpoint is on disk."""
        return sorted(set(REGISTRY) | set(_KIND_CLASSES))

    def _discover(self) -> Dict[str, Dict[str, str]]:
        """``kind -> name -> directory``, from each subdirectory's ``config.json``."""
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
