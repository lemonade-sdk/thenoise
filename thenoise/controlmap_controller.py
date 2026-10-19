"""Standalone control-map extraction for the ``/controlmap`` endpoint and CLI: owns the
inference lock like ``PixelUpscaleController``, and may run beside a loaded model.
"""
from __future__ import annotations

from dataclasses import MISSING, asdict, fields
from typing import Any, Dict

from PIL import Image

from thenoise.inference import inference, inference_lock
from thenoise.controlmap.base import ControlMapOptions, ControlMap
from thenoise.controlmap.manager import ControlMapManager
from thenoise.utils.png import build_controlmap_pnginfo


class ControlMapController:
    """Owns extraction: lock + inference mode + run + PNG provenance."""

    def __init__(self, controlmaps: ControlMapManager):
        self._controlmaps = controlmaps
        self._lock = inference_lock

    def run(
        self,
        image: Image.Image,
        kind: str,
        options: ControlMapOptions,
        name: str = "",
    ) -> Image.Image:
        """Extract the ``kind`` control map from a PIL image, at the image's own size.
        ``options`` is the kind's own option dataclass, ``name`` its implementation.
        The weights it needed are released on the way out, failed run included.
        """
        with inference():
            with self._lock:
                try:
                    controlmap: ControlMap = self._controlmaps.get(kind, name)
                    out = controlmap.run(image, options)
                finally:
                    # One map, not a session: whatever runs next may need this room.
                    self._controlmaps.unload()

        out._pnginfo = build_controlmap_pnginfo(
            image, kind, controlmap.name, _options_dict(options)
        )
        return out


def _options_dict(options: ControlMapOptions) -> Dict[str, Any]:
    """Only the fields the request changed, so the PNG records intent, not defaults."""
    defaults = {
        field.name: field.default for field in fields(options) if field.default is not MISSING
    }
    return {k: v for k, v in asdict(options).items() if v != defaults.get(k)}


__all__ = ["ControlMapController"]
