"""Standalone control-map extraction for the ``/preprocess`` endpoint and CLI: owns the
inference lock like ``PixelUpscaleController``, and may run beside a loaded model.
"""
from __future__ import annotations

from dataclasses import MISSING, asdict, fields
from typing import Any, Dict

from PIL import Image

from thenoise.inference import inference, inference_lock
from thenoise.preprocess.base import PreprocessOptions, Preprocessor
from thenoise.preprocess.manager import PreprocessorManager
from thenoise.utils.png import build_preprocess_pnginfo


class PreprocessController:
    """Owns extraction: lock + inference mode + run + PNG provenance."""

    def __init__(self, preprocessors: PreprocessorManager):
        self._preprocessors = preprocessors
        self._lock = inference_lock

    def run(
        self,
        image: Image.Image,
        kind: str,
        options: PreprocessOptions,
        name: str = "",
    ) -> Image.Image:
        """Extract the ``kind`` control map from a PIL image, at the image's own size.
        ``options`` is the kind's own option dataclass, ``name`` its implementation."""
        with inference():
            with self._lock:
                preprocessor: Preprocessor = self._preprocessors.get(kind, name)
                out = preprocessor.run(image, options)

        out._pnginfo = build_preprocess_pnginfo(
            image, kind, preprocessor.name, _options_dict(options)
        )
        return out


def _options_dict(options: PreprocessOptions) -> Dict[str, Any]:
    """Only the fields the request changed, so the PNG records intent, not defaults."""
    defaults = {
        field.name: field.default for field in fields(options) if field.default is not MISSING
    }
    return {k: v for k, v in asdict(options).items() if v != defaults.get(k)}


__all__ = ["PreprocessController"]
