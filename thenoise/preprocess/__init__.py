"""Control-map extraction: pose, depth and edge maps from a plain image, with no
diffusion model loaded — model-free like the pixel upscalers.
"""
from __future__ import annotations

from .manager import PreprocessorManager

__all__ = ["PreprocessorManager"]
