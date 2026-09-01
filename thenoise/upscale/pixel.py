"""Standalone pixel-domain upscaler manager.

Pixel upscaling operates purely in *pixel space* (post-decode / post-process)
and needs no diffusion model at all. It is therefore NOT a model concern: the
``upscaler_dir`` is server configuration (like host/port), and the loaded
upscaler + per-name detected scales live here in one model-free component.

This component is shared by the generation pipeline controller AND by a future
standalone ``/upscale`` endpoint (which accepts an input image and runs only
pixel upscaling). Only the last-used upscaler is kept loaded (switched on
change).

Not thread-safe: loading weights onto the device must be serialized by the
caller (the pipeline controller holds an inference lock around ``apply``).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Union

import torch

from thenoise.upscale import load_pixel_upscaler, detect_pixel_upscaler_scale
from thenoise.utils.model_dir import (
    ensure_safetensors,
    strip_safetensors,
    resolve_in_dir,
    list_safetensors,
)

logger = logging.getLogger(__name__)


class PixelUpscalerManager:
    """Owns the pixel-domain upscaler pool: dir, scales, last-used loaded model."""

    def __init__(self, upscaler_dir: str, device: Union[str, torch.device]):
        self.upscaler_dir = upscaler_dir
        self.device = device
        self._pixel_upscaler = None
        self._pixel_upscaler_name: Optional[str] = None  # currently loaded name
        self._pixel_upscaler_scales: Dict[str, int] = {}  # per-name detected scale

    # ------------------------------------------------------------- listing
    def list(self) -> list[str]:
        """List available pixel-upscaler names relative to ``upscaler_dir``.

        Names are relative paths with the ``.safetensors`` suffix stripped, so
        they can be used directly as a request's ``pixel_upscaler`` value.
        """
        return list_safetensors(self.upscaler_dir)

    # ------------------------------------------------------------- validation
    def _parse_name(self, name: str) -> str:
        """Return the canonical pixel-upscaler name (strip optional suffix)."""
        return strip_safetensors(name)

    def _resolve_path(self, filename: str) -> str:
        """Resolve a pixel-upscaler filename within ``upscaler_dir`` (guarded)."""
        return resolve_in_dir(self.upscaler_dir, filename)

    def validate(self, name: str) -> str:
        """Validate a pixel-upscaler name; return its canonical form.

        Raises if no ``upscaler_dir`` is configured or the named model does not
        exist in it.
        """
        if not self.upscaler_dir:
            raise ValueError(
                "no pixel upscaler configured; pass --upscaler-dir PATH "
                "(or run scripts/download_esrgan.py)"
            )
        name = self._parse_name(name)
        filepath = self._resolve_path(ensure_safetensors(name))
        if not os.path.isfile(filepath):
            raise ValueError(
                f"pixel upscaler '{name}' not found in {self.upscaler_dir}"
            )
        return name

    # ------------------------------------------------------------- scale
    def scale(self, name: str) -> int:
        """Detected scale of the requested pixel upscaler (0 if none), cached.

        Reads only the safetensors header on first use per name; the value is
        then reused. Tests may pre-populate ``_pixel_upscaler_scales`` to skip
        file access.
        """
        if not self.upscaler_dir or not name:
            return 0
        name = self._parse_name(name)
        scale = self._pixel_upscaler_scales.get(name)
        if scale is None:
            filepath = self._resolve_path(ensure_safetensors(name))
            scale = detect_pixel_upscaler_scale(filepath)
            self._pixel_upscaler_scales[name] = scale
        return scale

    # ------------------------------------------------------------- switching
    def switch(self, name: str) -> None:
        """Load the requested pixel upscaler, keeping only the last-used loaded.

        Swaps (unloads) any previously loaded upscaler when the requested name
        differs; repeated requests with the same name are no-ops. Must be called
        under the caller's lock (it loads weights onto the device).
        """
        name = self._parse_name(name)
        if self._pixel_upscaler_name == name:
            return  # no-op: same upscaler
        filepath = self._resolve_path(ensure_safetensors(name))
        logger.info("Loading pixel upscaler: %s", filepath)
        self._pixel_upscaler, scale = load_pixel_upscaler(
            filepath, device=self.device
        )
        self._pixel_upscaler_name = name
        self._pixel_upscaler_scales[name] = scale

    # ------------------------------------------------------------- execution
    def apply(
        self,
        name: str,
        pixels: torch.Tensor,
        scale: int,
    ) -> torch.Tensor:
        """Apply the pixel-domain upscaler by ``scale``x (if > 0).

        Loads (or reuses) the requested pixel upscaler, keeping only the
        last-used model loaded. The model operates on RGB in [0, 1] while the
        pipeline's decoded pixels are in [-1, 1]; convert to [0, 1] before the
        model and back afterwards so downstream postprocessing stays unchanged.
        """
        if not scale or not name:
            return pixels
        self.switch(name)
        model = self._pixel_upscaler
        x = (pixels.unsqueeze(0) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        with torch.no_grad():
            out = model.forward_tiled(x)
        out = out * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return out[0]


__all__ = ["PixelUpscalerManager"]
