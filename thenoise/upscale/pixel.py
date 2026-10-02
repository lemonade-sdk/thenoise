"""Standalone pixel-domain upscaler manager.

Pixel upscaling operates purely in *pixel space* (post-decode / post-process) and
needs no diffusion model, so ``upscaler_dir`` is server configuration (like
host/port). Only the last-used upscaler is kept loaded (switched on change).

Not thread-safe: loading weights onto the device must be serialized by the caller.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Union

import torch

from thenoise.upscale import load_pixel_upscaler, detect_pixel_upscaler_scale
from thenoise.utils.image_tensor import resize_to_target
from thenoise.utils.model_dir import (
    ensure_safetensors,
    strip_safetensors,
    resolve_in_dir,
    list_safetensors,
)

logger = logging.getLogger(__name__)

# White in the pipeline's ``[-1, 1]`` pixel range.
_WHITE = 1.0


class PixelUpscalerManager:
    """Owns the pixel-domain upscaler pool: dir, scales, last-used loaded model."""

    def __init__(self, upscaler_dir: str, device: Union[str, torch.device]):
        self.upscaler_dir = upscaler_dir
        self.device = device
        self._pixel_upscaler = None
        self._pixel_upscaler_name: Optional[str] = None
        self._pixel_upscaler_scales: Dict[str, int] = {}

    # ------------------------------------------------------------- listing
    def list(self) -> list[str]:
        """Available pixel-upscaler names: relative paths, ``.safetensors`` stripped."""
        return list_safetensors(self.upscaler_dir)

    # ------------------------------------------------------------- validation
    def _parse_name(self, name: str) -> str:
        """Return the canonical pixel-upscaler name (strip optional suffix)."""
        return strip_safetensors(name)

    def _resolve_path(self, filename: str) -> str:
        """Resolve a pixel-upscaler filename within ``upscaler_dir`` (guarded)."""
        return resolve_in_dir(self.upscaler_dir, filename)

    def validate(self, name: str) -> str:
        """Validate a pixel-upscaler name; return its canonical form."""
        if not self.upscaler_dir:
            raise ValueError(
                "no pixel upscaler configured; pass --upscaler-dir PATH "
                "(or run scripts/download.py --model esrgan)"
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
        """Detected scale of the requested pixel upscaler (0 if none), cached per name."""
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
        """Load the requested pixel upscaler, unloading the previously loaded one.

        Repeated requests with the same name are no-ops. Must be called under the
        caller's lock (it loads weights onto the device).
        """
        name = self._parse_name(name)
        if self._pixel_upscaler_name == name:
            return
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

        The model operates on RGB in [0, 1] while the pipeline's decoded pixels are
        in [-1, 1]. It is a 3-channel model, so an RGBA input is composited onto
        white and its alpha resampled by the same factor and re-attached.
        """
        if not scale or not name:
            return pixels
        self.switch(name)
        model = self._pixel_upscaler

        rgb = pixels[:3]
        alpha = pixels[3:4] if pixels.shape[0] > 3 else None
        if alpha is not None:
            a = (alpha + 1.0) / 2.0  # alpha as a [0, 1] blend factor
            rgb = rgb * a + _WHITE * (1.0 - a)

        x = (rgb.unsqueeze(0) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        out = model.forward_tiled(x)
        out = (out * 2.0 - 1.0)[0]  # [0, 1] -> [-1, 1], batch axis gone

        if alpha is not None:
            out = torch.cat(
                [out, resize_to_target(alpha, out.shape[-1], out.shape[-2])], dim=0
            )
        return out


__all__ = ["PixelUpscalerManager"]
