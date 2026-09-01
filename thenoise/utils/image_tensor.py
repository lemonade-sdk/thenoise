"""Shared PIL <-> tensor conversions for the pipeline and standalone upscaling.

``PipelineController`` and ``PixelUpscaleController`` both operate on GPU fp32
tensors in ``[-1, 1]`` with shape ``[C, H, W]`` and need the same PIL conversion + resize helpers.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pil_to_pixels(image: Image.Image) -> torch.Tensor:
    """PIL RGB -> [C, H, W] fp32 tensor in [-1, 1] (the range upscalers expect)."""
    arr = np.asarray(image.convert("RGB")).astype(np.float32)  # [H, W, C] 0..255
    return torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0


def pixels_to_pil(pixels: torch.Tensor) -> Image.Image:
    """GPU fp32 [C, H, W] tensor in [-1, 1] -> PIL RGB image."""
    x = torch.clamp(pixels, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    return Image.fromarray(x.transpose(1, 2, 0))  # C, H, W -> H, W, C


def resize_to_target(
    pixels: torch.Tensor, target_w: int, target_h: int
) -> torch.Tensor:
    """GPU bilinear resize of [C, H, W] to (target_w, target_h); no-op if equal."""
    c, h, w = pixels.shape
    if (w, h) == (target_w, target_h):
        return pixels
    with torch.no_grad():
        return F.interpolate(
            pixels.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0]


def center_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop ``image`` to ``(width, height)``."""
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def resize_to_cover_center_crop(
    image: Image.Image, width: int, height: int
) -> Image.Image:
    """ComfyUI-style ref resize: scale to cover ``(width, height)``, center-crop.

    Images matching the target aspect ratio are only resized; ComfyUI does not pad.
    """
    if (image.width, image.height) == (width, height):
        return image
    scale = max(width / image.width, height / image.height)
    new_w = round(image.width * scale)
    new_h = round(image.height * scale)
    scaled = image.resize((new_w, new_h), Image.LANCZOS)
    return center_crop(scaled, width, height)


__all__ = [
    "pil_to_pixels",
    "pixels_to_pil",
    "resize_to_target",
    "center_crop",
    "resize_to_cover_center_crop",
]
