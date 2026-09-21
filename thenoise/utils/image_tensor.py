"""Shared PIL <-> tensor conversions for the pipeline and standalone upscaling.

``PipelineController`` and ``PixelUpscaleController`` both operate on GPU fp32
tensors in ``[-1, 1]`` with shape ``[C, H, W]`` and need the same PIL conversion +
resize helpers.

The channel count of that tensor is the *VAE's* (``DiffusionModel.pixel_channels``:
3 for the RGB VAEs, 4 for the RGBA Qwen-Image 2.1 one), so these helpers are the
two places where a channel count gets decided — on the way in
(:func:`pil_to_pixels`, told how many channels the target wants) and on the way
out (:func:`pixels_to_pil`, which just mirrors what it was handed). Wherever a
stage cannot carry an alpha (an RGB-only VAE, a text encoder's vision tokens, the
pixel-domain upscaler) the alpha is *composited* onto :data:`ALPHA_BACKGROUND`
with :func:`flatten_alpha`, never dropped: PIL's ``convert("RGB")`` throws the
channel away and keeps the RGB of the transparent pixels, which is black at best
and encoding garbage at worst.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Background an alpha channel is composited onto whenever a stage cannot carry it.
# White matches the opaque padding value an RGBA VAE pads RGB inputs with
# (``AutoencoderKLWan22.pad_channel_value`` = 1.0 in the ``[-1, 1]`` range).
ALPHA_BACKGROUND: Tuple[int, int, int] = (255, 255, 255)

# PIL modes that carry a real transparency channel. ``P`` is handled separately:
# a paletted image is transparent only when its header declares one.
_ALPHA_MODES = ("RGBA", "LA")

# PIL mode per channel count of the ``[C, H, W]`` convention (out), and the
# channel counts a destination can ask for (in: no VAE is single-channel).
_MODES = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}
_PIXEL_CHANNELS = (3, 4)


def has_alpha(image: Image.Image) -> bool:
    """True when ``image`` actually carries transparency."""
    if image.mode in _ALPHA_MODES:
        return True
    return image.mode == "P" and "transparency" in image.info


def flatten_alpha(
    image: Image.Image, background: Tuple[int, int, int] = ALPHA_BACKGROUND
) -> Image.Image:
    """Composite any transparency onto ``background`` and return RGB.

    Images without an alpha channel are only reformatted (``convert("RGB")``).
    """
    if not has_alpha(image):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    base = Image.new("RGBA", rgba.size, (*background, 255))
    return Image.alpha_composite(base, rgba).convert("RGB")


def load_image(source: Union[str, "os.PathLike", Image.Image]) -> Image.Image:
    """Open an input image keeping its alpha when it has one.

    The wire layer used to ``convert("RGB")`` on the way in, which fixed the
    channel count before anyone asked the model: a transparent input reaching an
    RGBA model lost its alpha, and one reaching an RGB model kept the transparent
    pixels' RGB. Normalising to RGB or RGBA (never ``P``/``L``/16-bit) keeps the
    question "does this carry transparency?" answerable downstream, where the
    destination's channel count is known.
    """
    image = source if isinstance(source, Image.Image) else Image.open(source)
    image.load()  # resolve the lazy decode (and P-mode transparency) now
    return image.convert("RGBA" if has_alpha(image) else "RGB")


def pil_to_pixels(image: Image.Image, channels: Optional[int] = 3) -> torch.Tensor:
    """PIL -> [C, H, W] fp32 tensor in [-1, 1] with exactly ``channels`` channels.

    ``channels`` is the pixel width of the destination (a VAE's
    ``pixel_channels``, or 3 for the pixel-domain upscaler): an image without
    transparency entering an RGBA destination gets an opaque alpha, one *with*
    transparency entering an RGB destination is composited onto white.

    ``channels=None`` means "as it came in" — RGBA when the image carries
    transparency, RGB when it does not — for a stage that has no opinion of its
    own and only has to avoid losing information (the standalone upscaler).
    """
    if channels is None:
        channels = 4 if has_alpha(image) else 3
    if channels not in _PIXEL_CHANNELS:
        raise ValueError(f"expected 3 or 4 channels, got {channels}")
    src = image.convert("RGBA") if channels == 4 else flatten_alpha(image)
    arr = np.asarray(src).astype(np.float32)  # [H, W, C] 0..255
    return torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0


def pixels_to_pil(pixels: torch.Tensor) -> Image.Image:
    """GPU fp32 [C, H, W] tensor in [-1, 1] -> PIL image with ``C`` channels.

    The mode mirrors the channel count, so pixels decoded by an RGBA VAE become
    an RGBA image and the PNG encoder writes the alpha through.
    """
    c = pixels.shape[0]
    if c not in _MODES:
        raise ValueError(f"cannot build a PIL image from {c} channels")
    x = torch.clamp(pixels, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    arr = np.ascontiguousarray(x.transpose(1, 2, 0))  # C, H, W -> H, W, C
    return Image.fromarray(arr, mode=_MODES[c])


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


def resize_to_area(image: Image.Image, area: int = 384 * 384) -> Image.Image:
    """Scale a PIL image to ``area`` (area-based, aspect-preserving).

    Vision encoders tokenize each patch (e.g. 14x14 for Qwen2.5-VL) into image
    tokens; full-resolution edit images would inject thousands of tokens, drowning
    out a short instruction and overflowing the RoPE buffer.
    """
    w, h = image.size
    scale = (area / (w * h)) ** 0.5
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    if (new_w, new_h) != (w, h):
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


__all__ = [
    "ALPHA_BACKGROUND",
    "has_alpha",
    "flatten_alpha",
    "load_image",
    "pil_to_pixels",
    "pixels_to_pil",
    "resize_to_target",
    "center_crop",
    "resize_to_cover_center_crop",
    "resize_to_area",
]
