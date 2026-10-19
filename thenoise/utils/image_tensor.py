"""PIL <-> tensor conversions on ``[C, H, W]`` fp32 pixels in ``[-1, 1]``.

The channel count of that tensor is the *VAE's* (``DiffusionModel.pixel_channels``),
so it gets decided on the way in (:func:`pil_to_pixels`, told how many channels the
destination wants) and on the way out (:func:`pixels_to_pil`, which mirrors what it
was handed). Wherever a stage cannot carry an alpha, :func:`flatten_alpha`
*composites* it onto :data:`ALPHA_BACKGROUND`: PIL's ``convert("RGB")`` keeps the RGB
of the transparent pixels, which is garbage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Background an alpha channel is composited onto whenever a stage cannot carry it.
# Matches the opaque padding value an RGBA VAE pads RGB inputs with.
ALPHA_BACKGROUND: Tuple[int, int, int] = (255, 255, 255)

# PIL modes that carry a real transparency channel. ``P`` is transparent only when
# its header declares one.
_ALPHA_MODES = ("RGBA", "LA")

# PIL mode per channel count of the ``[C, H, W]`` convention (out), and the channel
# counts a destination can ask for (in).
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

    Normalises to RGB or RGBA (never ``P``/``L``/16-bit) so "does this carry
    transparency?" stays answerable downstream, where the destination's channel
    count is known.
    """
    image = source if isinstance(source, Image.Image) else Image.open(source)
    image.load()  # resolve the lazy decode (and P-mode transparency) now
    return image.convert("RGBA" if has_alpha(image) else "RGB")


def pil_to_pixels(image: Image.Image, channels: Optional[int] = 3) -> torch.Tensor:
    """PIL -> [C, H, W] fp32 tensor in [-1, 1] with exactly ``channels`` channels.

    ``channels`` is the pixel width of the destination: an image without
    transparency entering an RGBA destination gets an opaque alpha, one *with*
    transparency entering an RGB destination is composited onto white. ``None``
    means "as it came in".
    """
    if channels is None:
        channels = 4 if has_alpha(image) else 3
    if channels not in _PIXEL_CHANNELS:
        raise ValueError(f"expected 3 or 4 channels, got {channels}")
    src = image.convert("RGBA") if channels == 4 else flatten_alpha(image)
    arr = np.asarray(src).astype(np.float32)  # [H, W, C] 0..255
    return torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0


def pixels_to_pil(pixels: torch.Tensor) -> Image.Image:
    """GPU fp32 [C, H, W] tensor in [-1, 1] -> PIL image with ``C`` channels."""
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


def resize_to_area(image: Image.Image, area: int = 384 * 384) -> Image.Image:
    """Scale a PIL image to ``area`` (area-based, aspect-preserving).

    Vision encoders tokenize each patch into image tokens; a full-resolution edit
    image would inject thousands of them, drowning out a short instruction and
    overflowing the RoPE buffer.
    """
    w, h = image.size
    scale = (area / (w * h)) ** 0.5
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    if (new_w, new_h) != (w, h):
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


def resize_to_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    """Cap a PIL image's longest side at ``long_edge``, aspect preserved, never enlarged.

    Unlike :func:`resize_to_area` this fixes the dominant dimension, so an elongated
    image keeps its pixels-per-unit-width.
    """
    w, h = image.size
    scale = long_edge / max(w, h)
    if scale >= 1.0:
        return image
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


def align_down(value: int, multiple: int, minimum: int = 0) -> int:
    """Largest multiple of ``multiple`` not above ``value``, clamped to ``minimum``."""
    if multiple < 1:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return max(minimum, (int(value) // multiple) * multiple)


# The fitting rules ReferenceSizing accepts.
REF_FITS = ("area", "long_edge")


@dataclass(frozen=True)
class ReferenceSizing:
    """How an editing model fits one reference image before encoding it.

    Aspect is always preserved and nothing is ever cropped, so the references of a
    multi-reference edit keep the shape they arrived in. ``fit`` gives the single
    ``cap`` its meaning: a pixel **area** to scale to, or a **longest side** to cap
    without enlarging. The result is floored to ``align`` pixels, which keeps a
    reference on whole latent cells and inside the cap.
    """

    fit: str = "area"
    cap: int = 1024 * 1024
    align: int = 16

    def __post_init__(self) -> None:
        if self.fit not in REF_FITS:
            raise ValueError(
                f"unknown reference sizing fit {self.fit!r}; expected one of {REF_FITS}"
            )
        if self.align < 1:
            raise ValueError(f"align must be positive, got {self.align}")
        minimum = self.align * self.align if self.fit == "area" else self.align
        if self.cap < minimum:
            raise ValueError(
                f"{self.fit} cap of {self.cap} cannot hold one {self.align}px cell "
                f"(needs at least {minimum})"
            )

    def target_size(self, width: int, height: int) -> Tuple[int, int]:
        """The size :meth:`apply` produces for a ``width`` x ``height`` reference."""
        if width < 1 or height < 1:
            raise ValueError(f"invalid reference size {width}x{height}")
        if self.fit == "long_edge":
            scale = min(1.0, self.cap / max(width, height))
        else:
            scale = (self.cap / (width * height)) ** 0.5
        return (
            align_down(round(width * scale), self.align, self.align),
            align_down(round(height * scale), self.align, self.align),
        )

    def apply(self, image: Image.Image) -> Image.Image:
        """Fit ``image`` to :meth:`target_size`; the input itself when it matches."""
        size = self.target_size(image.width, image.height)
        if size == image.size:
            return image
        return image.resize(size, Image.LANCZOS)


__all__ = [
    "ALPHA_BACKGROUND",
    "REF_FITS",
    "ReferenceSizing",
    "align_down",
    "has_alpha",
    "flatten_alpha",
    "load_image",
    "pil_to_pixels",
    "pixels_to_pil",
    "resize_to_target",
    "center_crop",
    "resize_to_area",
    "resize_to_long_edge",
]
