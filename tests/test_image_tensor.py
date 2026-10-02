"""PIL <-> tensor conversions, and the alpha boundaries they own.

The engine carries a ``[C, H, W]`` fp32 tensor whose channel count is the VAE's, so
these conversions are where an alpha is either kept or deliberately composited away.
"""
from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from thenoise.utils.image_tensor import (
    ALPHA_BACKGROUND,
    flatten_alpha,
    has_alpha,
    load_image,
    pil_to_pixels,
    pixels_to_pil,
    resize_to_long_edge,
)


def _rgba(rgba: list[list[tuple]]) -> Image.Image:
    """A tiny RGBA image from ``[[ (r,g,b,a), ... ], ...]`` row-major pixels."""
    img = Image.new("RGBA", (len(rgba[0]), len(rgba)))
    img.putdata([px for row in rgba for px in row])
    return img


# ------------------------------------------------------------------------ has_alpha


def test_has_alpha_follows_the_mode():
    assert has_alpha(Image.new("RGB", (2, 2))) is False
    assert has_alpha(Image.new("L", (2, 2))) is False
    assert has_alpha(Image.new("RGBA", (2, 2))) is True
    assert has_alpha(Image.new("LA", (2, 2))) is True

    # A paletted image only has alpha if it declares one.
    paletted = Image.new("P", (2, 2))
    assert has_alpha(paletted) is False
    transparent = paletted.copy()
    transparent.info["transparency"] = 0
    assert has_alpha(transparent) is True


# --------------------------------------------------------------------- flatten_alpha


def test_flatten_alpha_composites_onto_white_rather_than_dropping():
    """``convert("RGB")`` would keep the transparent pixel's own RGB (here: red)."""
    flat = flatten_alpha(_rgba([[(255, 0, 0, 0), (255, 0, 0, 255), (1, 2, 3, 255)]]))

    assert flat.mode == "RGB"
    # Fully transparent -> pure background; fully opaque -> the pixel itself.
    assert flat.getpixel((0, 0)) == ALPHA_BACKGROUND
    assert flat.getpixel((1, 0)) == (255, 0, 0)
    assert flat.getpixel((2, 0)) == (1, 2, 3)  # an already opaque image is a reformat


# ------------------------------------------------------------------------ load_image


def test_load_image_normalises_the_format_to_the_presence_of_alpha():
    buf = io.BytesIO()
    _rgba([[(0, 0, 255, 128)]]).save(buf, format="PNG")
    assert load_image(io.BytesIO(buf.getvalue())).mode == "RGBA"

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), "gray").save(buf, format="PNG")
    assert load_image(io.BytesIO(buf.getvalue())).mode == "RGB"


def test_load_image_resolves_a_transparent_palette_to_rgba(tmp_path):
    path = tmp_path / "indexed.png"
    img = Image.new("P", (1, 1), 0)
    img.info["transparency"] = 0
    img.save(path)

    assert load_image(path).getpixel((0, 0))[3] == 0


# -------------------------------------------------------------------- pil_to_pixels


def test_pil_to_pixels_maps_the_range_and_pads_alpha_on_demand():
    pixels = pil_to_pixels(Image.new("RGB", (2, 2), (255, 128, 0)))
    assert pixels.shape == (3, 2, 2)
    assert pixels.dtype == torch.float32
    # 255 -> +1, 0 -> -1, 128 lands just above zero (the range is [-1, 1], not [0, 1]).
    assert torch.allclose(pixels[:, 0, 0], torch.tensor([1.0, 0.0039216, -1.0]), atol=1e-6)

    # An opaque input entering an RGBA destination gets a fully opaque alpha.
    assert torch.equal(pil_to_pixels(Image.new("RGB", (2, 2), "black"), 4)[3], torch.ones(2, 2))


def test_pil_to_pixels_keeps_the_alpha_of_an_rgba_input():
    pixels = pil_to_pixels(_rgba([[(0, 0, 0, 0), (0, 0, 0, 255)]]), 4)
    assert torch.allclose(pixels[3], torch.tensor([[-1.0, 1.0]]))


def test_pil_to_pixels_composites_for_an_rgb_destination():
    """An RGBA input entering an RGB VAE must not reach it as raw RGB."""
    pixels = pil_to_pixels(_rgba([[(255, 0, 0, 0)]]), 3)
    assert torch.allclose(pixels[:, 0, 0], torch.tensor([1.0, 1.0, 1.0]))  # white


def test_pil_to_pixels_auto_follows_the_input():
    """``channels=None`` means "lose nothing": keep whatever came in."""
    assert pil_to_pixels(Image.new("RGB", (2, 2)), None).shape[0] == 3
    assert pil_to_pixels(_rgba([[(0, 0, 0, 0)]]), None).shape[0] == 4


def test_pil_to_pixels_rejects_a_channel_count_no_vae_uses():
    with pytest.raises(ValueError, match="3 or 4 channels"):
        pil_to_pixels(Image.new("RGB", (2, 2)), 2)


# -------------------------------------------------------------------- pixels_to_pil


def test_pixels_to_pil_mirrors_the_channel_count():
    assert pixels_to_pil(torch.zeros(3, 2, 2)).mode == "RGB"
    assert pixels_to_pil(torch.zeros(4, 2, 2)).mode == "RGBA"
    with pytest.raises(ValueError, match="cannot build a PIL image"):
        pixels_to_pil(torch.zeros(5, 2, 2))


def test_rgba_pixels_survive_the_round_trip():
    """What an RGBA VAE decoded is what the PNG writer is handed."""
    img = _rgba([[(10, 200, 30, 40), (250, 5, 5, 255)], [(0, 0, 0, 0), (255, 255, 255, 128)]])

    out = pixels_to_pil(pil_to_pixels(img, 4))

    assert out.mode == "RGBA"
    # 8-bit -> [-1,1] -> 8-bit is lossy by one quantization step at worst.
    assert torch.allclose(
        pil_to_pixels(out, 4), pil_to_pixels(img, 4), atol=1.5 / 127.5
    )


def test_opaque_pixels_do_not_grow_an_alpha_on_the_way_out():
    """An RGB VAE's output stays RGB: no alpha is invented on the way to the PNG."""
    assert pixels_to_pil(pil_to_pixels(Image.new("RGB", (4, 4), "gray"), 3)).mode == "RGB"


# ---------------------------------------------------------------- vision-token cap


def test_long_edge_cap_keeps_the_dominant_dimension():
    """Where ``resize_to_area`` fixes the area, this fixes the long side, and never
    enlarges: growing an image adds vision tokens but no information.
    """
    assert resize_to_long_edge(Image.new("RGB", (1600, 400)), 384).size == (384, 96)

    small = Image.new("RGB", (300, 300))
    assert resize_to_long_edge(small, 384) is small
    assert resize_to_long_edge(Image.new("RGB", (384, 20)), 384).size == (384, 20)
