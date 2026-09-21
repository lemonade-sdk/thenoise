"""PIL <-> tensor conversions, and the alpha boundaries they own.

The engine carries a ``[C, H, W]`` fp32 tensor whose channel count is the VAE's
(3 for the RGB family, 4 for the RGBA Qwen-Image 2.1 one), so these conversions
are where an alpha is either kept or deliberately composited away. The tests
below are all about that choice: a dropped alpha is invisible in a test that only
looks at RGB, but on a transparent PNG it is the difference between a cut-out and
a black silhouette.
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
)


def _rgba(rgba: list[list[tuple]]) -> Image.Image:
    """A tiny RGBA image from ``[[ (r,g,b,a), ... ], ...]`` row-major pixels."""
    img = Image.new("RGBA", (len(rgba[0]), len(rgba)))
    img.putdata([px for row in rgba for px in row])
    return img


# ------------------------------------------------------------------------ has_alpha


@pytest.mark.parametrize(
    "mode,expected",
    [("RGB", False), ("L", False), ("RGBA", True), ("LA", True)],
)
def test_has_alpha_follows_the_mode(mode, expected):
    assert has_alpha(Image.new(mode, (2, 2))) is expected


def test_has_alpha_needs_a_paletted_image_to_declare_transparency():
    opaque = Image.new("P", (2, 2))
    assert has_alpha(opaque) is False

    transparent = opaque.copy()
    transparent.info["transparency"] = 0
    assert has_alpha(transparent) is True


# --------------------------------------------------------------------- flatten_alpha


def test_flatten_alpha_composites_onto_white_rather_than_dropping():
    """``convert("RGB")`` would keep the transparent pixel's own RGB (here: red)."""
    img = _rgba([[(255, 0, 0, 0), (255, 0, 0, 255)]])

    flat = flatten_alpha(img)

    assert flat.mode == "RGB"
    # Fully transparent -> pure background; fully opaque -> the pixel itself.
    assert flat.getpixel((0, 0)) == ALPHA_BACKGROUND
    assert flat.getpixel((1, 0)) == (255, 0, 0)


def test_flatten_alpha_of_an_opaque_image_is_just_a_reformat():
    assert flatten_alpha(Image.new("RGB", (2, 2), (1, 2, 3))).getpixel((0, 0)) == (1, 2, 3)


# ------------------------------------------------------------------------ load_image


def test_load_image_keeps_transparency_and_normalises_the_format():
    buf = io.BytesIO()
    _rgba([[(0, 0, 255, 128)]]).save(buf, format="PNG")
    buf.seek(0)

    assert load_image(buf).mode == "RGBA"


def test_load_image_flattens_an_opaque_input_to_rgb():
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), "gray").save(buf, format="PNG")
    buf.seek(0)

    assert load_image(buf).mode == "RGB"


def test_load_image_resolves_a_transparent_palette_to_rgba(tmp_path):
    path = tmp_path / "indexed.png"
    img = Image.new("P", (1, 1), 0)
    img.info["transparency"] = 0
    img.save(path)

    opened = load_image(path)
    assert opened.mode == "RGBA"
    assert opened.getpixel((0, 0))[3] == 0


# -------------------------------------------------------------------- pil_to_pixels


def test_pil_to_pixels_defaults_to_rgb():
    pixels = pil_to_pixels(Image.new("RGB", (2, 2), (255, 128, 0)))
    assert pixels.shape == (3, 2, 2)
    assert pixels.dtype == torch.float32
    # 255 -> +1, 0 -> -1, 128 lands just above zero (the range is [-1, 1], not [0, 1]).
    assert torch.allclose(pixels[:, 0, 0], torch.tensor([1.0, 0.0039216, -1.0]), atol=1e-6)


def test_pil_to_pixels_pads_an_opaque_alpha_for_an_rgba_destination():
    pixels = pil_to_pixels(Image.new("RGB", (2, 2), "black"), 4)
    assert pixels.shape == (4, 2, 2)
    assert torch.equal(pixels[3], torch.ones(2, 2))  # alpha 1.0 == opaque


def test_pil_to_pixels_keeps_the_alpha_of_an_rgba_input():
    pixels = pil_to_pixels(_rgba([[(0, 0, 0, 0), (0, 0, 0, 255)]]), 4)
    assert torch.allclose(pixels[3], torch.tensor([[-1.0, 1.0]]))


def test_pil_to_pixels_composites_for_an_rgb_destination():
    """An RGBA input entering an RGB VAE must not reach it as raw RGB."""
    pixels = pil_to_pixels(_rgba([[(255, 0, 0, 0)]]), 3)
    assert pixels.shape == (3, 1, 1)
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


def test_pixels_to_pil_rejects_a_channel_count_pil_cannot_name():
    with pytest.raises(ValueError, match="cannot build a PIL image"):
        pixels_to_pil(torch.zeros(5, 2, 2))


def test_rgba_pixels_survive_the_round_trip():
    """What an RGBA VAE decoded is what the PNG writer is handed."""
    img = _rgba([[(10, 200, 30, 40), (250, 5, 5, 255)], [(0, 0, 0, 0), (255, 255, 255, 128)]])

    out = pixels_to_pil(pil_to_pixels(img, 4))

    assert out.mode == "RGBA"
    # 8-bit -> [-1,1] -> 8-bit is lossy by one quantization step at worst (the
    # conversion truncates), which is the same for alpha as for the colours.
    assert torch.allclose(
        pil_to_pixels(out, 4), pil_to_pixels(img, 4), atol=1.5 / 127.5
    )


def test_opaque_pixels_do_not_grow_an_alpha_on_the_way_out():
    """An RGB VAE's output stays RGB: no alpha is invented on the way to the PNG."""
    assert pixels_to_pil(pil_to_pixels(Image.new("RGB", (4, 4), "gray"), 3)).mode == "RGB"
