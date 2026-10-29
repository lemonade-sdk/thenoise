"""The pure-torch lineart: thin dark lines on a white canvas, from the same Sobel +
NMS pipeline as Canny but with a single threshold and the inverted palette.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from thenoise.controlmap.lineart import LineartOptions, LineartControlMap


@pytest.fixture
def lineart():
    controlmap = LineartControlMap()
    controlmap.load("cpu")
    return controlmap


def map_of(lineart, image, **options) -> np.ndarray:
    rendered = lineart.run(image, LineartOptions(**options))
    array = np.asarray(rendered)
    assert rendered.mode == "L" and array.ndim == 2
    return array


def rectangle(size, box, fill="white", background="black") -> Image.Image:
    image = Image.new("RGB", size, background)
    ImageDraw.Draw(image).rectangle(list(box), fill=fill)
    return image


def test_uniform_image_has_no_lines(lineart):
    # The whole canvas stays white: a flat image has no gradient.
    assert map_of(lineart, Image.new("RGB", (48, 48), "grey")).min() == 255


def test_a_step_edge_is_one_pixel_wide(lineart):
    image = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(image).rectangle([32, 0, 63, 63], fill="white")
    row = map_of(lineart, image)[32]
    assert np.nonzero(row < 255)[0].tolist() == [32]  # the dark line, not the white canvas


def test_lines_are_dark_on_a_white_canvas(lineart):
    """The palette is the inverse of Canny: background 255, the outline 0."""
    array = map_of(lineart, rectangle((32, 16), (8, 4, 24, 12)))
    assert array[4, 12] == 0  # on the top border
    assert array[0, 0] == 255  # background
    assert set(array.flatten().tolist()) <= {0, 255}


def test_detect_returns_a_binary_mask(lineart):
    found = lineart.detect(rectangle((32, 16), (8, 4, 24, 12)), LineartOptions())
    assert set(found.flatten().tolist()) <= {0.0, 1.0}
    assert found.shape == (1, 16, 32)


def test_a_higher_threshold_keeps_fewer_lines(lineart):
    """The threshold is on the per-image normalised magnitude, so a strong (white)
    contour always stays while a faint (grey) one fades out as the threshold rises."""
    image = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(image).rectangle([4, 4, 20, 20], fill="white")
    ImageDraw.Draw(image).rectangle([36, 36, 52, 52], fill=(80, 80, 80))
    assert (map_of(lineart, image, threshold=0.2) < 255).sum() > (
        map_of(lineart, image, threshold=0.9) < 255
    ).sum()


def test_thickness_thickens_without_losing_lines(lineart):
    image = rectangle((48, 48), (12, 12, 36, 36))
    thin, thick = map_of(lineart, image), map_of(lineart, image, thickness=3)
    assert (thick < 255).sum() > (thin < 255).sum()
    assert np.all(thick[thin < 255] < 255)


def test_options_are_validated():
    with pytest.raises(ValueError, match="threshold must be in"):
        LineartOptions(threshold=1.5)
    with pytest.raises(ValueError, match="thickness must be >= 1"):
        LineartOptions(thickness=0)
