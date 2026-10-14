"""The pure-torch Canny, checked on synthetic images whose edges are known beforehand:
edges sit on the steps, NMS keeps them one pixel wide, hysteresis alone decides weak ones.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from thenoise.controlmap.canny import CannyOptions, CannyControlMap, luminance
from thenoise.utils.image_tensor import pil_to_pixels


@pytest.fixture
def canny():
    controlmap = CannyControlMap()
    controlmap.load("cpu")
    return controlmap


def map_of(canny, image, **options) -> np.ndarray:
    """The rendered map as a ``[H, W]`` numpy array of 0/255."""
    rendered = canny.run(image, CannyOptions(**options))
    array = np.asarray(rendered)
    assert rendered.mode == "L" and array.ndim == 2
    return array


def rectangle(size, box, fill="white", background="black") -> Image.Image:
    image = Image.new("RGB", size, background)
    ImageDraw.Draw(image).rectangle(list(box), fill=fill)
    return image


def test_uniform_image_has_no_edges(canny):
    assert map_of(canny, Image.new("RGB", (48, 48), "grey")).max() == 0


def test_a_step_edge_is_one_pixel_wide(canny):
    image = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(image).rectangle([32, 0, 63, 63], fill="white")

    row = map_of(canny, image)[32]
    assert np.nonzero(row)[0].tolist() == [32]  # NMS, not a 3px smear


def test_edges_stay_on_the_steps(canny):
    box = (8, 4, 24, 12)
    rows, columns = np.nonzero(map_of(canny, rectangle((32, 16), box)))
    on_a_border = np.isclose(rows[:, None], [box[1], box[3]], atol=1).any(1)
    on_a_side = np.isclose(columns[:, None], [box[0], box[2]], atol=1).any(1)
    assert (on_a_border | on_a_side).all()
    assert len(rows) >= 2 * ((box[2] - box[0]) + (box[3] - box[1]))  # the whole outline


def test_detect_returns_a_binary_mask(canny):
    found = canny.detect(rectangle((32, 16), (8, 4, 24, 12)), CannyOptions())
    assert set(found.flatten().tolist()) <= {0.0, 1.0}


def test_detect_measures_without_resampling_the_raster(canny):
    """Every stage keeps the geometry, so the map lines up with the image pixel for
    pixel — which is the whole point of pairing them later."""
    assert canny.detect(rectangle((32, 16), (8, 4, 24, 12)), CannyOptions()).shape == (1, 16, 32)


def test_a_faint_edge_needs_faint_thresholds(canny):
    image = rectangle((48, 48), (12, 12, 36, 36), fill=(60, 60, 60))
    assert map_of(canny, image).max() == 0
    assert map_of(canny, image, low=10, high=20).max() == 255


def test_higher_thresholds_keep_fewer_edges(canny):
    image = rectangle((48, 48), (12, 12, 36, 36), fill=(120, 120, 120))
    assert (map_of(canny, image, low=20, high=40) > 0).sum() > (
        map_of(canny, image, low=300, high=600) > 0
    ).sum()


# A black-to-white step measures ~600 Sobel units here, black-to-grey(80) ~190, so the
# faint contour has no strong pixel to grow from and hysteresis alone decides it.
STRONG = (4, 4, 20, 20)
FAINT = (36, 36, 52, 52)


@pytest.fixture
def strong_and_faint():
    image = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(image).rectangle(list(STRONG), fill="white")
    ImageDraw.Draw(image).rectangle(list(FAINT), fill=(80, 80, 80))
    return image


def in_box(array, box) -> np.ndarray:
    rows, columns = np.mgrid[: array.shape[0], : array.shape[1]]
    inside = (rows >= box[1]) & (rows <= box[3]) & (columns >= box[0]) & (columns <= box[2])
    return array[inside]


def test_a_weak_contour_is_dropped(strong_and_faint, canny):
    array = map_of(canny, strong_and_faint, low=100, high=300)
    assert in_box(array, STRONG).max() == 255
    assert in_box(array, FAINT).max() == 0


def test_the_same_contour_is_kept_once_it_is_strong_enough(strong_and_faint, canny):
    array = map_of(canny, strong_and_faint, low=100, high=150)
    assert in_box(array, STRONG).max() == 255
    assert in_box(array, FAINT).max() == 255


def test_thickness_thickens_without_losing_edges(canny):
    image = rectangle((48, 48), (12, 12, 36, 36))
    thin, thick = map_of(canny, image), map_of(canny, image, thickness=3)
    assert (thick > 0).sum() > (thin > 0).sum()
    assert np.all(thick[thin > 0] > 0)


@pytest.mark.parametrize(
    "colour, expected", [((255, 0, 0), 0.2126), ((0, 255, 0), 0.7152), ((0, 0, 255), 0.0722)]
)
def test_luminance_uses_rec_709(canny, colour, expected):
    rgb = pil_to_pixels(Image.new("RGB", (2, 2), colour), 3)
    assert luminance(rgb).max().item() == pytest.approx(expected, abs=1e-4)


def test_thresholds_are_validated():
    with pytest.raises(ValueError, match="exceeds high threshold"):
        CannyOptions(low=200, high=100)
    with pytest.raises(ValueError, match="thickness must be >= 1"):
        CannyOptions(thickness=0)
    with pytest.raises(ValueError, match="low must be >= 0"):
        CannyOptions(low=-1)
