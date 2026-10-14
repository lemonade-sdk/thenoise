"""The pure-torch softedge: continuous, anti-aliased dark edges on a white canvas, so a
soft edge keeps its grey gradient rather than being binarised like Canny.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from thenoise.controlmap.softedge import SoftedgeOptions, SoftedgeControlMap


@pytest.fixture
def softedge():
    controlmap = SoftedgeControlMap()
    controlmap.load("cpu")
    return controlmap


def map_of(softedge, image, **options) -> np.ndarray:
    rendered = softedge.run(image, SoftedgeOptions(**options))
    array = np.asarray(rendered)
    assert rendered.mode == "L" and array.ndim == 2
    return array


def rectangle(size, box, fill="white", background="black") -> Image.Image:
    image = Image.new("RGB", size, background)
    ImageDraw.Draw(image).rectangle(list(box), fill=fill)
    return image


def test_uniform_image_has_no_edges(softedge):
    # The whole canvas stays white: a flat image has no gradient.
    assert map_of(softedge, Image.new("RGB", (48, 48), "grey")).max() == 0


def test_edges_are_soft_not_binary(softedge):
    """A step edge fades through grey rather than snapping to 0/255, and the canvas is
    white where there is no edge."""
    array = map_of(softedge, rectangle((32, 16), (8, 4, 24, 12)))
    assert array.max() == 255  # background
    assert array.min() < 255  # something dark
    grey = array[(array > 0) & (array < 255)]
    assert grey.size > 0  # the transition is continuous, not a single hard line


def test_detect_returns_a_continuous_field(softedge):
    found = softedge.detect(rectangle((32, 16), (8, 4, 24, 12)), SoftedgeOptions())
    assert found.shape == (1, 16, 32)
    assert found.min() >= 0.0 and found.max() <= 1.0


def test_edges_stay_on_the_steps(softedge):
    """Without blur the soft edge sits on the step; blur is what spreads it."""
    box = (8, 4, 24, 12)
    rows, columns = np.nonzero(map_of(softedge, rectangle((32, 16), box), blur=0) < 255)
    on_a_border = np.isclose(rows[:, None], [box[1], box[3]], atol=2).any(1)
    on_a_side = np.isclose(columns[:, None], [box[0], box[2]], atol=2).any(1)
    assert (on_a_border | on_a_side).all()


def test_blur_spreads_the_edge(softedge):
    image = rectangle((48, 48), (12, 12, 36, 36))
    crisp, soft = map_of(softedge, image, blur=0), map_of(softedge, image, blur=3.0)
    assert (soft < 255).sum() > (crisp < 255).sum()


def test_a_higher_threshold_keeps_fewer_edges(softedge):
    """The threshold is on the per-image normalised magnitude, so a strong (white)
    contour always stays while a faint (grey) one fades out as the threshold rises."""
    image = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(image).rectangle([4, 4, 20, 20], fill="white")
    ImageDraw.Draw(image).rectangle([36, 36, 52, 52], fill=(80, 80, 80))
    assert (map_of(softedge, image, threshold=0.2) < 255).sum() > (
        map_of(softedge, image, threshold=0.9) < 255
    ).sum()


def test_options_are_validated():
    with pytest.raises(ValueError, match="threshold must be in"):
        SoftedgeOptions(threshold=1.5)
    with pytest.raises(ValueError, match="blur must be >= 0"):
        SoftedgeOptions(blur=-1)
