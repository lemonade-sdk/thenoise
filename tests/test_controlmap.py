"""Control-map plumbing: the request options, the pool, the controller boundary.
Weight-free and CPU-only: canny stands in for the pool and needs no checkpoints.
"""
from __future__ import annotations

import io
import json
from dataclasses import fields

import pytest
import torch
from PIL import Image, ImageDraw

from thenoise.controlmap.base import ControlMapOptions
from thenoise.controlmap.canny import CannyOptions, CannyControlMap
from thenoise.controlmap.manager import ControlMapManager
from thenoise.controlmap_controller import ControlMapController


@pytest.fixture
def square():
    image = Image.new("RGB", (32, 16), "black")
    ImageDraw.Draw(image).rectangle([8, 4, 24, 12], fill="white")
    return image


@pytest.fixture
def controller():
    return ControlMapController(ControlMapManager(device="cpu"))


def chunk(image, keyword):
    """The ``keyword`` PNG text chunk of an image as saved by the controller."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", pnginfo=image._pnginfo)
    buf.seek(0)
    return Image.open(buf).info.get(keyword)


# ---------------------------------------------------------------- the pool


def test_pool_offers_the_registered_kinds():
    assert ControlMapManager(device="cpu").available() == {
        "canny": ["canny"],
        "lineart": ["lineart"],
        "softedge": ["softedge"],
    }


def test_the_first_implementation_of_a_kind_is_its_default():
    """What ``get()`` falls back to, and what ``/controlmaps`` reports as such."""
    assert ControlMapManager(device="cpu").defaults() == {
        "canny": "canny",
        "lineart": "lineart",
        "softedge": "softedge",
    }


def test_a_kind_the_engine_has_never_heard_of_names_the_available_ones():
    with pytest.raises(
        ValueError, match=r"no control map for 'normals'; available: canny, lineart, softedge"
    ):
        ControlMapManager(device="cpu").get("normals")


def test_a_kind_missing_its_weights_names_the_download_command():
    """A kind the engine does have, but whose checkpoint does not, is a different
    answer: the message has to say what to run."""
    with pytest.raises(ValueError, match=r"the depth control map has no checkpoint"):
        ControlMapManager(device="cpu").get("depth")


def test_unknown_name_names_the_kind_implementations():
    with pytest.raises(ValueError, match="no canny control map named 'canny-xl'"):
        ControlMapManager(device="cpu").get("canny", "canny-xl")


def test_the_resident_controlmap_is_reused():
    manager = ControlMapManager(device="cpu")
    assert manager.get("canny") is manager.get("canny")


def test_unload_forces_a_fresh_instance():
    manager = ControlMapManager(device="cpu")
    first = manager.get("canny")
    manager.unload()
    assert manager.get("canny") is not first


def test_unload_without_anything_resident_is_a_no_op():
    ControlMapManager(device="cpu").unload()


def test_options_are_built_from_the_kinds_own_dataclass():
    manager = ControlMapManager(device="cpu")
    assert manager.options_for("canny", low=30) == CannyOptions(low=30)


def test_a_request_carries_no_geometry():
    """A map is always its image's size, so a kind with nothing to ask has no fields."""
    assert fields(ControlMapOptions) == ()


# ---------------------------------------------------------------- controller


def test_run_paints_a_greyscale_map_at_the_input_size(controller, square):
    out = controller.run(square, "canny", CannyOptions())
    assert (out.mode, out.size) == ("L", square.size)


def test_run_records_the_request_in_a_png_chunk(controller, square):
    out = controller.run(square, "canny", CannyOptions(low=30, high=120))
    assert json.loads(chunk(out, "controlmap_data")) == {
        "type": "canny",
        "model": "canny",
        "options": {"low": 30, "high": 120},
    }


def test_run_records_only_the_options_that_changed(controller, square):
    """Defaults are the boring case, so the chunk states just the intent."""
    out = controller.run(square, "canny", CannyOptions())
    assert json.loads(chunk(out, "controlmap_data"))["options"] == {}


def test_run_carries_over_the_source_metadata(controller, square):
    square.info["generation_data"] = '{"seed": 7}'
    out = controller.run(square, "canny", CannyOptions())
    assert chunk(out, "generation_data") == '{"seed": 7}'


def test_a_run_leaves_nothing_resident(controller, square):
    """Extraction is one-shot: the room its weights took belongs to whatever runs next."""
    used = controller._controlmaps.get("canny")
    controller.run(square, "canny", CannyOptions())
    assert controller._controlmaps.get("canny") is not used


def test_run_reaches_the_kernels_inside_the_inference_boundary(controller, square, monkeypatch):
    seen = {}
    detect = CannyControlMap.detect

    def spy(self, image, options):
        found = detect(self, image, options)
        seen["inference"] = bool(torch.is_inference(found))
        return found

    monkeypatch.setattr(CannyControlMap, "detect", spy)
    controller.run(square, "canny", CannyOptions())
    assert seen["inference"]


def test_transparent_pixels_are_composited_before_measuring(controller):
    # Composited onto white the image is blank, so edges here would mean the alpha
    # was dropped and the raw RGB read instead.
    image = Image.new("RGBA", (32, 16), (255, 0, 0, 0))
    ImageDraw.Draw(image).rectangle([8, 4, 24, 12], fill=(255, 255, 255, 255))
    assert set(controller.run(image, "canny", CannyOptions()).get_flattened_data()) == {0}
