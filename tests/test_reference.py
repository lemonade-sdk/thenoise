"""Flux2 reference-latent editing infrastructure tests (no weights / no GPU).

Covers the Flux2 reference helpers (``thenoise.dit.flux2.models``), the
``PipelineCache`` reference stage, and the edit path of the pipeline: size
derivation, per-image reference caching and the rejections a user can hit.
"""
from __future__ import annotations

import pytest
import torch
from PIL import Image

from conftest import EditingStubModel, StubModel
from thenoise.dit.flux2.models import concat_reference, slice_reference_output
from thenoise.models.config import EncodePromptArgs, GenerateRequest
from thenoise.pipeline import PipelineController
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.pipeline_cache import PipelineCache


def _edit_controller(model=None):
    return PipelineController(
        model or EditingStubModel(),
        PixelUpscalerManager(upscaler_dir="", device="cpu"),
    )


# ------------------------------------------------------------- reference helpers


def test_concat_reference_none_is_noop():
    img = torch.zeros(1, 4, 8)
    img_ids = torch.zeros(1, 4, 4, dtype=torch.long)
    out_img, out_ids = concat_reference(img, img_ids, None, None)
    assert out_img is img and out_ids is img_ids


def test_concat_and_slice_reference():
    img = torch.ones(1, 4, 8)
    img_ids = torch.zeros(1, 4, 4, dtype=torch.long)
    ref_tokens = torch.zeros(1, 2, 8)
    ref_ids = torch.ones(1, 2, 4, dtype=torch.long)

    cat_img, cat_ids = concat_reference(img, img_ids, ref_tokens, ref_ids)
    assert cat_img.shape == (1, 6, 8)
    assert cat_ids.shape == (1, 6, 4)
    # Reference ids are the trailing tokens.
    assert torch.all(cat_ids[0, 4:] == 1)
    assert torch.all(cat_ids[0, :4] == 0)

    # Slicing back drops the reference tokens -> original image tokens.
    out = slice_reference_output(cat_img, 4)
    assert out.shape == (1, 4, 8)
    assert torch.all(out == 1)


# ------------------------------------------------------------- cache cascade


def test_pipeline_cache_reference_cascade():
    c = PipelineCache()
    c.reference_store(("ref", 1), "R1")
    c.prompt_store(("p", 1), "C1")
    c.sampling_store(("s", 1), "S1")
    c.decode_store(("d", 1), "D1")

    # A reference miss clears all downstream stages.
    c.reference_store(("ref", 2), "R2")
    assert c.reference_hit(("ref", 2))
    assert c.prompt_key is None
    assert c.sampling_key is None
    assert c.decode_key is None


# ---------------------------------------------------------------- edit rejections


def test_edit_rejects_a_model_without_edit_support():
    """``edit()`` on a non-editing model must fail loudly, not generate."""
    controller = _edit_controller(StubModel())  # supports_edit is False
    assert not controller.model.supports_edit

    with pytest.raises(ValueError, match="does not support image editing"):
        controller.edit(GenerateRequest(prompt="p", image=Image.new("RGB", (8, 8))))


def test_edit_requires_an_input_image():
    controller = _edit_controller()
    with pytest.raises(ValueError, match="requires an input image"):
        controller.edit(GenerateRequest(prompt="p"))

    # An empty list is as good as no image at all.
    with pytest.raises(ValueError, match="requires an input image"):
        controller.edit(GenerateRequest(prompt="p", image=[]))


# ------------------------------------------------------------------ size handling


@pytest.mark.parametrize(
    "images,expected",
    [
        # 2:1 landscape -> largest side 1024, height 512.
        ([Image.new("RGB", (100, 50), "red")], (1024, 512)),
        # 1:2 portrait -> largest side (height) 1024.
        ([Image.new("RGB", (50, 100), "blue")], (512, 1024)),
        # Multi-ref: the FIRST image sets the aspect ratio, the rest are ignored.
        (
            [Image.new("RGB", (100, 50), "red"), Image.new("RGB", (50, 100), "blue")],
            (1024, 512),
        ),
    ],
    ids=["landscape", "portrait", "multi-ref-first-wins"],
)
def test_edit_derives_size_from_the_image_at_1024(images, expected):
    """No width/height -> the (first) image resized to 1024 on its largest side.

    The caller's request is left unmutated; the derived size is applied to a
    local copy only.
    """
    controller = _edit_controller()
    request = GenerateRequest(prompt="p", image=images[0] if len(images) == 1 else images, seed=1)
    controller.edit(request)

    assert controller.model.sizes == [expected]
    assert request.width is None and request.height is None


def test_edit_explicit_width_height_are_used_and_kept_on_the_request():
    controller = _edit_controller()
    request = GenerateRequest(
        prompt="p", image=Image.new("RGB", (100, 50), "red"), seed=1, width=128, height=64
    )
    controller.edit(request)

    assert controller.model.sizes == [(128, 64)]
    assert (request.width, request.height) == (128, 64)


# --------------------------------------------------------------- prompt plumbing


def test_encode_prompt_receives_the_args_struct():
    """The pipeline hands ``encode_prompt`` an ``EncodePromptArgs`` incl. the image."""
    controller = _edit_controller()
    img = Image.new("RGB", (64, 64), "red")
    controller.edit(
        GenerateRequest(prompt="p", negative_prompt="n", image=img, seed=1, guidance_scale=4.0)
    )

    received = controller.model.encode_prompt_args
    assert isinstance(received, EncodePromptArgs)
    assert (received.prompt, received.negative_prompt) == ("p", "n")
    assert received.guidance_scale == 4.0
    assert received.image is img


# ------------------------------------------------------------- reference caching


@pytest.mark.parametrize("as_list", [False, True], ids=["single", "one-element-list"])
def test_edit_reuses_the_reference_for_the_same_image(as_list):
    """A bare image and a one-element list are the same cache entry."""
    controller = _edit_controller()
    img = Image.new("RGB", (64, 64), "red")
    wrap = (lambda i: [i]) if as_list else (lambda i: i)

    controller.edit(GenerateRequest(prompt="p1", image=wrap(img), seed=1))
    assert controller.model.calls["encode_reference"] == 1

    # Same image, different prompt -> reference reused (no re-encode).
    controller.edit(GenerateRequest(prompt="p2", image=wrap(img), seed=2))
    assert controller.model.calls["encode_reference"] == 1

    # A different image is a different reference.
    controller.edit(GenerateRequest(prompt="p2", image=wrap(Image.new("RGB", (64, 64), "blue")), seed=2))
    assert controller.model.calls["encode_reference"] == 2


def test_multi_edit_encodes_each_reference_once_and_order_matters():
    controller = _edit_controller()
    img_a = Image.new("RGB", (64, 64), "red")
    img_b = Image.new("RGB", (64, 64), "blue")

    controller.edit(GenerateRequest(prompt="p1", image=[img_a, img_b], seed=1))
    assert controller.model.calls["encode_reference"] == 2

    # Same two images, different prompt -> refs reused.
    controller.edit(GenerateRequest(prompt="p2", image=[img_a, img_b], seed=2))
    assert controller.model.calls["encode_reference"] == 2

    # Reordering the refs changes the packed reference -> re-encode both.
    controller.edit(GenerateRequest(prompt="p3", image=[img_b, img_a], seed=3))
    assert controller.model.calls["encode_reference"] == 4
