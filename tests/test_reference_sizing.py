"""The reference-image sizing policy: what each editing model declares, and how the
pipeline applies it. By default a reference keeps the shape it arrived in and only the
*output* size comes from the request; a model whose edit lines up with its reference
overrides ``fit_references`` to lay that reference over the canvas instead.
"""
from __future__ import annotations

from PIL import Image

from conftest import EditingStubModel
from thenoise.models import (
    FluxKleinModel,
    MageFlowModel,
    QwenImage21Model,
    QwenImageModel,
)
from thenoise.models.base import DiffusionModel
from thenoise.models.config import GenerateRequest
from thenoise.pipeline import PipelineController
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.image_tensor import ReferenceSizing, cover_crop


class _ToySizedEditingStub(EditingStubModel):
    """The editing stub with a toy reference budget: the shared policy, small numbers."""

    REFERENCE_SIZING = ReferenceSizing(cap=64 * 64, align=16)


class _CanvasFittedEditingStub(EditingStubModel):
    """The editing stub of an aligned-edit model: its lone reference covers the canvas."""

    def fit_references(self, images, target):
        if len(images) == 1:
            return [cover_crop(images[0], *target)]
        return super().fit_references(images, target)


def _controller(model) -> PipelineController:
    manager = PixelUpscalerManager(upscaler_dir="", device="cpu")
    manager._pixel_upscaler_scales = {}
    return PipelineController(model, manager)


def _request(**kwargs) -> GenerateRequest:
    return GenerateRequest(**{"prompt": "a fox", "seed": 1, "steps": 1, **kwargs})


def _refs(*sizes) -> list[Image.Image]:
    return [Image.new("RGB", size, "gray") for size in sizes]


def test_each_editing_model_declares_the_rule_upstream_scales_by():
    """One ``cap`` per model whose meaning its ``fit`` decides, aligned to its own
    latent cell (32 px on Qwen-Image 2.1, whose reference latent replaces whole
    32x32 vision tokens)."""
    assert QwenImageModel.REFERENCE_SIZING == ReferenceSizing(cap=1024 * 1024, align=16)
    assert FluxKleinModel.REFERENCE_SIZING == ReferenceSizing(cap=1024 * 1024, align=16)
    assert QwenImage21Model.REFERENCE_SIZING == ReferenceSizing(cap=1024 * 1024, align=32)
    assert MageFlowModel.REFERENCE_SIZING == ReferenceSizing(
        fit="long_edge", cap=1024, align=16
    )


def test_references_keep_their_own_aspect_ratio():
    model = _ToySizedEditingStub()

    _controller(model).edit(_request(image=_refs((200, 100), (100, 200)), width=64, height=64))

    # One latent per reference, in request order, each at its own shape.
    assert [tuple(ref.shape[-2:]) for ref in model.prepared[0]["ref"]] == [
        (32, 80),
        (80, 32),
    ]


def test_the_reference_size_does_not_follow_the_output_size():
    """The default ``fit_references`` ignores the canvas it is handed."""
    shapes = []
    for size in (32, 128):
        model = _ToySizedEditingStub()
        _controller(model).edit(_request(image=_refs((200, 100)), width=size, height=size))
        shapes.append(tuple(model.prepared[0]["ref"][0].shape))

    assert shapes == [(1, 3, 32, 80), (1, 3, 32, 80)]


def test_the_first_reference_still_picks_the_output_size():
    """A size decision only: the other references neither steer the output nor are
    resized to it."""
    model = _ToySizedEditingStub()

    _controller(model).edit(_request(image=_refs((200, 100), (300, 300))))

    assert model.sizes[0] == model.default_edit_size(200, 100)
    assert tuple(model.prepared[0]["ref"][1].shape[-2:]) == (64, 64)


def test_an_unnamed_edit_size_is_the_shared_area_at_the_reference_aspect():
    model = EditingStubModel()

    for size in ((200, 100), (100, 200), (4000, 3000), (900, 900)):
        width, height = model.default_edit_size(*size)
        assert abs((width / height) / (size[0] / size[1]) - 1) < 0.01
        assert abs(width * height / DiffusionModel.EDIT_DEFAULT_AREA - 1) < 0.01


def test_a_reference_is_no_longer_cropped_to_the_output():
    """The old behaviour threw away everything outside the output aspect ratio."""
    model = _ToySizedEditingStub()

    _controller(model).edit(_request(image=_refs((400, 100)), width=64, height=64))

    assert tuple(model.prepared[0]["ref"][0].shape[-2:]) == (32, 128)


def test_a_model_can_still_lay_its_reference_over_the_canvas():
    """``fit_references`` is the opt-in for a model whose edit lines up with the
    result: here the reference shares the canvas grid, crop and all."""
    model = _CanvasFittedEditingStub()

    _controller(model).edit(_request(image=_refs((400, 100)), width=64, height=64))

    assert tuple(model.prepared[0]["ref"][0].shape[-2:]) == (64, 64)


def test_the_reference_cache_key_follows_the_policy_and_the_canvas():
    """Either input to ``fit_references`` decides the stored latents, so either one
    changing has to miss the cache — a policy, or the size a policy is given."""
    image = Image.new("RGB", (200, 100), "gray")

    plain = _controller(EditingStubModel())._cache_key_reference([image], (64, 64))
    toy = _controller(_ToySizedEditingStub())._cache_key_reference([image], (64, 64))

    assert plain != toy
    assert plain != _controller(EditingStubModel())._cache_key_reference([image], (128, 64))
    assert plain == _controller(EditingStubModel())._cache_key_reference([image], (64, 64))
