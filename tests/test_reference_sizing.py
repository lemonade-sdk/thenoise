"""The reference-image sizing policy: what each editing model declares, and how the
pipeline applies it. A reference keeps the shape it arrived in; only the *output* size
comes from the request.
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
from thenoise.models.config import GenerateRequest
from thenoise.pipeline import PipelineController
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.image_tensor import ReferenceSizing


class _ToySizedEditingStub(EditingStubModel):
    """The editing stub with a toy reference budget: the shared policy, small numbers."""

    REFERENCE_SIZING = ReferenceSizing(cap=64 * 64, align=16)


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

    assert model.sizes[0] == (1024, 512)  # long edge capped, aspect of reference #1
    assert tuple(model.prepared[0]["ref"][1].shape[-2:]) == (64, 64)


def test_a_reference_is_no_longer_cropped_to_the_output():
    """The old behaviour threw away everything outside the output aspect ratio."""
    model = _ToySizedEditingStub()

    _controller(model).edit(_request(image=_refs((400, 100)), width=64, height=64))

    assert tuple(model.prepared[0]["ref"][0].shape[-2:]) == (32, 128)


def test_the_reference_cache_key_follows_the_policy_not_the_output_size():
    image = Image.new("RGB", (200, 100), "gray")

    plain = _controller(EditingStubModel())._cache_key_reference([image])
    toy = _controller(_ToySizedEditingStub())._cache_key_reference([image])

    assert plain != toy
    assert plain == _controller(EditingStubModel())._cache_key_reference([image])
