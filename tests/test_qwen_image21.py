"""Qwen-Image 2.1 adapter specifics that need no weights.

The architecture lives in :mod:`thenoise.dit.qwen_image21`; what is tested here is
the adapter's side of the RGBA contract — the one place in it where a transparency
has to disappear (the text encoder's vision tokens are RGB) and the fact that it
no longer narrows its own VAE output to RGB.
"""
from __future__ import annotations

import torch
from PIL import Image

from thenoise.models import QwenImage21Model
from thenoise.models.base import DiffusionModel
from thenoise.models.config import EncodePromptArgs


def _bare(**attrs):
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(QwenImage21Model)
    model.device = "cpu"
    model.dtype = torch.float32
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def test_the_vision_tokens_get_an_rgb_composite_not_a_dropped_alpha():
    """A transparent reference must not reach Qwen3-VL as its raw RGB.

    ``convert("RGB")`` deletes the alpha channel and keeps the colour underneath,
    which for a cut-out is whatever the exporter left there — often black. The
    encoder should see the cut-out on the same white background the pipeline
    composites it onto everywhere else.
    """
    transparent_red = Image.new("RGBA", (32, 32), (255, 0, 0, 0))

    images = _bare()._encoder_images(
        EncodePromptArgs(prompt="p", image=transparent_red, width=32, height=32)
    )

    assert [img.mode for img in images] == ["RGB"]
    assert images[0].getpixel((0, 0)) == (255, 255, 255)


def test_every_reference_in_a_list_is_composited():
    images = _bare()._encoder_images(
        EncodePromptArgs(
            prompt="p",
            image=[Image.new("RGBA", (32, 32), (0, 0, 255, 0)), Image.new("RGB", (32, 32), "red")],
            width=32,
            height=32,
        )
    )

    assert images[0].getpixel((0, 0)) == (255, 255, 255)  # transparent -> white
    assert images[1].getpixel((0, 0)) == (255, 0, 0)  # opaque -> untouched


def test_the_adapter_does_not_override_the_shared_decode():
    """The alpha drops-out used to be a ``decode`` override; it must not come back.

    ``DiffusionModel.decode`` hands back everything the VAE returned, which is what
    makes an RGBA model's output transparent instead of merely RGB.
    """
    assert "decode" not in QwenImage21Model.__dict__
    assert QwenImage21Model.decode is DiffusionModel.decode
