"""Qwen-Image 2.1 adapter specifics that need no weights."""
from __future__ import annotations

import torch
from PIL import Image

from thenoise.models import QwenImage21Model
from thenoise.models.base import DiffusionModel
from thenoise.models.config import EncodePromptArgs
from thenoise.upscale import LatentTranscodeNet, Qwen21TranscodeUpscaler
from thenoise.upscale import qwen21_transcode


class FakeVAE:
    """Just the surface the transcoder is allowed to reach for."""

    z_dim = 4

    def decode_features(self, latents, *, blocks=1):
        b, _, h, w = latents.shape
        return torch.zeros(b, 6, 2 * h, 2 * w, dtype=latents.dtype)


def _bare(**attrs):
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(QwenImage21Model)
    model.device = "cpu"
    model.dtype = torch.float32
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def test_the_vision_tokens_get_an_rgb_composite_not_a_dropped_alpha():
    """A transparent reference must not reach Qwen3-VL as its raw RGB."""
    transparent_red = Image.new("RGBA", (32, 32), (255, 0, 0, 0))

    images = _bare()._encoder_images(
        EncodePromptArgs(prompt="p", image=transparent_red)
    )

    assert [img.mode for img in images] == ["RGB"]
    assert images[0].getpixel((0, 0)) == (255, 255, 255)


def test_every_reference_in_a_list_is_composited():
    images = _bare()._encoder_images(
        EncodePromptArgs(
            prompt="p",
            image=[Image.new("RGBA", (32, 32), (0, 0, 255, 0)), Image.new("RGB", (32, 32), "red")],
        )
    )

    assert images[0].getpixel((0, 0)) == (255, 255, 255)  # transparent -> white
    assert images[1].getpixel((0, 0)) == (255, 0, 0)  # opaque -> untouched


def test_the_vision_half_and_the_latent_half_are_fitted_by_the_same_rule():
    """The reference latent is spliced into the vision tokens' place, so both halves
    have to agree on shape."""
    refs = [Image.new("RGB", (200, 100), "gray"), Image.new("RGB", (60, 300), "gray")]

    images = _bare()._encoder_images(EncodePromptArgs(prompt="p", image=refs))

    assert [img.size for img in images] == [
        QwenImage21Model.REFERENCE_SIZING.target_size(*img.size) for img in refs
    ]


def test_the_adapter_does_not_override_the_shared_decode():
    """The alpha drop-out used to be a ``decode`` override; it must not come back."""
    assert "decode" not in QwenImage21Model.__dict__
    assert QwenImage21Model.decode is DiffusionModel.decode


def test_the_adapter_upscales_with_the_trained_transcoder(monkeypatch):
    """The advertised 2x is the bridge, conditioned on this model's own VAE."""
    loaded = []

    def fake_load(device, dtype):
        loaded.append((device, dtype))
        return LatentTranscodeNet(feature_channels=6, latent_channels=4, width=4, depth=1)

    monkeypatch.setattr(qwen21_transcode, "_load_net", fake_load)
    model = _bare(_upscaler=None, vae=FakeVAE())

    upscaler = model.get_upscaler()

    assert isinstance(upscaler, Qwen21TranscodeUpscaler)
    assert upscaler.vae is model.vae  # conditioned on the model's own decoder
    assert upscaler.scale == QwenImage21Model.UPSCALE_SCALE
    assert loaded == [(torch.device("cpu"), torch.float32)]
    assert model.get_upscaler() is upscaler  # built once, then cached
