"""Alpha-aware pipeline tests: the channels an RGBA model gets in and gives out.

The pipeline carries the VAE's channel count from the input image to the PNG, so
these tests are the seams where that can leak: the reference (edit) encode, the
decode -> PIL tail, and the reference cache key. They run over the weight-free
``StubModel`` (CPU, fp32) with its ``pixel_channels`` overridden to 4.
"""
from __future__ import annotations

import pytest
import torch
from PIL import Image

from conftest import EditingStubModel, StubModel
from thenoise.models.config import GenerateRequest
from thenoise.pipeline import PipelineController
from thenoise.upscale.pixel import PixelUpscalerManager


def _controller(model, *, upscaler_dir="", upscaler_scales=None):
    manager = PixelUpscalerManager(upscaler_dir=upscaler_dir, device="cpu")
    manager._pixel_upscaler_scales = dict(upscaler_scales or {})
    return PipelineController(model, manager)


def _request(**kwargs) -> GenerateRequest:
    return GenerateRequest(**{"prompt": "a fox", "seed": 1, **kwargs})


class _RGBAStubModel(EditingStubModel):
    """An editing stub on an RGBA VAE: 4 channels in, 4 channels out."""

    pixel_channels = 4

    def decode(self, latents):
        self.calls["decode"] += 1
        h, w = latents.shape[-2:]
        pixels = torch.zeros(4, h * self._VAE_SCALE, w * self._VAE_SCALE)
        pixels[3] = 0.25  # a recognisable, non-trivial matte
        return pixels


def _transparent_red(size=(64, 64)) -> Image.Image:
    """Fully transparent red: anything that drops the alpha shows up as red."""
    return Image.new("RGBA", size, (255, 0, 0, 0))


# ------------------------------------------------------------------- encode side


def test_reference_pixels_take_the_vaes_channel_count():
    """An RGBA VAE is handed the alpha; an RGB one is handed it composited."""
    rgba = _RGBAStubModel()
    _controller(rgba).edit(_request(image=_transparent_red(), steps=1))
    ref = rgba.prepared[0]["ref"][0]  # [1, C, H, W] as ``encode_reference`` returned it
    assert ref.shape[1] == 4
    assert torch.all(ref[:, 3] < 0)  # the (fully transparent) alpha reached the VAE

    rgb = EditingStubModel()
    _controller(rgb).edit(_request(image=_transparent_red(), steps=1))
    ref = rgb.prepared[0]["ref"][0]
    assert ref.shape[1] == 3
    # Composited onto white, NOT dropped: dropped would have left the red through.
    assert torch.allclose(ref[0, :, 0, 0], torch.ones(3))


def test_an_opaque_reference_is_unaffected_by_the_alpha_plumbing():
    """The RGB path must not start compositing opaque images onto anything."""
    model = EditingStubModel()
    _controller(model).edit(
        _request(image=Image.new("RGB", (64, 64), (255, 0, 0)), steps=1)
    )
    ref = model.prepared[0]["ref"][0]
    assert torch.allclose(ref[0, :, 0, 0], torch.tensor([1.0, -1.0, -1.0]))


def test_reference_cache_key_sees_the_alpha():
    """Two references that differ only in transparency are not the same input."""
    controller = _controller(StubModel())
    key = lambda img: controller._cache_key_reference([img], 64, 64)

    opaque = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
    clear = Image.new("RGBA", (8, 8), (1, 2, 3, 0))
    rgb = Image.new("RGB", (8, 8), (1, 2, 3))

    assert key(opaque) != key(clear)
    assert key(opaque) != key(rgb)
    assert key(opaque) == key(Image.new("RGBA", (8, 8), (1, 2, 3, 255)))


# -------------------------------------------------------------------- decode side


def test_generate_returns_rgba_when_the_vae_decodes_four_channels():
    """The alpha the VAE decoded is the alpha in the returned PIL image."""
    image = _controller(_RGBAStubModel()).generate(_request(steps=1))

    assert image.mode == "RGBA"
    # The matte is the one the stub decode wrote (0.25 -> 159/255), not 255.
    assert image.getpixel((0, 0))[3] == pytest.approx(159, abs=1)


def test_postprocessing_does_not_strip_the_alpha():
    """Sharpening and grain work on RGB and must leave the matte standing."""
    image = _controller(_RGBAStubModel()).generate(
        _request(steps=1, sharpening=0.5, film_grain=2.0)
    )
    assert image.mode == "RGBA"


def test_pixel_upscale_of_an_rgba_generation_keeps_it_rgba(tmp_path, monkeypatch):
    """The RGB-only ESRGAN step must not be where a generation loses its alpha."""
    (tmp_path / "esrgan.safetensors").write_text("x")
    controller = _controller(
        _RGBAStubModel(), upscaler_dir=str(tmp_path), upscaler_scales={"esrgan": 4}
    )
    seen = []

    def fake_apply(_self, name, pixels, scale):
        seen.append((pixels.shape[0], scale))
        # Mimic the manager: RGB for the model, matte resampled and re-attached.
        rgb, alpha = pixels[:3], pixels[3:]
        return torch.cat(
            [
                torch.nn.functional.interpolate(
                    rgb.unsqueeze(0), scale_factor=scale, mode="nearest"
                )[0],
                torch.nn.functional.interpolate(
                    alpha.unsqueeze(0), scale_factor=scale, mode="bilinear",
                    align_corners=False,
                )[0],
            ],
            dim=0,
        )

    monkeypatch.setattr(PixelUpscalerManager, "apply", fake_apply)
    image = controller.generate(
        _request(
            steps=1,
            upscale=True,
            upscale_factor=4.0,
            upscale_type="no-refiner",
            pixel_upscaler="esrgan",
        )
    )

    assert seen == [(4, 4)]  # the alpha was still there when ESRGAN was called
    assert image.mode == "RGBA"
    assert image.size == (256, 256)  # 64 * the requested 4x


def test_the_alpha_survives_into_the_png_bytes():
    """What the UI/browser receives is RGBA, not just what ``generate`` returned."""
    import io

    image = _controller(_RGBAStubModel()).generate(_request(steps=1))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    assert Image.open(buf).mode == "RGBA"
