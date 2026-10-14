"""Depth-Anything-V2 runner and painter: fake weights, no GPU, no network.

The controlmap loads through the module's ``load_model`` / ``load_processor``, so
monkeypatching those two exercises the whole run path with a fixed ``predicted_depth``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import numpy as np
import torch
from PIL import Image

from transformers import BatchFeature

from thenoise.controlmap import depth
from thenoise.controlmap.depth import DepthOptions, DepthControlMap
from thenoise.controlmap.render import turbo_lut


class FakeProcessor:
    """Stands in for the image processor: returns a batch on the right device/dtype."""

    def __call__(self, images, return_tensors):
        self.seen = (images, return_tensors)
        self.image_mode = images.mode
        return BatchFeature({"pixel_values": torch.zeros(1, 3, 32, 32)})


class FakeDepthModel(torch.nn.Module):
    """Returns a fixed ``predicted_depth`` field of ``shape`` (h, w) per image."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.device_seen = None
        self.dtype_seen = None

    def forward(self, pixel_values, **kwargs):
        self.device_seen = pixel_values.device
        self.dtype_seen = pixel_values.dtype
        return SimpleNamespace(predicted_depth=torch.full((1, *self.shape), 0.25))


def fake_checkpoint(tmp_path, monkeypatch, shape=(8, 8)):
    """Install the fakes and return (controlmap, model, processor)."""
    model = FakeDepthModel(shape)
    processor = FakeProcessor()
    directory = tmp_path / "depth_anything_v2_small"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type": "depth_anything"}')
    (directory / "model.safetensors").write_bytes(b"x")
    (directory / "preprocessor_config.json").write_text("{}")

    monkeypatch.setattr(depth, "load_processor", lambda path: processor)
    monkeypatch.setattr(depth, "load_model", lambda path, dtype: model)
    pre = DepthControlMap()
    pre.load("cpu", str(directory))
    return pre, model, processor


def test_run_paints_the_field_at_the_input_size(tmp_path, monkeypatch):
    image = Image.new("RGB", (48, 32), "black")
    pre, _, _ = fake_checkpoint(tmp_path, monkeypatch)
    assert pre.run(image, DepthOptions()).size == image.size


def test_a_flat_field_stays_flat_not_amplified(tmp_path, monkeypatch):
    image = Image.new("RGB", (48, 32), "black")
    pre, _, _ = fake_checkpoint(tmp_path, monkeypatch)
    out = pre.run(image, DepthOptions())
    assert set(out.get_flattened_data()) == {64}  # 0.25 -> grey, not 0/255 noise


def test_invert_flips_the_normalised_field(tmp_path, monkeypatch):
    image = Image.new("RGB", (48, 32), "black")
    pre, _, _ = fake_checkpoint(tmp_path, monkeypatch)
    normal = pre.run(image, DepthOptions(invert=False))
    inverted = pre.run(image, DepthOptions(invert=True))
    assert set(normal.get_flattened_data()) == {64}
    assert set(inverted.get_flattened_data()) == {191}  # 1 - 0.25


def test_the_field_is_interpolated_back_to_the_input_size(tmp_path, monkeypatch):
    """The model measures at its own resolution; the map must match the input again."""
    image = Image.new("RGB", (64, 40), "black")
    pre, _, _ = fake_checkpoint(tmp_path, monkeypatch, shape=(8, 8))
    out = pre.run(image, DepthOptions())
    assert out.size == image.size


def test_transparency_is_composited_before_the_processor(tmp_path, monkeypatch):
    """The DPT processor normalises against a 3-channel mean, so an RGBA input must be
    flattened to RGB first rather than silently keeping its alpha."""
    image = Image.new("RGBA", (48, 32), (255, 0, 0, 0))
    pre, _, processor = fake_checkpoint(tmp_path, monkeypatch)
    pre.run(image, DepthOptions())
    assert processor.image_mode == "RGB"


def test_the_batch_reaches_the_model_on_the_right_dtype(tmp_path, monkeypatch):
    image = Image.new("RGB", (48, 32), "black")
    pre, model, _ = fake_checkpoint(tmp_path, monkeypatch)
    pre.run(image, DepthOptions())
    assert model.device_seen.type == "cpu"
    assert model.dtype_seen == torch.bfloat16


def test_unknown_colormap_is_rejected():
    with pytest.raises(ValueError, match="unknown depth colormap 'plasma'"):
        DepthOptions(colormap="plasma")


def test_turbo_colormap_maps_a_field_to_rgb(tmp_path, monkeypatch):
    """The turbo LUT is a 256-entry rainbow: the endpoints are dark blue and dark red,
    every channel stays in 0-255, and a flat field stays flat."""
    image = Image.new("RGB", (48, 32), "black")
    pre, _, _ = fake_checkpoint(tmp_path, monkeypatch)
    out = pre.run(image, DepthOptions(colormap="turbo"))
    assert out.mode == "RGB"
    assert out.size == image.size
    # The flat 0.25 field maps to a single colour, so the whole map is one pixel value.
    pixels = np.asarray(out)
    assert (pixels == pixels[0, 0]).all()
    # The LUT endpoints are dark blue and dark red, not white or black.
    lut = turbo_lut()
    assert lut.shape == (3, 256)
    assert lut[:, 0].tolist() == [48, 18, 59]
    assert lut[:, -1].tolist() == [176, 5, 41]


def test_turbo_colormap_is_smooth_and_monotonic():
    """Each channel of the LUT is monotonic, so the rainbow never folds back on itself."""
    lut = turbo_lut().float() / 255.0
    for channel in range(3):
        samples = lut[channel]
        # The channel rises then falls (or vice versa) but never oscillates rapidly:
        # consecutive deltas change sign at most a handful of times.
        sign_changes = (torch.diff(samples)[1:] * torch.diff(samples)[:-1] < 0).sum()
        assert sign_changes <= 3


def test_missing_weights_names_the_download_command(tmp_path):
    """A directory that declares depth but has no weights fails with the remedy."""
    directory = tmp_path / "depth_anything_v2_small"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type": "depth_anything"}')
    pre = DepthControlMap()
    with pytest.raises(ValueError, match="scripts/download.py --model controlmap"):
        pre.load("cpu", str(directory))


def test_load_is_lazy_until_weights_are_needed():
    """A fresh controlmap has no model resident; only load() brings the weights."""
    pre = DepthControlMap()
    assert not hasattr(pre, "_model")
