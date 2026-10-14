"""Depth-Anything-V2 runner and painter: fake weights, no GPU, no network.

The preprocessor loads through the module's ``load_model`` / ``load_processor``, so
monkeypatching those two exercises the whole run path with a fixed ``predicted_depth``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from transformers import BatchFeature

from thenoise.preprocess import depth
from thenoise.preprocess.depth import DepthOptions, DepthPreprocessor


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
    """Install the fakes and return (preprocessor, model, processor)."""
    model = FakeDepthModel(shape)
    processor = FakeProcessor()
    directory = tmp_path / "depth_anything_v2_small"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type": "depth_anything"}')
    (directory / "model.safetensors").write_bytes(b"x")
    (directory / "preprocessor_config.json").write_text("{}")

    monkeypatch.setattr(depth, "load_processor", lambda path: processor)
    monkeypatch.setattr(depth, "load_model", lambda path, dtype: model)
    pre = DepthPreprocessor()
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
    with pytest.raises(ValueError, match="unknown depth colormap 'turbo'"):
        DepthOptions(colormap="turbo")


def test_missing_weights_names_the_download_command(tmp_path):
    """A directory that declares depth but has no weights fails with the remedy."""
    directory = tmp_path / "depth_anything_v2_small"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type": "depth_anything"}')
    pre = DepthPreprocessor()
    with pytest.raises(ValueError, match="scripts/download.py --model preprocess"):
        pre.load("cpu", str(directory))


def test_load_is_lazy_until_weights_are_needed():
    """A fresh preprocessor has no model resident; only load() brings the weights."""
    pre = DepthPreprocessor()
    assert not hasattr(pre, "_model")
