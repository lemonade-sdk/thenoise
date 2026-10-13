"""Pool discovery and residency: ``config.json`` -> kind, defaults, one resident.
Weight-free and CPU-only; depth stands in via monkeypatched ``_hf`` loaders.
"""
from __future__ import annotations

import json

import pytest
import torch

from thenoise.preprocess import _hf
from thenoise.preprocess.manager import PreprocessorManager


def _depth_dir(tmp_path, name="depth_anything_v2_small"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "depth_anything"}))
    (d / "model.safetensors").write_bytes(b"x")
    (d / "preprocessor_config.json").write_text("{}")
    return d


def test_discovery_maps_model_type_to_kind(tmp_path):
    _depth_dir(tmp_path)
    manager = PreprocessorManager(device="cpu", preprocessor_dir=str(tmp_path))
    assert manager.available() == {
        "canny": ["canny"],
        "depth": ["depth_anything_v2_small"],
    }
    assert manager.defaults()["depth"] == "depth_anything_v2_small"


def test_a_non_preprocessor_directory_is_ignored(tmp_path):
    (tmp_path / "not_a_preprocessor").mkdir()
    (tmp_path / "not_a_preprocessor" / "config.json").write_text('{"model_type": "gpt2"}')
    manager = PreprocessorManager(device="cpu", preprocessor_dir=str(tmp_path))
    assert manager.available() == {"canny": ["canny"]}


def test_unknown_name_names_the_kind_implementations(tmp_path):
    _depth_dir(tmp_path)
    manager = PreprocessorManager(device="cpu", preprocessor_dir=str(tmp_path))
    with pytest.raises(ValueError, match="no depth preprocessor named 'depth-xl'"):
        manager.get("depth", "depth-xl")


def test_canny_is_available_with_an_empty_dir(tmp_path):
    manager = PreprocessorManager(device="cpu", preprocessor_dir=str(tmp_path))
    assert "canny" in manager.available()


def test_last_used_stays_resident_and_swaps_on_kind_change(tmp_path, monkeypatch):
    """One resident at a time: asking for depth drops canny and loads only once."""
    _depth_dir(tmp_path)
    calls = []

    def fake_processor(path):
        calls.append("processor")
        return object()

    def fake_model(path, dtype):
        calls.append("model")
        return torch.nn.Module()

    monkeypatch.setattr(_hf, "load_depth_processor", fake_processor)
    monkeypatch.setattr(_hf, "load_depth_model", fake_model)

    manager = PreprocessorManager(device="cpu", preprocessor_dir=str(tmp_path))
    canny = manager.get("canny")
    depth = manager.get("depth")
    assert canny is not depth
    assert calls == ["processor", "model"]
    # Re-requesting the same preprocessor is a no-op.
    assert manager.get("depth") is depth
    assert calls == ["processor", "model"]
