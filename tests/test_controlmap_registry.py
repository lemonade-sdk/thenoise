"""Pool discovery and residency: ``config.json`` -> kind, defaults, one resident.
Weight-free and CPU-only; the checkpoint kinds stand in via monkeypatched loaders.
"""
from __future__ import annotations

import json

import pytest
import torch

from thenoise.controlmap import depth, pose
from thenoise.controlmap.manager import ControlMapManager


def _checkpoint_dir(tmp_path, name, model_type):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    (directory / "model.safetensors").write_bytes(b"x")
    (directory / "preprocessor_config.json").write_text("{}")
    return directory


def _depth_dir(tmp_path, name="depth_anything_v2_small"):
    return _checkpoint_dir(tmp_path, name, "depth_anything")


def _pose_dirs(tmp_path, detector="rtdetr_r18vd"):
    """A pose run is two checkpoints: the keypoint model and a person detector."""
    _checkpoint_dir(tmp_path, "vitpose_base_simple", "vitpose")
    return _checkpoint_dir(tmp_path, detector, "rt_detr")


def test_discovery_maps_model_type_to_kind(tmp_path):
    _depth_dir(tmp_path)
    _pose_dirs(tmp_path)
    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
    assert manager.available() == {
        "canny": ["canny"],
        "lineart": ["lineart"],
        "softedge": ["softedge"],
        "depth": ["depth_anything_v2_small"],
        "pose": ["vitpose_base_simple"],
    }
    assert manager.defaults()["depth"] == "depth_anything_v2_small"
    assert manager.defaults()["pose"] == "vitpose_base_simple"


def test_a_non_controlmap_directory_is_ignored(tmp_path):
    (tmp_path / "not_a_controlmap").mkdir()
    (tmp_path / "not_a_controlmap" / "config.json").write_text('{"model_type": "gpt2"}')
    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
    assert manager.available() == {
        "canny": ["canny"],
        "lineart": ["lineart"],
        "softedge": ["softedge"],
    }


def test_pose_is_offered_only_with_both_of_its_checkpoints(tmp_path):
    """An estimator with nothing to crop with measures nothing, so the pool does not
    offer it — and says what is missing rather than staying quiet about the kind."""
    _checkpoint_dir(tmp_path, "vitpose_base_simple", "vitpose")
    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
    assert "pose" not in manager.available()
    assert manager.missing()["pose"] == (
        "needs a person detector (RT-DETR) beside its keypoint model; run "
        "'scripts/download.py --model controlmaps --variant pose' to fetch it"
    )
    with pytest.raises(ValueError, match="needs a person detector"):
        manager.get("pose")
    _checkpoint_dir(tmp_path, "rtdetr_r18vd", "rt_detr")
    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
    assert manager.available()["pose"] == ["vitpose_base_simple"]
    assert manager.detectors() == ["rtdetr_r18vd"]
    assert "pose" not in manager.missing()


def test_a_kind_with_no_checkpoint_at_all_is_simply_not_on_offer(tmp_path):
    """Nothing half-installed: a kind with no checkpoint is absent, and ``--list``
    explains what would complete it."""
    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
    assert manager.available() == {
        "canny": ["canny"],
        "lineart": ["lineart"],
        "softedge": ["softedge"],
    }
    assert manager.missing() == {
        "depth": "has no checkpoint in the control-map directory; run "
        "'scripts/download.py --model controlmaps' to fetch it",
        "pose": "has no checkpoint in the control-map directory; run "
        "'scripts/download.py --model controlmaps --variant pose' to fetch it",
    }


def test_each_kind_loads_only_what_it_runs_with(tmp_path, monkeypatch):
    """Depth must not load a person detector, and pose must not load a depth model:
    each kind loads only what it runs with."""
    loaded = []
    module = torch.nn.Module()
    module.config = type("Config", (), {"edges": [], "id2label": {}})()
    monkeypatch.setattr(
        pose, "load_model", lambda path, dtype: loaded.append("estimator") or module
    )
    monkeypatch.setattr(
        pose, "load_detector", lambda path, dtype: loaded.append("detector") or module
    )
    monkeypatch.setattr(
        pose, "load_processor", lambda path: loaded.append("processor") or object()
    )
    monkeypatch.setattr(
        depth, "load_model", lambda path, dtype: loaded.append("depth") or module
    )
    monkeypatch.setattr(
        depth, "load_processor", lambda path: loaded.append("depth processor") or object()
    )
    _depth_dir(tmp_path)
    _pose_dirs(tmp_path)

    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
    manager.get("depth")
    assert loaded == ["depth processor", "depth"]
    manager.get("pose")
    # The detector waits for the first run: a request that names another one would
    # otherwise have paid for a model it never uses.
    assert loaded == ["depth processor", "depth", "processor", "estimator"]


def test_unknown_name_names_the_kind_implementations(tmp_path):
    _depth_dir(tmp_path)
    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
    with pytest.raises(ValueError, match="no depth control map named 'depth-xl'"):
        manager.get("depth", "depth-xl")


def test_canny_is_available_with_an_empty_dir(tmp_path):
    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
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

    monkeypatch.setattr(depth, "load_processor", fake_processor)
    monkeypatch.setattr(depth, "load_model", fake_model)

    manager = ControlMapManager(device="cpu", controlmap_dir=str(tmp_path))
    canny = manager.get("canny")
    depth_pre = manager.get("depth")
    assert canny is not depth_pre
    assert calls == ["processor", "model"]
    # Re-requesting the same controlmap is a no-op.
    assert manager.get("depth") is depth_pre
    assert calls == ["processor", "model"]
