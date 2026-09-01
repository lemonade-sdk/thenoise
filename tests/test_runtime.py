"""Runtime tests using a fake model class (no torch adapters, no real weights)."""
from __future__ import annotations

import types

from thenoise.runtime import Settings, ModelPaths, NotLoadedError, Runtime


def _runtime_with_fake_model(monkeypatch):
    import thenoise.models as dm

    constructed = []

    class FakeModel:
        name = "fake"

        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr(dm, "MODEL_CATALOG", [FakeModel])
    monkeypatch.setattr(dm, "resolve", lambda path: FakeModel)
    return Runtime(Settings()), constructed


def test_empty_runtime(monkeypatch):
    runtime, _ = _runtime_with_fake_model(monkeypatch)
    assert runtime.available() == []
    assert runtime.model_capabilities() == {}
    try:
        runtime.model
        assert False, "expected NotLoadedError"
    except NotLoadedError:
        pass


def test_model_capabilities_reports_supports_edit(monkeypatch):
    import thenoise.models as dm

    class FakeEditModel:
        name = "fake"
        supports_edit = True
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(dm, "MODEL_CATALOG", [FakeEditModel])
    monkeypatch.setattr(dm, "resolve", lambda path: FakeEditModel)
    runtime = Runtime(Settings())
    runtime.load(ModelPaths("dit", "vae", "te"))
    assert runtime.model_capabilities() == {"supports_edit": True}


def test_load_resolves_model_from_dit(monkeypatch):
    runtime, constructed = _runtime_with_fake_model(monkeypatch)
    runtime.load(ModelPaths("dit", "vae", "te"))
    assert runtime.available() == ["fake"]
    assert runtime.model_name == "fake"
    assert constructed[0]["config"].dit_path == "dit"
    assert runtime.pipeline is not None


def test_load_swaps_single_model(monkeypatch):
    runtime, constructed = _runtime_with_fake_model(monkeypatch)
    runtime.load(ModelPaths("dit", "vae", "te"))
    # Loading a second model swaps (unloads) the first -- still one resident.
    runtime.load(ModelPaths("dit2", "vae2", "te2"))
    assert runtime.available() == ["fake"]
    assert len(constructed) == 2
    assert constructed[1]["config"].dit_path == "dit2"


def test_upscaler_dir_is_server_config(monkeypatch, tmp_path):
    """upscaler_dir comes from Settings (server config), not ModelPaths."""
    import thenoise.models as dm

    class FakeModel:
        name = "fake"
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(dm, "MODEL_CATALOG", [FakeModel])
    monkeypatch.setattr(dm, "resolve", lambda path: FakeModel)

    runtime = Runtime(Settings(upscaler_dir=str(tmp_path)))
    assert runtime.pixel_upscalers.upscaler_dir == str(tmp_path)
    runtime.load(ModelPaths("dit", "vae", "te"))
    assert runtime.pipeline is not None
