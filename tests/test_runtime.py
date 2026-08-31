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


def test_generate_pixel_upscaler_sets_dir_before_runtime(monkeypatch, tmp_path):
    """``--pixel-upscaler`` must populate Settings before Runtime is constructed.

    Regression: Runtime builds the pixel-upscaler manager from
    ``settings.upscaler_dir`` at construction; if the generate path split
    ``--pixel-upscaler`` after that, the manager ended up with an empty dir.
    """
    import thenoise.generate as gen
    import thenoise.runtime as rt

    captured = {}

    class _Image:
        width = 64
        height = 64

        def save(self, path, pnginfo=None):
            pass

    class _Pipeline:
        def generate(self, request):
            captured["pixel_upscaler"] = request.pixel_upscaler
            return _Image()

    class _Runtime:
        def __init__(self, settings):
            captured["upscaler_dir"] = settings.upscaler_dir
            self.pipeline = _Pipeline()
            self.model_name = "fake"

        def load(self, paths):
            pass

    monkeypatch.setattr(rt, "Runtime", _Runtime)

    from thenoise.cli import build_parser

    args = build_parser().parse_args([
        "generate", "--checkpoint", "/tmp/mix.safetensors",
        "--prompt", "a fox",
        "--pixel-upscaler", f"{tmp_path}/RealESRGAN_x4plus.safetensors",
        "--upscale", "--out", "/tmp/out.png",
    ])
    gen.run_generate(args)

    assert captured["upscaler_dir"] == str(tmp_path)
    assert captured["pixel_upscaler"] == "RealESRGAN_x4plus"
