"""Runtime tests using fake catalog entries (no adapters, no real weights)."""
from __future__ import annotations

import pytest

import thenoise.models as model_catalog
from thenoise.runtime import ModelPaths, NotLoadedError, Runtime, Settings


@pytest.fixture
def catalog(monkeypatch, fake_model_cls):
    """Replace the model catalog with a single fake entry and record construction."""

    def install(**attrs):
        constructed: list = []
        cls = fake_model_cls(constructed=constructed, **attrs)
        monkeypatch.setattr(model_catalog, "MODEL_CATALOG", [cls])
        monkeypatch.setattr(model_catalog, "resolve", lambda path: cls)
        return constructed

    return install


def test_empty_runtime(catalog):
    catalog()
    runtime = Runtime(Settings())
    assert runtime.available() == []
    assert runtime.model_capabilities() == {}
    with pytest.raises(NotLoadedError):
        _ = runtime.model
    with pytest.raises(NotLoadedError):
        _ = runtime.model_name
    assert runtime.pipeline is None


def test_load_resolves_the_model_from_the_dit(catalog):
    constructed = catalog()
    runtime = Runtime(Settings())
    runtime.load(ModelPaths("dit", "vae", "te"))

    assert runtime.available() == ["fake"]
    assert runtime.model_name == "fake"
    config = constructed[0]["config"]
    assert (config.dit_path, config.vae_path, config.text_encoder_path) == ("dit", "vae", "te")
    assert runtime.pipeline is not None


def test_model_capabilities_reports_supports_edit(catalog):
    catalog(supports_edit=True)
    runtime = Runtime(Settings())
    runtime.load(ModelPaths("dit", "vae", "te"))
    assert runtime.model_capabilities() == {"supports_edit": True}


def test_loading_swaps_the_single_resident_model(catalog):
    constructed = catalog()
    runtime = Runtime(Settings())
    runtime.load(ModelPaths("dit", "vae", "te"))
    # Loading a second model unloads the first -- still exactly one resident.
    runtime.load(ModelPaths("dit2", "vae2", "te2"))

    assert runtime.available() == ["fake"]
    assert len(constructed) == 2
    assert constructed[1]["config"].dit_path == "dit2"


def test_load_passes_device_offload_device_and_lora_dir(catalog):
    constructed = catalog()
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(ModelPaths("dit", "vae", "te", lora_dir="/loras"))

    config = constructed[0]["config"]
    assert (config.device, config.offload_device, config.lora_dir) == ("cpu", "cpu", "/loras")


def test_upscaler_dir_is_server_config(catalog, tmp_path):
    """``upscaler_dir`` comes from ``Settings`` (server config), not ``ModelPaths``."""
    catalog()
    runtime = Runtime(Settings(upscaler_dir=str(tmp_path)))
    assert runtime.pixel_upscalers.upscaler_dir == str(tmp_path)

    runtime.load(ModelPaths("dit", "vae", "te"))
    # The pipeline shares the runtime's single upscaler pool (no second copy).
    assert runtime.pipeline._pixel_upscalers is runtime.pixel_upscalers
