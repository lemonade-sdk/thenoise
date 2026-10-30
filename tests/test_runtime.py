"""Runtime tests using fake catalog entries (no adapters, no real weights)."""
from __future__ import annotations

import pytest

import thenoise.models as model_catalog
from thenoise.errors import ModelReloadNeeded
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


def test_load_resolves_the_model_from_the_dit_and_passes_the_paths_through(catalog):
    constructed = catalog()
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(ModelPaths("dit", "vae", "te", lora_dir="/loras"))

    assert runtime.available() == ["fake"]
    assert runtime.model_name == "fake"
    config = constructed[0]["config"]
    assert (config.dit_path, config.vae_path, config.text_encoder_path) == ("dit", "vae", "te")
    assert (config.device, config.offload_device, config.lora_dir) == ("cpu", "cpu", "/loras")
    assert runtime.pipeline is not None


def test_model_capabilities_are_reported_verbatim(catalog):
    """``/health`` gets the adapter's own capability dict, copied not aliased."""
    catalog(CAPABILITIES={"edit": True, "kv_cache": False})
    runtime = Runtime(Settings())
    runtime.load(ModelPaths("dit", "vae", "te"))
    assert runtime.model_capabilities() == {"edit": True, "kv_cache": False}
    runtime.model_capabilities()["edit"] = False
    assert runtime.model_capabilities()["edit"] is True


def test_loading_swaps_the_single_resident_model(catalog):
    constructed = catalog()
    runtime = Runtime(Settings())
    runtime.load(ModelPaths("dit", "vae", "te"))
    # Loading a second model unloads the first -- still exactly one resident.
    runtime.load(ModelPaths("dit2", "vae2", "te2"))

    assert runtime.available() == ["fake"]
    assert len(constructed) == 2
    assert constructed[1]["config"].dit_path == "dit2"


def test_upscaler_dir_is_server_config(catalog, tmp_path):
    """``upscaler_dir`` comes from ``Settings`` (server config), not ``ModelPaths``."""
    catalog()
    runtime = Runtime(Settings(upscaler_dir=str(tmp_path)))
    assert runtime.pixel_upscalers.upscaler_dir == str(tmp_path)

    runtime.load(ModelPaths("dit", "vae", "te"))
    # The pipeline shares the runtime's single upscaler pool (no second copy).
    assert runtime.pipeline._pixel_upscalers is runtime.pixel_upscalers


# ------------------------------------------------------------ rebuilding a model
def test_reload_rebuilds_from_the_same_paths(catalog):
    """The repair for a state no request can undo from inside: the checkpoints, again."""
    constructed = catalog()
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(ModelPaths("dit", "vae", "te"))

    runtime.reload()
    assert len(constructed) == 2
    assert runtime.model_name == "fake"
    assert runtime.pipeline is not None


def test_reload_without_a_model_says_so(catalog):
    catalog()
    with pytest.raises(NotLoadedError):
        Runtime(Settings()).reload()


def test_run_retries_once_against_the_rebuilt_model(catalog):
    """A model that reports itself unfit is rebuilt, and the retry reaches the new one."""
    constructed = catalog()
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(ModelPaths("dit", "vae", "te"))
    handed: list = []

    def action(pipeline):
        handed.append(pipeline)
        if len(handed) == 1:
            raise ModelReloadNeeded("the DiT holds a half-applied LoRA")
        return "an image"

    assert runtime.run(action) == "an image"
    assert len(constructed) == 2
    assert handed[0] is not handed[1], "the retry cannot run on the instance it left"""


def test_run_rebuilds_once_and_no_more(catalog):
    """One rebuild is the budget; a model unfit twice is somebody else's problem."""
    constructed = catalog()
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(ModelPaths("dit", "vae", "te"))

    def action(_pipeline):
        raise ModelReloadNeeded("still unfit")

    with pytest.raises(ModelReloadNeeded):
        runtime.run(action)
    assert len(constructed) == 2
