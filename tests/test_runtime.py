"""Runtime tests using fake catalog entries (no adapters, no real weights)."""
from __future__ import annotations

import threading

import pytest

import thenoise.models as model_catalog
from thenoise.errors import ModelReloadNeeded
from thenoise.inference import inference_lock
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


def test_video_weights_are_ready_when_runtime_load_returns(catalog, monkeypatch):
    from thenoise.models.video_base import VideoModel
    import thenoise.models as model_catalog

    events = []

    class FakeVideo(VideoModel):
        name = "video-test"
        DEFAULT_PREFS = {}

        @staticmethod
        def detect(_checkpoint):
            return True

        def load_weights(self):
            events.append("loaded")

        def generate(self, spec, token, progress, output):
            raise AssertionError("not part of this test")

        def _clear_weights(self):
            pass

    catalog()
    monkeypatch.setattr(model_catalog, "resolve", lambda _path: FakeVideo)
    rt = Runtime(Settings(device="cpu", offload_device="cpu"))
    rt.load(ModelPaths("dit", "vae", "te"))
    assert events == ["loaded"]
    assert rt.model_name == "video-test"
    rt.reload()
    assert events == ["loaded", "loaded"]

    class FailingVideo(FakeVideo):
        def load_weights(self):
            raise RuntimeError("corrupt video checkpoint")

    monkeypatch.setattr(model_catalog, "resolve", lambda _path: FailingVideo)
    with pytest.raises(RuntimeError, match="corrupt video checkpoint"):
        rt.load(ModelPaths("dit", "vae", "te"))
    assert rt.available() == []  # never publish an incompletely loaded adapter


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


def _counting_model(events: list):
    """A catalog entry that records its own construction and collection."""

    class Fake:
        name = "fake"
        CAPABILITIES: dict = {}
        detect = staticmethod(lambda path: False)

        def __init__(self, **kwargs):
            events.append("built")

        def __del__(self):
            events.append("released")

    return Fake


def _install(monkeypatch, cls) -> Runtime:
    monkeypatch.setattr(model_catalog, "MODEL_CATALOG", [cls])
    monkeypatch.setattr(model_catalog, "resolve", lambda path: cls)
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(ModelPaths("dit", "vae", "te"))
    return runtime


def test_a_swap_releases_the_replaced_model_before_building_the_next(monkeypatch):
    """The pipeline holds the model, so a swap dropping only the runtime's reference has
    both models resident at once.
    """
    events: list[str] = []
    runtime = _install(monkeypatch, _counting_model(events))
    runtime.load(ModelPaths("dit", "vae", "te"))

    assert events == ["built", "released", "built"]


def test_a_rebuild_after_a_refusal_frees_the_model_it_replaces(monkeypatch):
    """The refused request's own frames hold the model the rebuild must replace."""
    events: list[str] = []
    runtime = _install(monkeypatch, _counting_model(events))

    def action(pipeline):
        if len(events) == 1:
            raise ModelReloadNeeded("the DiT holds a half-applied LoRA")
        return "an image"

    assert runtime.run(action) == "an image"
    assert events == ["built", "released", "built"]


def test_a_request_touches_no_model_outside_the_inference_lock(catalog):
    """``finalize`` runs after the pipeline released the lock and still holds the model:
    a rebuild beside it stacks two models.
    """
    catalog()
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(ModelPaths("dit", "vae", "te"))
    attempts: list[int] = []

    worker = threading.Thread(
        target=lambda: runtime.run(lambda _pipeline: attempts.append(1)), daemon=True)
    with inference_lock:
        worker.start()
        worker.join(timeout=0.3)
        assert worker.is_alive(), "the request waits for the device's owner"
        assert attempts == []
    worker.join(timeout=5)
    assert attempts == [1]


def test_a_generation_runs_under_the_lock_the_request_holds(catalog):
    """A non-blocking acquire from the owner answers for reentrancy; a plain lock fails
    here rather than hanging the suite.
    """
    catalog()
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(ModelPaths("dit", "vae", "te"))

    def action(_pipeline):
        assert inference_lock.acquire(timeout=0), "the request must hold the lock"
        inference_lock.release()
        return "an image"

    assert runtime.run(action) == "an image"


def test_requests_refused_by_one_instance_rebuild_it_once(catalog):
    """Requests refused in succession rebuild once between them."""
    constructed = catalog()
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(ModelPaths("dit", "vae", "te"))
    refused = runtime.model
    outcomes: list = []

    def action(_pipeline):
        if runtime.model is refused:
            raise ModelReloadNeeded("the DiT holds a half-applied LoRA")
        return "an image"

    workers = [threading.Thread(
        target=lambda: outcomes.append(_run(runtime, action)), daemon=True)
        for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert outcomes == ["an image", "an image"], outcomes
    assert len(constructed) == 2, "one rebuild between them"


def test_running_without_a_model_says_so(catalog):
    catalog()
    with pytest.raises(NotLoadedError):
        Runtime(Settings()).run(lambda pipeline: pipeline)


def test_loading_swaps_the_single_resident_model(catalog):
    constructed = catalog()
    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
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


def _run(runtime, action):
    """``action`` through the runtime, or the error it raised, for a worker thread."""
    try:
        return runtime.run(action)
    except Exception as error:           # noqa: BLE001 - reported through the outcome
        return error
