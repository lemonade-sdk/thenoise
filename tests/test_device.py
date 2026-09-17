"""Device helpers and the offload-device auto-detection decision.

The backend calls are faked (this machine may have no GPU at all), so what is
under test is the *decision logic*: which backend hook runs for a device string,
and whether the resident weights are judged to fit in VRAM.
"""
from __future__ import annotations

import pytest
import torch

from conftest import StubModel
from thenoise.models.config import ModelConfig
from thenoise.utils import device as device_mod
from thenoise.utils.device import (
    clean_memory_on_device,
    get_device_memory,
    synchronize_device,
)


@pytest.fixture
def calls(monkeypatch):
    """Fakes for every backend hook the helpers can call, recording ``backend.hook``."""
    recorded = []

    class _Fake:
        def __init__(self, backend):
            self._backend = backend

        def __getattr__(self, name):
            def hook(*args, **kwargs):
                recorded.append(f"{self._backend}.{name}")

            return hook

    for backend in ("cuda", "mps", "xpu"):
        monkeypatch.setattr(device_mod.torch, backend, _Fake(backend), raising=False)
    return recorded


@pytest.mark.parametrize(
    "device,expected",
    [
        (None, []),
        ("cpu", []),
        (torch.device("cpu"), []),
        ("cuda", ["cuda.empty_cache"]),
        ("cuda:0", ["cuda.empty_cache"]),
        ("mps", ["mps.empty_cache"]),
    ],
)
def test_clean_memory_on_device(calls, device, expected):
    clean_memory_on_device(device)
    assert calls == expected


@pytest.mark.parametrize(
    "device,expected",
    [
        (None, []),
        ("cpu", []),
        ("cuda", ["cuda.synchronize"]),
        ("xpu", ["xpu.synchronize"]),
        ("mps", ["mps.synchronize"]),
    ],
)
def test_synchronize_device(calls, device, expected):
    synchronize_device(device)
    assert calls == expected


def test_get_device_memory_only_knows_about_cuda(monkeypatch):
    class _Props:
        total_memory = 128 * 1024**3

    monkeypatch.setattr(
        device_mod.torch.cuda, "get_device_properties", lambda device: _Props()
    )
    assert get_device_memory("cuda") == 128 * 1024**3
    assert get_device_memory(torch.device("cuda")) == 128 * 1024**3
    # Unknown device kinds report "unknown" rather than guessing.
    assert get_device_memory(None) is None
    assert get_device_memory("cpu") is None


def test_get_device_memory_accepts_a_device_object_with_an_index(monkeypatch):
    seen = []

    def properties(device):
        seen.append(device)
        return type("P", (), {"total_memory": 32 * 1024**3})

    monkeypatch.setattr(device_mod.torch.cuda, "get_device_properties", properties)
    assert get_device_memory("cuda:1") == 32 * 1024**3
    assert seen[0].index == 1


# ------------------------------------------------- offload device auto-detect


GB = 1024**3


def _config(tmp_path, *, dit_size=0, vae_size=0, te_size=0, offload_device=""):
    """A ModelConfig over three empty checkpoint files plus their faked sizes.

    The files are never actually populated: the sizes are handed back as a
    mapping for the ``fake_device`` fixture to install as ``_file_size``, so no
    multi-gigabyte files are written to disk.
    """
    paths, sizes = {}, {}
    for name, size in (("dit", dit_size), ("vae", vae_size), ("te", te_size)):
        path = tmp_path / f"{name}.safetensors"
        path.touch()  # empty file, size is faked
        paths[f"{name}_path"] = str(path)
        sizes[str(path)] = size
    config = ModelConfig(
        device="cuda",
        offload_device=offload_device,
        dtype=torch.float32,
        dit_path=paths["dit_path"],
        vae_path=paths["vae_path"],
        text_encoder_path=paths["te_path"],
    )
    return config, sizes


@pytest.fixture
def fake_device(monkeypatch):
    """Fake VRAM and per-file sizes so no real checkpoint files are written.

    ``get_device_memory`` and ``DiffusionModel._file_size`` are replaced; the
    decision logic under test (does the resident weight estimate fit VRAM) sees
    the exact numbers the caller provides.
    """

    def configure(*, vram=None, sizes=None):
        monkeypatch.setattr("thenoise.models.base.get_device_memory", lambda dev: vram)
        sizes = sizes or {}
        monkeypatch.setattr(
            "thenoise.models.base.DiffusionModel._file_size",
            staticmethod(lambda path: sizes.get(path, 0)),
        )

    return configure


def test_offload_device_stays_resident_when_the_weights_fit(tmp_path, fake_device):
    """8GB of weights against 128GB VRAM leaves plenty of activation headroom."""
    config, sizes = _config(tmp_path, dit_size=int(8 * GB))
    fake_device(vram=128 * GB, sizes=sizes)
    model = StubModel(config=config)
    assert model.offload_device == "cuda"


def test_offload_device_falls_to_cpu_when_they_do_not_fit(tmp_path, fake_device):
    """Weights may occupy at most 60% of VRAM; beyond that they offload to CPU."""
    config, sizes = _config(tmp_path, dit_size=int(6 * GB))
    fake_device(vram=8 * GB, sizes=sizes)
    model = StubModel(config=config)
    assert model.offload_device == "cpu"


def test_offload_device_stays_resident_when_vram_is_unknown(tmp_path, fake_device):
    """No VRAM information (e.g. a cpu-only build) -> no offloading."""
    config, sizes = _config(tmp_path, dit_size=int(40 * GB))
    fake_device(vram=None, sizes=sizes)
    model = StubModel(config=config)
    assert model.offload_device == "cuda"


def test_offload_device_is_configurable(tmp_path, fake_device):
    """``--offload-device`` overrides the size-based decision entirely."""
    config, sizes = _config(tmp_path, dit_size=int(1 * GB), offload_device="cpu")
    fake_device(vram=128 * GB, sizes=sizes)
    model = StubModel(config=config)
    assert model.offload_device == "cpu"


def test_missing_checkpoint_files_count_as_zero(tmp_path, fake_device):
    """An unreadable path must not blow the estimate up into an offload."""
    fake_device(vram=1 * GB)
    config = ModelConfig(
        dit_path=str(tmp_path / "nope.safetensors"),
        vae_path=str(tmp_path / "nope2.safetensors"),
        text_encoder_path=str(tmp_path / "nope3.safetensors"),
        device="cuda",
        dtype=torch.float32,
    )
    assert StubModel(config=config).offload_device == "cuda"
