"""Device helpers and the offload-device auto-detection decision.

The backend calls are faked (this machine may have no GPU), so what is under test is
the decision logic: which backend hook runs for a device string, and whether the
resident weights are judged to fit in VRAM.
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
        ("cuda", ["cuda.empty_cache", "cuda.synchronize"]),
        ("cuda:0", ["cuda.empty_cache", "cuda.synchronize"]),
        ("xpu", ["xpu.synchronize"]),
        ("mps", ["mps.empty_cache", "mps.synchronize"]),
    ],
)
def test_backend_hooks_only_fire_for_accelerator_devices(calls, device, expected):
    clean_memory_on_device(device)
    synchronize_device(device)
    assert calls == expected


def test_get_device_memory_only_knows_about_cuda(monkeypatch):
    seen = []

    def properties(device):
        seen.append(device)
        return type("P", (), {"total_memory": 128 * 1024**3})

    monkeypatch.setattr(device_mod.torch.cuda, "get_device_properties", properties)
    assert get_device_memory("cuda") == 128 * 1024**3
    assert get_device_memory(torch.device("cuda:1")) == 128 * 1024**3
    assert [d.index for d in seen] == [None, 1]  # the device index reaches the backend

    # Unknown device kinds report "unknown" rather than guessing.
    assert get_device_memory(None) is None
    assert get_device_memory("cpu") is None


# ------------------------------------------------- offload device auto-detect


GB = 1024**3


def _config(tmp_path, *, dit_size=0, vae_size=0, te_size=0, offload_device=""):
    """A ModelConfig over three empty checkpoint files, with faked per-file sizes."""
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
    """Fake VRAM and per-file sizes so the fit/no-fit decision needs no real files."""

    def configure(*, vram=None, sizes=None):
        monkeypatch.setattr("thenoise.models.base.get_device_memory", lambda dev: vram)
        sizes = sizes or {}
        monkeypatch.setattr(
            "thenoise.models.base.DiffusionModel._file_size",
            staticmethod(lambda path: sizes.get(path, 0)),
        )

    return configure


@pytest.mark.parametrize(
    "vram,dit_size,expected",
    [
        # 8GB of weights against 128GB VRAM leaves plenty of activation headroom.
        (128 * GB, int(8 * GB), "cuda"),
        # Weights may occupy at most 60% of VRAM; beyond that they offload to CPU.
        (8 * GB, int(6 * GB), "cpu"),
        # No VRAM information (e.g. a cpu-only build) -> no offloading.
        (None, int(40 * GB), "cuda"),
    ],
    ids=["fits", "does-not-fit", "unknown-vram"],
)
def test_offload_device_follows_the_weight_estimate(tmp_path, fake_device, vram, dit_size, expected):
    config, sizes = _config(tmp_path, dit_size=dit_size)
    fake_device(vram=vram, sizes=sizes)
    assert StubModel(config=config).offload_device == expected


def test_offload_device_is_configurable(tmp_path, fake_device):
    """``--offload-device`` overrides the size-based decision entirely."""
    config, sizes = _config(tmp_path, dit_size=int(1 * GB), offload_device="cpu")
    fake_device(vram=128 * GB, sizes=sizes)
    assert StubModel(config=config).offload_device == "cpu"
