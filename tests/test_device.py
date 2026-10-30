"""Device helpers and where a model builds its weights.

The backend calls are faked (this machine may have no GPU), so what is under test is
the decision logic: which backend hook runs for a device string, and how a component's
size reaches the manager that decides where it is built.
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


def test_release_device_memory_collects_before_emptying(monkeypatch):
    """Collecting is what makes the empty useful: a tensor only frees its allocator
    block once the last reference to it is gone, so the order is the whole point."""
    order = []
    monkeypatch.setattr(device_mod.gc, "collect", lambda: order.append("collect"))
    monkeypatch.setattr(
        device_mod, "clean_memory_on_device", lambda d: order.append(("empty", d))
    )
    device_mod.release_device_memory("cuda")
    assert order == ["collect", ("empty", "cuda")]


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


# ------------------------------------------------------ load-time admission
GB = 1024**3


def _config(tmp_path, *, offload_device="cpu"):
    return ModelConfig(
        device="cuda",
        offload_device=offload_device,
        dtype=torch.float32,
        dit_path=str(tmp_path / "dit.safetensors"),
        vae_path=str(tmp_path / "vae.safetensors"),
        text_encoder_path=str(tmp_path / "te"),
    )


def test_idle_weights_go_where_the_operator_said(tmp_path):
    assert StubModel(config=_config(tmp_path)).offload_device == "cpu"
    assert StubModel(
        config=_config(tmp_path, offload_device="cuda")).offload_device == "cuda"


def test_a_model_carries_each_checkpoint_to_the_manager(tmp_path, monkeypatch):
    """The adapter knows the file, the manager owns the answer; nothing decides here."""
    config = _config(tmp_path)
    monkeypatch.setattr("thenoise.memory.get_free_memory", lambda _dev: 10 * GB)
    monkeypatch.setattr("thenoise.memory._RESERVE_BASE_BYTES", GB)
    model = StubModel(config=config)
    monkeypatch.setattr("thenoise.models.base.checkpoint_bytes",
                        {config.dit_path: 12 * GB, config.vae_path: GB}.get)

    assert model._load_target(config.dit_path) == "cpu", "12 GiB into 10 free leaves no floor"
    assert model._load_target(config.vae_path) == "cuda"
