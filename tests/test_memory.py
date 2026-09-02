"""MemoryManager tests (no GPU).

The manager's real device moves are exercised with cpu (load) and meta (offload)
for the directions torch supports. The ``ensure``-loads-an-offloaded-component path
(offload -> load) is tested with a recording mock, since a real ``meta -> cpu``
move cannot carry data.
"""
from __future__ import annotations

import torch
from torch import nn

from thenoise.memory import MemoryManager


class _Comp(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(4, 4))


class _MockComp:
    """Records ``.to(device)`` calls; no real parameters (device = None)."""

    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(str(torch.device(device)))
        return self

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())


class _WrapperComp:
    """Mimics a wrapper embedder: exposes ``.device``/``.to`` over a real module."""

    def __init__(self, device):
        self._model = _Comp().to(device)
        self.to_calls = []

    @property
    def device(self):
        return next(self._model.parameters()).device

    def to(self, device):
        self.to_calls.append(str(torch.device(device)))
        self._model.to(device)
        return self

    def parameters(self):
        return iter(())


def _dev(m):
    return next(m.parameters()).device


# ------------------------------------------------------------ resident mode
def test_resident_mode_is_noop():
    m = _Comp()
    mm = MemoryManager("cpu", "cpu")  # offload == load -> no moves
    assert not mm.offloads
    mm.register("comp", m)
    mm.ensure("comp")
    assert "comp" in mm.resident()
    mm.offload("comp")
    # Resident mode: offload is a no-op, the component stays on the load device.
    assert _dev(m) == torch.device("cpu")
    assert "comp" in mm.resident()


# ------------------------------------------------------------ offload mode
def test_register_tracks_initial_residency():
    on_cpu = _Comp()
    on_meta = _Comp().to("meta")  # cpu -> meta is supported
    mm = MemoryManager("cpu", "meta")
    mm.register("a", on_cpu)
    mm.register("b", on_meta)
    assert "a" in mm.resident()
    assert "b" not in mm.resident()


def test_ensure_loads_offloaded_component():
    m = _MockComp()
    mm = MemoryManager("cpu", "meta")
    mm.register("comp", m)  # no params -> not resident
    assert "comp" not in mm.resident()
    mm.ensure("comp")
    assert m.to_calls == ["cpu"]
    assert "comp" in mm.resident()


def test_ensure_is_idempotent():
    m = _MockComp()
    mm = MemoryManager("cpu", "meta")
    mm.register("comp", m)
    mm.ensure("comp")
    mm.ensure("comp")
    assert m.to_calls == ["cpu"]  # only one move


def test_offload_moves_to_offload_device():
    m = _MockComp()
    mm = MemoryManager("cpu", "meta")
    mm.register("comp", m)
    mm.ensure("comp")
    mm.offload("comp")
    assert m.to_calls == ["cpu", "meta"]
    assert "comp" not in mm.resident()


def test_missing_component_is_noop():
    mm = MemoryManager("cpu", "meta")
    mm.ensure("nope")
    mm.offload("nope")
    assert mm.resident() == set()


# ------------------------------------------------------- wrappers (embedders)
def test_wrapper_component_tracks_residency():
    w = _WrapperComp("cpu")
    mm = MemoryManager("cpu", "meta")
    mm.register("te", w)
    assert "te" in mm.resident()


def test_wrapper_offload_moves_model():
    w = _WrapperComp("cpu")
    mm = MemoryManager("cpu", "meta")
    mm.register("te", w)
    mm.offload("te")
    assert w.device == torch.device("meta")
    assert "te" not in mm.resident()
