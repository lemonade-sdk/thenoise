"""MemoryManager (no GPU).

Real device moves are exercised with cpu (load) and meta (offload); the offload ->
load direction needs a recording mock, since a ``meta -> cpu`` move cannot carry data.
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
    mm.offload("comp")
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


def test_ensure_then_offload_moves_once_each_way():
    m = _MockComp()
    mm = MemoryManager("cpu", "meta")
    mm.register("comp", m)  # no params -> not resident
    assert "comp" not in mm.resident()

    mm.ensure("comp")
    mm.ensure("comp")  # idempotent: only one move
    assert m.to_calls == ["cpu"]
    assert "comp" in mm.resident()

    mm.offload("comp")
    assert m.to_calls == ["cpu", "meta"]
    assert "comp" not in mm.resident()


def test_missing_component_is_noop():
    mm = MemoryManager("cpu", "meta")
    mm.ensure("nope")
    mm.offload("nope")
    assert mm.resident() == set()


# ------------------------------------------------------- wrappers (embedders)
def test_wrapper_offload_moves_the_inner_model():
    w = _WrapperComp("cpu")
    mm = MemoryManager("cpu", "meta")
    mm.register("te", w)
    assert "te" in mm.resident()

    mm.offload("te")
    assert w.device == torch.device("meta")
    assert "te" not in mm.resident()
