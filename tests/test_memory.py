"""``MemoryManager``: placement under pressure, residency by default (no GPU).

Device moves are exercised with cpu (load) and meta (offload). A meta tensor reports
its byte size and can travel in either direction, which is what lets the residency
rules be tested where there is no device — see ``conftest.TrackedModule``.
"""
from __future__ import annotations

import torch
from torch import nn

from conftest import TrackedModule
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


def _dev(m):
    return next(m.parameters()).device


def _mib(n: int) -> int:
    return n * 1024 * 1024


def _capacity(monkeypatch, mm: MemoryManager, capacity: int) -> None:
    """Free memory as a device reports it: capacity minus what is resident right now."""
    monkeypatch.setattr(
        mm, "free_bytes",
        lambda: capacity - sum(mm.weight_bytes(n) for n in mm.resident()),
    )


# ------------------------------------------------------------ resident mode
def test_resident_mode_is_noop():
    m = _Comp()
    mm = MemoryManager("cpu", "cpu")  # offload == load -> no moves
    assert not mm.offloads
    mm.register("comp", m)
    with mm.stage("comp", activations=0):
        pass
    mm.offload("comp")
    assert _dev(m) == torch.device("cpu")
    assert "comp" in mm.resident()


# ------------------------------------------------------- swappable components
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


# ------------------------------------------------------------------ wrappers
class _WrapperComp:
    """Mimics an embedder that is not a plain module: ``device`` and ``to``, no parameters."""

    def __init__(self, device: str):
        self._at = torch.device(device)
        self.to_calls: list[str] = []

    @property
    def device(self):
        return self._at

    def to(self, device):
        self.to_calls.append(str(torch.device(device)))
        self._at = torch.device(device)
        return self


def test_a_wrapper_is_placed_by_the_device_it_reports():
    swappable = _WrapperComp("meta")
    arrived = _WrapperComp("cpu")
    mm = MemoryManager("cpu", "meta")
    mm.register("embedder", swappable)
    mm.register("onboard", arrived)

    mm.ensure("embedder")
    assert swappable.to_calls == ["cpu"] and "embedder" in mm.resident()

    mm.offload("embedder")
    assert swappable.to_calls == ["cpu", "meta"]

    mm.offload("onboard")
    assert arrived.to_calls == [], "it came up on the compute device, so it is fixed there"


# --------------------------------------------------------------- residency
def test_leaving_a_stage_moves_nothing(monkeypatch):
    """Residency is where things are left, not a state somebody grants."""
    mm = MemoryManager("cpu", "meta")
    dit = TrackedModule(1024)
    mm.register("dit", dit)

    with mm.stage("dit", activations=_mib(1)):
        assert mm.resident() == {"dit"}
    assert dit.moves == ["cpu"]
    assert mm.resident() == {"dit"}, "the next stage gets it without a reload"


def test_a_component_that_arrived_on_the_device_is_not_traded_away(monkeypatch):
    """The VAE is built on the compute device, and a stranger's claim cannot evict it."""
    mm = MemoryManager("cpu", "meta")
    vae, dit = TrackedModule(_mib(1), device="cpu"), TrackedModule(_mib(4))
    mm.register("vae", vae)
    mm.register("dit", dit)

    _capacity(monkeypatch, mm, _mib(5))
    with mm.stage("dit", activations=_mib(16)):     # more than the device can hold
        assert mm.resident() == {"vae", "dit"}
    assert vae.moves == []


def test_an_idle_weight_set_pays_for_a_claim_it_cannot_beside(monkeypatch):
    """Largest first, and only when the device genuinely cannot hold both."""
    mm = MemoryManager("cpu", "meta")
    small, big, wanted = (TrackedModule(_mib(1)), TrackedModule(_mib(4)),
                          TrackedModule(_mib(1)))
    mm.register("small", small)
    mm.register("big", big)
    mm.register("wanted", wanted)

    _capacity(monkeypatch, mm, _mib(20))
    with mm.stage("small", "big", activations=_mib(1)):
        pass
    assert mm.resident() == {"small", "big"}, "there was room; nothing was disturbed"

    _capacity(monkeypatch, mm, _mib(5))      # one eviction's worth short
    with mm.stage("wanted", activations=_mib(1)):
        assert mm.resident() == {"small", "wanted"}
    assert big.moves == ["cpu", "meta"], "one eviction, the one that freed the most"
    assert small.moves == ["cpu"]


def test_a_running_stage_never_loses_what_it_holds(monkeypatch):
    """Nesting: the stage inside cannot evict the weights the stage around it runs on."""
    mm = MemoryManager("cpu", "meta")
    dit, vae = TrackedModule(_mib(4)), TrackedModule(_mib(1))
    mm.register("dit", dit)
    mm.register("vae", vae)

    _capacity(monkeypatch, mm, _mib(5))
    with mm.stage("dit", activations=_mib(1)):
        with mm.stage("vae", activations=_mib(16)):   # more than the device holds
            assert mm.resident() == {"dit", "vae"}
            assert dit.moves == ["cpu"]
    assert dit.moves == ["cpu"], "and it was still there when the inner stage ended"


def test_a_claim_the_device_cannot_meet_still_loads(monkeypatch):
    """The manager gives up everything it may and lets the allocator make the call."""
    mm = MemoryManager("cpu", "meta")
    dit, other = TrackedModule(_mib(1)), TrackedModule(_mib(1))
    mm.register("dit", dit)
    mm.register("other", other)

    _capacity(monkeypatch, mm, 0)
    with mm.stage("dit", activations=0):
        assert "dit" in mm.resident()
    assert "other" not in mm.resident()


def test_a_stage_with_no_component_still_claims_room(monkeypatch):
    """A consumer that owns its own weights says so with bytes, not with a name."""
    mm = MemoryManager("cpu", "meta")
    idle = TrackedModule(_mib(4))
    mm.register("idle", idle)

    _capacity(monkeypatch, mm, _mib(20))
    with mm.stage("idle", activations=_mib(1)):
        pass
    assert mm.resident() == {"idle"}, "a stage that needs nothing leaves well enough"

    with mm.stage(activations=_mib(18)):
        assert mm.resident() == set()
        assert mm.resident() == set()


# ------------------------------------------------------------------- squeeze
def test_squeeze_hands_over_the_device_except_what_is_held(monkeypatch):
    """The recovery a stage takes when its own claim turned out too small."""
    mm = MemoryManager("cpu", "meta")
    vae, dit, te = (TrackedModule(_mib(1)), TrackedModule(_mib(4)),
                    TrackedModule(_mib(8)))
    mm.register("vae", vae)
    mm.register("dit", dit)
    mm.register("text_encoder", te)

    _capacity(monkeypatch, mm, _mib(20480))
    with mm.stage("vae", activations=_mib(1)):
        mm.ensure("dit", "text_encoder")
        freed = mm.squeeze(held="vae")

    assert freed == _mib(12)
    assert mm.resident() == {"vae"}, "the fixed one was already exempt, the held one named"


def test_squeeze_reports_nothing_to_take(monkeypatch):
    """Zero tells the caller a retry would fail the same way."""
    mm = MemoryManager("cpu", "meta")
    vae = TrackedModule(_mib(1), device="cpu")
    mm.register("vae", vae)
    _capacity(monkeypatch, mm, _mib(20480))
    assert mm.squeeze(held="vae") == 0


# ------------------------------------------------------------------ headroom
def test_a_device_that_will_not_report_a_size_has_no_headroom():
    """Non-CUDA backends have no figure to quote, and no policy is invented for them."""
    assert MemoryManager("cpu", "meta").free_bytes() is None


def test_an_unmeasurable_device_is_never_asked_to_move(monkeypatch):
    """Without a figure there is no policy: claims go unenforced, squeeze takes nothing."""
    mm = MemoryManager("cpu", "meta")
    dit, other = TrackedModule(_mib(1)), TrackedModule(_mib(1))
    mm.register("dit", dit)
    mm.register("other", other)
    monkeypatch.setattr(mm, "free_bytes", lambda: None)

    with mm.stage("dit", activations=1 << 40):
        assert mm.resident() == {"dit"}
    assert other.moves == []
    assert mm.squeeze() == 0


def test_weight_bytes_counts_a_component_as_it_sits_in_memory():
    """Placement reads bytes, so a quantized component must not be counted as its file."""
    mm = MemoryManager("cpu", "meta")
    mm.register("fp32", TrackedModule(_mib(3)))
    assert mm.weight_bytes("fp32") == _mib(3)
    assert mm.weight_bytes("absent") == 0
