"""``MemoryManager``: placement under pressure, residency by default (no GPU).

Device moves are exercised with cpu (load) and meta (offload). A meta tensor reports
its byte size and can travel in either direction, which is what lets the residency
rules be tested where there is no device — see ``conftest.TrackedModule``.
"""
from __future__ import annotations

import torch
from torch import nn

import pytest
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


def _reserve(monkeypatch, base: int, per_megapixel: int = 0) -> None:
    """The stage reserve, so the sizes below are about placement and not about scale."""
    monkeypatch.setattr("thenoise.memory._RESERVE_BASE_BYTES", base)
    monkeypatch.setattr("thenoise.memory._RESERVE_BYTES_PER_MEGAPIXEL", per_megapixel)


def _capacity(monkeypatch, mm: MemoryManager, capacity: int) -> None:
    """Free memory as a device reports it: capacity minus what is resident right now."""
    monkeypatch.setattr(
        mm, "free_bytes",
        lambda: capacity - sum(mm.weight_bytes(n) for n in mm.resident()),
    )


# ------------------------------------------------------------ resident mode
def test_resident_mode_is_noop(monkeypatch):
    m = _Comp()
    mm = MemoryManager("cpu", "cpu")  # offload == load -> no moves
    assert not mm.offloads
    mm.register("comp", m)
    _reserve(monkeypatch, 0)
    with mm.stage("comp", pixels=0):
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


def test_a_stage_moves_in_once_and_offload_moves_out():
    m = _MockComp()
    mm = MemoryManager("cpu", "meta")
    mm.register("comp", m)  # no params -> not resident
    assert "comp" not in mm.resident()

    with mm.stage("comp", pixels=0):
        with mm.stage("comp", pixels=0):   # nested: already where it needs to be
            pass
    assert m.to_calls == ["cpu"]
    assert "comp" in mm.resident()

    mm.offload("comp")
    assert m.to_calls == ["cpu", "meta"]
    assert "comp" not in mm.resident()


def test_missing_component_is_noop():
    mm = MemoryManager("cpu", "meta")
    with mm.stage("nope", pixels=0):
        pass
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

    with mm.stage("embedder", pixels=0):
        assert swappable.to_calls == ["cpu"] and "embedder" in mm.resident()

    mm.offload("embedder")
    assert swappable.to_calls == ["cpu", "meta"]

    assert mm.resident() == {"onboard"}, "it reported the load device, so it counts"

    mm.offload("onboard")
    assert arrived.to_calls == ["meta"], "and it is no more pinned than anything else"


# --------------------------------------------------------------- residency
def test_leaving_a_stage_moves_nothing(monkeypatch):
    """Residency is where things are left, not a state somebody grants."""
    mm = MemoryManager("cpu", "meta")
    dit = TrackedModule(1024)
    mm.register("dit", dit)

    _reserve(monkeypatch, _mib(1))
    with mm.stage("dit", pixels=0):
        assert mm.resident() == {"dit"}
    assert dit.moves == ["cpu"]
    assert mm.resident() == {"dit"}, "the next stage gets it without a reload"


def test_arriving_on_the_device_does_not_pin_a_component(monkeypatch):
    """Where the initial copy landed is a load-time choice, not a lifetime commitment."""
    mm = MemoryManager("cpu", "meta")
    vae, dit = TrackedModule(_mib(1), device="cpu"), TrackedModule(_mib(4))
    mm.register("vae", vae)
    mm.register("dit", dit)

    _capacity(monkeypatch, mm, _mib(5))
    _reserve(monkeypatch, _mib(2))
    with mm.stage("dit", pixels=0):
        assert mm.resident() == {"dit"}
    assert vae.moves == ["meta"], "idle, so it paid — coming up on the device aside"


def test_an_idle_weight_set_pays_for_a_claim_it_cannot_beside(monkeypatch):
    """Largest first, and only when the device genuinely cannot hold both."""
    mm = MemoryManager("cpu", "meta")
    small, big, wanted = (TrackedModule(_mib(1)), TrackedModule(_mib(4)),
                          TrackedModule(_mib(1)))
    mm.register("small", small)
    mm.register("big", big)
    mm.register("wanted", wanted)

    _capacity(monkeypatch, mm, _mib(20))
    _reserve(monkeypatch, _mib(1))
    with mm.stage("small", "big", pixels=0):
        pass
    assert mm.resident() == {"small", "big"}, "there was room; nothing was disturbed"

    _capacity(monkeypatch, mm, _mib(5))      # one eviction's worth short
    with mm.stage("wanted", pixels=0):
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
    _reserve(monkeypatch, _mib(1))
    with mm.stage("dit", pixels=0):
        _reserve(monkeypatch, _mib(16))   # more than the device holds
        with mm.stage("vae", pixels=0):
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
    _reserve(monkeypatch, 0)
    with mm.stage("dit", pixels=0):
        assert "dit" in mm.resident()
    assert "other" not in mm.resident()


def test_a_stage_with_no_component_still_claims_room(monkeypatch):
    """A consumer that owns its own weights says so with bytes, not with a name."""
    mm = MemoryManager("cpu", "meta")
    idle = TrackedModule(_mib(4))
    mm.register("idle", idle)

    _capacity(monkeypatch, mm, _mib(20))
    _reserve(monkeypatch, _mib(1))
    with mm.stage("idle", pixels=0):
        pass
    assert mm.resident() == {"idle"}, "a stage that needs nothing leaves well enough"

    _reserve(monkeypatch, _mib(18))
    with mm.stage(pixels=0):
        assert mm.resident() == set()
        assert mm.resident() == set()


# ---------------------------------------------------------------- admission
def _free(monkeypatch, nbytes) -> None:
    """What the driver would hand out on the load device."""
    monkeypatch.setattr("thenoise.memory.get_free_memory", lambda _dev: nbytes)


def test_weights_are_admitted_while_the_stage_floor_survives(monkeypatch):
    """The one figure admission keeps free is the fixed part of a stage's reserve."""
    from thenoise import memory as memory_module

    floor = memory_module._RESERVE_BASE_BYTES
    mm = MemoryManager("cpu", "meta")
    _free(monkeypatch, floor + 5 * (1 << 30))

    assert mm.device_for(5 * (1 << 30)) == mm.load_device, "exactly the floor left"
    assert mm.device_for(5 * (1 << 30) + 1) == mm.offload_device


def test_a_device_that_gives_no_free_figure_admits(monkeypatch):
    """No figure, no policy: build where the work happens and let the allocator judge."""
    mm = MemoryManager("cpu", "meta")
    _free(monkeypatch, None)
    assert mm.device_for(1 << 60) == mm.load_device


def test_resident_mode_admits_everything_without_asking(monkeypatch):
    """With no offload target there is nothing to weigh, and the driver is not queried."""
    def never(_dev):
        raise AssertionError("resident mode has no second device to compare against")

    monkeypatch.setattr("thenoise.memory.get_free_memory", never)
    assert MemoryManager("cpu", "cpu").device_for(1 << 60) == torch.device("cpu")


# ------------------------------------------------------ measured headroom
def _reserves(mm: MemoryManager) -> list[int]:
    """The room each stage entry asked for, in order."""
    seen: list[int] = []
    original = mm._reserve_for

    def spy(names, pixels):
        room = original(names, pixels)
        seen.append(room)
        return room

    mm._reserve_for = spy
    return seen


def _measures(monkeypatch, transient: int) -> list:
    """A device that reports peaks: every measured stage is seen using ``transient``."""
    begins: list[int] = []

    def begin(device):
        begins.append(1)
        return 0

    monkeypatch.setattr(MemoryManager, "_peak_begin", staticmethod(begin))
    monkeypatch.setattr(MemoryManager, "_peak_end", staticmethod(lambda device, base: transient))
    return begins


def test_a_stage_reserves_what_it_used_last_time(monkeypatch):
    """The fleet figure is a floor the device's own measurement overrides."""
    _reserve(monkeypatch, 0, per_megapixel=_mib(1))       # 1 MiB to the megapixel
    _measures(monkeypatch, _mib(6))
    mm = MemoryManager("cpu", "meta")
    asked = _reserves(mm)

    with mm.stage("dit", pixels=2_000_000):
        pass
    assert asked == [_mib(2)], "the first one runs on the guess"

    with mm.stage("dit", pixels=2_000_000):
        pass
    assert asked == [_mib(2), _mib(6)]


def test_a_measurement_teaches_one_stage_one_size(monkeypatch):
    """Nothing is extrapolated across stages, up to the next size, or down to the floor."""
    _reserve(monkeypatch, 0, per_megapixel=_mib(1))
    _measures(monkeypatch, _mib(6))
    mm = MemoryManager("cpu", "meta")
    asked = _reserves(mm)

    with mm.stage("vae", pixels=4_000_000):     # learns 6 MiB at its own bucket
        pass
    with mm.stage("vae", pixels=4_000_000):     # same bucket
        pass
    with mm.stage("vae", pixels=3_000_000):     # same bucket again
        pass
    with mm.stage("dit", pixels=4_000_000):     # another stage
        pass
    with mm.stage("vae", pixels=16_000_000):    # far above what was measured
        pass

    assert asked == [_mib(4), _mib(6), _mib(6), _mib(4), _mib(16)]


def test_a_peak_under_the_fleet_figure_changes_nothing(monkeypatch):
    """Other devices and stacks are still counted: the figure is a floor, not a target."""
    _reserve(monkeypatch, 0, per_megapixel=_mib(1))
    _measures(monkeypatch, _mib(1))
    mm = MemoryManager("cpu", "meta")
    asked = _reserves(mm)

    for _ in range(2):
        with mm.stage("dit", pixels=4_000_000):
            pass
    assert asked == [_mib(4), _mib(4)]


def test_a_device_that_keeps_its_peak_silent_teaches_nothing(monkeypatch):
    """No reading, no learning: the fleet figure stands, as it does on cpu."""
    _reserve(monkeypatch, 0, per_megapixel=_mib(1))
    monkeypatch.setattr(MemoryManager, "_peak_begin", staticmethod(lambda device: None))
    mm = MemoryManager("cpu", "meta")
    asked = _reserves(mm)

    for _ in range(2):
        with mm.stage("dit", pixels=4_000_000):
            pass
    assert asked == [_mib(4), _mib(4)]


def test_a_nested_stage_is_measured_with_the_one_around_it(monkeypatch):
    """One peak counter, so only the outer window is a reading of anything."""
    _reserve(monkeypatch, 0, per_megapixel=_mib(1))
    begins = _measures(monkeypatch, _mib(9))
    mm = MemoryManager("cpu", "meta")
    asked = _reserves(mm)

    with mm.stage("dit", pixels=2_000_000):
        with mm.stage("vae", pixels=2_000_000):
            pass

    assert len(begins) == 1
    with mm.stage("dit", pixels=2_000_000):
        pass
    with mm.stage("vae", pixels=2_000_000):
        pass
    assert asked[-2:] == [_mib(9), _mib(2)], "the window belongs to the outer stage"


def test_a_stage_measures_only_itself_and_not_the_weights_under_it(monkeypatch):
    """The baseline is what was allocated on the way in: weights, caches, everything."""
    _reserve(monkeypatch, 0, per_megapixel=_mib(1))
    monkeypatch.setattr(MemoryManager, "_peak_begin", staticmethod(lambda device: 100))
    seen: list[int] = []
    monkeypatch.setattr(  # as the real one does: the peak, less what was live on entry
        MemoryManager, "_peak_end",
        staticmethod(lambda device, base: seen.append(base) or 600 - base))
    mm = MemoryManager("cpu", "meta")

    with mm.stage("dit", pixels=2_000_000):
        pass
    assert seen == [100] and mm._peaks[("dit",)][2] == 500


# ------------------------------------------------------- running out of room
class _RunsShort:
    """A stage body the device runs out of memory inside, once."""

    def __init__(self):
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls == 1:
            raise torch.cuda.OutOfMemoryError("no room")
        return "pixels"


def test_a_stage_that_runs_out_hands_over_the_device_and_runs_again(monkeypatch):
    """The recovery belongs to whoever holds the reserve, so it belongs here."""
    mm = MemoryManager("cpu", "meta")
    vae, dit = TrackedModule(_mib(1)), TrackedModule(_mib(4))
    mm.register("vae", vae)
    mm.register("dit", dit)
    _capacity(monkeypatch, mm, _mib(10))
    _reserve(monkeypatch, _mib(1))

    with mm.stage("dit", pixels=0):
        pass                             # left resident, as residency is sticky

    work = _RunsShort()
    with mm.stage("vae", pixels=0) as s:  # the reserve held; the body needed more
        assert s.run(work) == "pixels"
    assert work.calls == 2
    assert "dit" not in mm.resident(), "the idle weight set paid for the retry"
    assert vae.moves == ["cpu"], "a stage never loses what it is running"


def test_a_stage_with_nothing_left_to_take_does_not_retry(monkeypatch):
    """No room to be had: the error stands rather than failing the same way twice."""
    mm = MemoryManager("cpu", "meta")
    vae = TrackedModule(_mib(1), device="cpu")   # the only resident thing, and held
    mm.register("vae", vae)
    _capacity(monkeypatch, mm, _mib(10))
    _reserve(monkeypatch, _mib(1))

    work = _RunsShort()
    with mm.stage("vae", pixels=0) as s:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            s.run(work)
    assert work.calls == 1


def test_the_reserve_is_a_fixed_part_plus_a_part_per_pixel():
    """Both figures live once, in the manager; the stage arithmetic is on them."""
    from thenoise import memory

    flat = memory._RESERVE_BASE_BYTES
    slope = memory._RESERVE_BYTES_PER_MEGAPIXEL
    assert memory.reserve_bytes(0) == flat
    assert memory.reserve_bytes(1_000_000) == flat + slope
    assert memory.reserve_bytes(4_000_000) == flat + 4 * slope


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
    _reserve(monkeypatch, 1 << 40)          # a claim no device could meet

    with mm.stage("dit", pixels=0):
        assert mm.resident() == {"dit"}
    assert other.moves == []
    assert mm.squeeze() == 0


def test_weight_bytes_counts_a_component_as_it_sits_in_memory():
    """Placement reads bytes, so a quantized component must not be counted as its file."""
    mm = MemoryManager("cpu", "meta")
    mm.register("fp32", TrackedModule(_mib(3)))
    assert mm.weight_bytes("fp32") == _mib(3)
    assert mm.weight_bytes("absent") == 0
