"""``MemoryManager``: placement under pressure, residency by default (no GPU).

Device moves are exercised with cpu (load) and meta (offload). A meta tensor reports
its byte size and can travel in either direction, which is what lets the residency
rules be tested where there is no device — see ``conftest.TrackedModule``.
"""
from __future__ import annotations

import torch
from torch import nn

import pytest
from conftest import TrackedModule, cuda_only, stub_room
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
def test_weights_are_admitted_while_the_stage_floor_survives(monkeypatch):
    """The one figure admission keeps free is the fixed part of a stage's reserve."""
    from thenoise import memory as memory_module

    floor = memory_module._RESERVE_BASE_BYTES
    mm = MemoryManager("cuda", "meta")
    stub_room(monkeypatch, floor + 5 * (1 << 30))

    assert mm.device_for(5 * (1 << 30)) == mm.load_device, "exactly the floor left"
    assert mm.device_for(5 * (1 << 30) + 1) == mm.offload_device


def test_a_load_admits_into_the_room_its_own_cache_leaves(monkeypatch):
    """The bytes our allocator holds unused are room: load-time scratch must not exile a
    component that fits.
    """
    from thenoise import memory as memory_module

    mm = MemoryManager("cuda", "meta")
    stub_room(monkeypatch, memory_module._RESERVE_BASE_BYTES, reserved=_mib(512))
    assert mm.device_for(_mib(512)) == mm.load_device


def test_a_device_that_gives_no_free_figure_admits(monkeypatch):
    """No figure, no policy: build where the work happens and let the allocator judge."""
    mm = MemoryManager("cuda", "meta")
    stub_room(monkeypatch, None)
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


def test_a_device_named_without_its_index_holds_its_weights_resident(monkeypatch):
    """Read ``cuda`` and ``cuda:0`` as two devices and nothing is ever resident, so
    nothing can ever be evicted.
    """
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    mm = MemoryManager("cuda", "cpu")     # how the adapters name it
    assert mm.load_device == torch.device("cuda", 0)
    mm.register("dit", TrackedModule(_mib(4), device="cuda:0"))   # as a tensor reports it
    assert mm.resident() == {"dit"}
    # Same resolution the other way: naming both devices alike is resident mode.
    assert not MemoryManager("cuda", "cuda:0").offloads


@cuda_only
def test_a_card_of_the_reported_size_swaps_the_encoder_for_the_dit(monkeypatch):
    """6.8 GiB DiT + 8.7 GiB encoder + 0.6 GiB VAE on a 15.5 GiB card: the encoder's
    stage takes the DiT's place and leaves the VAE alone.
    """
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    mm = MemoryManager("cuda", "cpu")
    dit = TrackedModule(_mib(6800), device="cuda:0")
    encoder = TrackedModule(_mib(8700), device="meta")
    vae = TrackedModule(_mib(600), device="cuda:0")
    for name, module in (("dit", dit), ("text_encoder", encoder), ("vae", vae)):
        mm.register(name, module)
    _capacity(monkeypatch, mm, _mib(15480))
    _reserve(monkeypatch, _mib(2048))

    with mm.stage("text_encoder", pixels=0):
        assert dit.moves == ["cpu"], "the DiT was the only thing big enough to displace"
        assert encoder.moves == ["cuda"]
        assert vae.moves == []

    with mm.stage("dit", pixels=0):
        assert encoder.moves == ["cuda", "cpu"], "and the way back takes the encoder"
        assert vae.moves == []


def test_a_stage_counts_from_after_its_own_placement(monkeypatch):
    """What the manager uploaded for a stage is resident, not the stage's transient."""
    order: list[str] = []
    monkeypatch.setattr(
        MemoryManager, "_place", lambda self, names, room: order.append("place"))
    monkeypatch.setattr(
        MemoryManager, "_peak_begin",
        staticmethod(lambda device: order.append("begin") or 0))
    monkeypatch.setattr(
        MemoryManager, "_peak_end",
        staticmethod(lambda device, baseline: order.append("end") or 0))
    mm = MemoryManager("cpu", "meta")
    with mm.stage("vae", pixels=0):
        pass
    assert order == ["place", "begin", "end"]


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


def test_the_retry_squeezes_after_the_failed_attempt_left_the_stack(monkeypatch):
    """The failed attempt's frames are the memory the retry needs: squeeze after them."""
    events: list[str] = []

    class Temporary:
        def __del__(self):
            events.append("released")

    mm = MemoryManager("cpu", "meta")
    monkeypatch.setattr(mm, "squeeze", lambda: events.append("squeezed") or (1 << 30))

    def work() -> str:
        if "squeezed" in events:
            return "pixels"
        temporary = Temporary()
        raise torch.cuda.OutOfMemoryError(f"no room beside {temporary!r}")

    with mm.stage("vae", pixels=0) as s:
        assert s.run(work) == "pixels"
    assert events == ["released", "squeezed"]


def test_an_inner_stage_leaves_the_outer_stays_hold_intact(monkeypatch):
    """An inner stage that names a held component must not hand it over on its way out."""
    mm = MemoryManager("cpu", "meta")
    dit, vae = TrackedModule(_mib(4), device="cpu"), TrackedModule(_mib(1), device="cpu")
    mm.register("dit", dit)
    mm.register("vae", vae)
    _reserve(monkeypatch, 0)

    with mm.stage("dit", "vae", pixels=0):
        with mm.stage("vae", pixels=0):
            pass
        mm.squeeze()
        assert vae.moves == [], "the outer stage is still running with the VAE"


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


def test_free_bytes_counts_the_other_tenants_and_ones_own_cache(monkeypatch):
    """Driver free counts another process's weights; the cache is still ours to reuse."""
    stub_room(monkeypatch, _mib(40_000), reserved=_mib(4_000), live=_mib(2_000))
    assert MemoryManager("cuda", "cpu").free_bytes() == _mib(42_000)


def test_free_bytes_without_a_driver_figure_is_the_device_total_minus_live_bytes(monkeypatch):
    stub_room(monkeypatch, None, total=_mib(10_000), live=_mib(3_000))
    assert MemoryManager("cuda", "cpu").free_bytes() == _mib(7_000)


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


class _Wrapper:
    """A plain object holding a model: no ``parameters``, forwards placement by hand."""

    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model.to(device)
        return self


def test_a_plain_wrapper_weighs_what_the_model_it_holds_weighs():
    """A wrapper counted at zero asks no room for its upload and is last to be traded."""
    mm = MemoryManager("cpu", "meta")
    mm.register("encoder", _Wrapper(TrackedModule(_mib(5))))
    assert mm.weight_bytes("encoder") == _mib(5)


def test_a_quantized_component_weighs_its_storage_not_its_logical_dtype():
    """The loader's int8 weights report the bf16 they stand in for: double the room they
    use.
    """
    from thenoise.utils.loader import _build_int8_qt

    rows, cols = 2048, 1024
    quantized = _build_int8_qt(torch.zeros(rows, cols, dtype=torch.int8), torch.ones(1), {})
    assert quantized.element_size() == 2      # bf16: not what it occupies
    holder = nn.Module()
    holder.register_buffer("weight", quantized)

    mm = MemoryManager("cpu", "meta")
    mm.register("int8", holder)
    assert mm.weight_bytes("int8") == rows * cols + quantized.params.scale.nbytes


def test_weight_bytes_deduplicates_shared_quantized_views_and_counts_scales():
    """Fused QKV owns one payload and one row-scale allocation, not four copies."""
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    rows, cols = 6, 8
    qdata = torch.arange(rows * cols, dtype=torch.int8).reshape(rows, cols)
    scale = torch.ones(rows, 1, dtype=torch.float32)
    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=(rows, cols),
        is_weight=True,
        convrot=True,
        convrot_groupsize=256,
    )
    fused = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
    parts = []
    for start in range(0, rows, 2):
        part = nn.Module()
        part.register_buffer(
            "weight",
            QuantizedTensor(
                qdata.narrow(0, start, 2),
                "TensorWiseINT8Layout",
                TensorWiseINT8Layout.Params(
                    scale=scale.narrow(0, start, 2),
                    orig_dtype=torch.bfloat16,
                    orig_shape=(2, cols),
                    is_weight=True,
                    convrot=True,
                    convrot_groupsize=256,
                ),
            ),
        )
        parts.append(part)
    owner = nn.Module()
    owner.register_buffer("weight", fused)
    holder = nn.ModuleList([owner, *parts])

    mm = MemoryManager("cpu", "meta")
    mm.register("qkv", holder)
    assert mm.weight_bytes("qkv") == qdata.nbytes + scale.nbytes


def test_a_component_that_reports_no_weights_says_so(caplog):
    """A component of unknown size cannot be traded for, so it says so."""
    import logging

    with caplog.at_level(logging.WARNING, logger="thenoise.memory"):
        MemoryManager("cpu", "meta").register("mystery", _MockComp())
    assert "reports no weights" in caplog.text


def test_weight_bytes_counts_a_component_as_it_sits_in_memory():
    """Placement reads bytes, so a quantized component must not be counted as its file."""
    mm = MemoryManager("cpu", "meta")
    mm.register("fp32", TrackedModule(_mib(3)))
    assert mm.weight_bytes("fp32") == _mib(3)
    assert mm.weight_bytes("absent") == 0
