"""``thenoise.multigpu``: parsing, refusals, unit discovery, assignment, wrapping and pinning
(no GPU).

The wrapping tests run on cpu alone: a split over one device must leave results unchanged,
and the per-forward cache must hand the same copy to every unit that asks.
"""
from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from thenoise import multigpu
from conftest import resolves_to
import thenoise.models as model_catalog
from thenoise.memory import MemoryManager
from thenoise.runtime import ModelPaths, Runtime, Settings


class _Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, temb=None):
        return self.lin(x) + (temb if temb is not None else 0)


class _ToyDiT(nn.Module):
    def __init__(self, n=6, dim=8):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(n))
        self.inner = nn.Module()
        self.inner.more = nn.ModuleList(_Block(dim) for _ in range(2))
        self.final = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        temb = torch.ones_like(x)
        x = self.embed(x)
        for b in self.blocks:
            x = b(x, temb=temb)
        for b in self.inner.more:
            x = b(x)
        return self.final(x) * self.scale


CPU = torch.device("cpu")
D0, D1, D2 = (torch.device("cuda", i) for i in range(3))


def test_parse_devices(monkeypatch):
    assert multigpu.parse_devices("").devices == []
    assert multigpu.parse_devices("cuda:0, cuda:1").devices == [D0, D1]
    with pytest.raises(ValueError):
        multigpu.parse_devices("cuda:0,cuda:0")
    # a bare "cuda" is the current device, not always index 0
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert multigpu.parse_devices("cuda,cuda:0").devices == [D1, D0]


def test_device_caps_belong_to_their_spec():
    spec = multigpu.parse_devices("cuda:0@1.5,cuda:1")
    assert spec.devices == [D0, D1]
    assert spec.caps == {D0: int(1.5 * (1 << 30))}
    assert multigpu.parse_devices("cuda:0,cuda:1").caps == {}      # nothing carried over


def test_a_cap_limits_a_budget(monkeypatch):
    monkeypatch.setattr(multigpu, "_available", lambda device: 100 << 30)
    caps = {D0: 6 << 30}
    budgets = multigpu._budgets(nn.Linear(2, 2), [D0, D1], caps)
    assert budgets[0] == 6 << 30 and budgets[1] > 6 << 30


def test_parse_placement():
    assert multigpu.parse_placement("dit=split, text-encoder=cuda:1") == {
        "dit": "split", "text_encoder": "cuda:1"}
    with pytest.raises(ValueError):
        multigpu.parse_placement("dit")


@pytest.mark.parametrize("devices, place, message", [
    ("", "dit=split", "--place needs --devices"),
    ("", "dit", "bad --place entry"),                       # malformed is refused, not skipped
    ("cuda:0,cuda:1", "text_encoder=cuda:2", "not in --devices"),
    ("cuda:0,cpu", "", "wrong image"),
    ("cuda:0", "dit=split", "two or more"),
    ("cuda:0,cuda:0", "", "twice"),
])
def test_bad_stacking_is_refused_before_anything_loads(devices, place, message):
    with pytest.raises(ValueError, match=message):
        Runtime(Settings(devices=devices, placement=place))


def test_a_dit_on_cpu_is_refused_before_it_loads(monkeypatch, fake_model_cls):
    constructed: list = []
    from thenoise.models.base import DiffusionModel
    cls = fake_model_cls(constructed=constructed, REQUIRED_COMPONENTS=(),
                         dit_roles=classmethod(DiffusionModel.dit_roles.__func__))
    monkeypatch.setattr(model_catalog, "resolve", resolves_to(cls))
    from thenoise.models.config import ModelConfig
    monkeypatch.setattr(ModelConfig, "all_headers", lambda self: {role: None for role in self.paths})
    runtime = Runtime(Settings(devices="cuda:0,cuda:1", placement="dit=cpu"))
    with pytest.raises(ValueError, match="dit=cpu"):
        runtime.load(ModelPaths("dit"))
    assert constructed == []
    low = Runtime(Settings(devices="cuda:0,cuda:1", placement="dit_low=cpu"))
    with pytest.raises(ValueError, match="dit_low=cpu"):
        low.load(ModelPaths("dit", dit_low_path="low"))


def test_a_text_encoder_may_go_to_cpu():
    spec = multigpu.parse_stack("cuda:0,cuda:1", "text-encoder=cpu,vae=cuda:1")
    assert spec.placement == {"text_encoder": "cpu", "vae": "cuda:1"}
    multigpu.check_dit_placement(spec, ["dit"])


def test_default_placement():
    plan = multigpu.default_placement(["dit", "dit_low"], ["dit", "dit_low", "text_encoder", "vae"],
                                      [D0, D1])
    assert plan == {"dit": "split", "dit_low": "split", "text_encoder": "cuda:1"}
    assert multigpu.default_placement(["dit"], ["dit"], [D0]) == {}


def test_without_devices_nothing_is_touched(monkeypatch, fake_model_cls):
    """The contract for everyone not using --devices: no hooks, no spreads, same device."""
    class Model:
        def __init__(self, **_):
            self.memory = MemoryManager("cpu", "cpu")
            self.dit, self.vae = _ToyDiT(), _Vae()
            self.memory.register("dit", self.dit)
            self.memory.register("vae", self.vae)

    cls = fake_model_cls(REQUIRED_COMPONENTS=(), __init__=lambda self, **kw: Model.__init__(self))
    monkeypatch.setattr(model_catalog, "resolve", resolves_to(cls))
    settings = Settings(device="cpu", offload_device="cpu")
    runtime = Runtime(settings)
    runtime.load(ModelPaths("dit"))
    model = runtime.model
    assert runtime._spreads == {} and settings.device == "cpu" and runtime.device == "cpu"
    for module in (*model.dit.modules(), *model.vae.modules()):
        assert not module._forward_hooks and not module._forward_pre_hooks
        assert "forward" not in module.__dict__
    assert not model.memory._pinned


def test_split_units_opens_module_lists_in_order():
    names = [(n, b) for n, _u, b in multigpu.split_units(_ToyDiT())]
    assert names == [("embed", False)] + [(f"blocks.{i}", True) for i in range(6)] + [
        ("inner.more.0", True), ("inner.more.1", True), ("final", False)]


def _blocks(units, where):
    return [w for (_, _, b), w in zip(units, where) if b]


def test_assign_fills_in_order_and_keeps_glue_on_primary():
    units = multigpu.split_units(_ToyDiT())
    block = multigpu._weight_bytes(units[1][1])
    glue = sum(multigpu._weight_bytes(u) for _, u, b in units if not b)
    # the primary's glue comes out of its budget: shares of 2, 3 and 3 of the 8 blocks
    where = multigpu.assign(units, [D0, D1, D2], [2 * block + glue, 3 * block, 3 * block])
    assert _blocks(units, where) == [D0, D0, D1, D1, D1, D2, D2, D2]
    # plenty of room everywhere: still spread in proportion, not all on the first device
    where = multigpu.assign(units, [D0, D1], [100 * block, 100 * block])
    assert _blocks(units, where).count(D0) == 4 and _blocks(units, where).count(D1) == 4
    assert where[0] == D0 and where[-1] == D0             # embed and final on primary


def test_assign_budgets_what_it_places(caplog):
    units = multigpu.split_units(_ToyDiT())
    block = multigpu._weight_bytes(units[1][1])
    glue = sum(multigpu._weight_bytes(u) for _, u, b in units if not b)
    with caplog.at_level("WARNING"):
        where = multigpu.assign(units, [D0, D1], [block, block])
    assert _blocks(units, where).count(D1) >= 6           # the last device takes the rest
    assert "may run out" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):                      # blocks fit; only the glue tips it over
        multigpu.assign(units, [D0, D1], [4 * block + glue, 4 * block])
    assert caplog.text == ""


def test_spread_on_one_device_keeps_results_and_shares_copies():
    torch.manual_seed(0)
    model = _ToyDiT().eval()
    x = torch.randn(3, 8)
    with torch.no_grad():
        want = model(x)
    sp = multigpu.spread(model, [CPU], budgets=[1 << 40])
    try:
        with torch.no_grad():
            got = model(x)
        torch.testing.assert_close(got, want)
        assert "forward" in model.blocks[0].__dict__
    finally:
        sp.remove()
    assert "forward" not in model.__dict__ and "forward" not in model.blocks[0].__dict__


def test_shared_cache_copies_once():
    cache = multigpu._Shared()
    t = torch.ones(4)
    dev = torch.device("meta")
    a = multigpu._to({"x": t, "y": [t]}, dev, cache)
    assert a["x"] is a["y"][0]                     # one copy, reused
    assert len(cache.copies) == 1
    u = torch.ones(4)                              # another tensor: never the cached copy
    b = multigpu._to(u, dev, cache)
    assert b is not a["x"] and len(cache.copies) == 2


def test_a_mapping_is_copied_not_edited():
    class Out(dict):
        def __init__(self, a=None):                # cannot be rebuilt from its items
            super().__init__(a=a)

    src = Out(a=torch.ones(2))
    moved = multigpu._to(src, torch.device("meta"), None)
    assert moved is not src and moved["a"].device.type == "meta"
    assert src["a"].device.type == "cpu"


def test_memory_manager_pin_blocks_moves():
    mm = MemoryManager("cpu", "meta")
    comp = nn.Linear(4, 4)
    mm.register("dit", comp)
    weight = mm.weight_bytes("dit")
    mm.pin("dit")
    assert mm.weight_bytes("dit") == weight        # still reports what it occupies
    mm.offload("dit")
    assert comp.weight.device.type == "cpu"        # pinned: offload left it alone
    with pytest.raises(KeyError):
        mm.pin("nope")


def test_apply_pins_what_it_moves(caplog):
    class Model:
        memory = MemoryManager("cpu", "meta")

    model = Model()
    model.memory.register("dit", _ToyDiT())
    model.memory.register("vae", _Vae())
    done = multigpu.apply(model, multigpu.DeviceSpec(devices=[CPU]),
                          {"dit": "split", "vae": "cpu", "text_encoder": "cpu"})
    assert set(done) == {"dit"}                    # vae is already on the primary
    assert model.memory._pinned == {"dit"}
    assert "no component 'text_encoder'" in caplog.text


class _Vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.dec = nn.Linear(4, 4)

    def forward(self, x):
        return self.dec(self.enc(x))

    def decode(self, z):
        return self.dec(z)

    def decode_frames(self, z):
        for _ in range(3):
            yield self.dec(z)


def test_place_wraps_public_methods_and_undoes():
    vae = _Vae().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        want_f, want_d = vae(x), vae.decode(x)
    sp = multigpu.place(vae, CPU, CPU)
    with torch.no_grad():
        torch.testing.assert_close(vae(x), want_f)
        torch.testing.assert_close(vae.decode(x), want_d)
    assert "decode" in vae.__dict__ and not vae.enc.__dict__.get("forward")   # no child hooks
    assert vae.dec._forward_hooks == {} and vae.dec._forward_pre_hooks == {}
    sp.remove()
    assert "decode" not in vae.__dict__ and "forward" not in vae.__dict__


def test_place_keeps_a_generator_entry_point_lazy_and_clears_per_item():
    vae = _Vae().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        want = list(vae.decode_frames(x))
    multigpu.place(vae, CPU, CPU)
    with torch.no_grad():
        out = vae.decode_frames(x)
        assert inspect.isgenerator(out)
        got = list(out)
    assert len(got) == 3
    for g in got:
        torch.testing.assert_close(g, want[0])


def test_a_generator_steps_on_its_device_and_clears_the_cache_per_item(monkeypatch):
    depth = []

    class Current:
        def __init__(self, device):
            pass

        def __enter__(self):
            depth.append(1)

        def __exit__(self, *exc):
            depth.pop()

    monkeypatch.setattr(torch.cuda, "device", Current)
    cache = multigpu._Shared()
    seen = []

    def frames():
        for i in range(3):
            seen.append(len(depth))                # current while the body runs
            cache.copies[(i, D1)] = (None, None)   # stands in for a copy made this step
            yield i

    out = multigpu._entry(frames, D1, None, clear=cache)()
    assert inspect.isgenerator(out) and seen == []
    for item in out:
        assert depth == [] and cache.copies == {}   # not current between items; cleared
    assert seen == [1, 1, 1]


def test_the_current_device_is_restored_when_a_unit_raises(monkeypatch):
    current = [0]

    class Current:
        def __init__(self, device):
            self.device = device.index

        def __enter__(self):
            self.prev, current[0] = current[0], self.device

        def __exit__(self, *exc):
            current[0] = self.prev

    monkeypatch.setattr(torch.cuda, "device", Current)

    def boom():
        assert current[0] == 1
        raise torch.cuda.OutOfMemoryError("retried by the memory manager")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        multigpu._entry(boom, D1, D0)()
    assert current[0] == 0
    multigpu._entry(lambda: None, D1, D0)()          # and the retry starts from there
    assert current[0] == 0


_REQUIRED = ["--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "p"]


@pytest.mark.parametrize("command, takes", [
    (["serve"], True), (["generate", *_REQUIRED], True),
    (["edit", *_REQUIRED, "--image", "i"], True), (["video", *_REQUIRED], True),
    (["mesh", "--dit", "d", "--image", "i"], True),
    (["upscale", "--pixel-upscaler", "u", "--input", "i"], False), (["controlmap"], False),
])
def test_only_commands_that_stack_take_the_flags(command, takes):
    from thenoise.cli import build_parser
    args = build_parser().parse_args(command)
    assert (hasattr(args, "devices") and hasattr(args, "place")) == takes


class _NestedVae(nn.Module):
    """Public methods that call each other, as real VAEs do (decode -> normalize)."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def normalize(self, z):
        return z * 2

    def decode(self, z):
        return self.lin(self.normalize(z))

    def frames(self, z):
        for i in range(2):
            yield self.decode(z + i)


def _count_moves(monkeypatch):
    calls = []
    real = multigpu._to

    def counting(obj, device, cache):
        calls.append(device)
        return real(obj, device, cache)

    monkeypatch.setattr(multigpu, "_to", counting)
    return calls


def test_place_nested_entry_points_move_once(monkeypatch):
    vae = _NestedVae().eval()
    cpu = torch.device("cpu")
    sp = multigpu.place(vae, cpu, cpu)
    calls = _count_moves(monkeypatch)
    with torch.no_grad():
        vae.decode(torch.randn(2, 4))
    assert len(calls) == 4          # args (+ its tensor) and kwargs in, result home: outer call only
    calls.clear()
    with torch.no_grad():
        out = list(vae.frames(torch.randn(2, 4)))
    assert len(out) == 2 and len(calls) == 3 + 2   # in once, each item home once
    sp.remove()
