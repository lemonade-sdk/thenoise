"""Spread one model over several GPUs so a model larger than one card fits.

The rest of TheNoise keeps one primary device. A *split* component is cut into units (the
blocks of its ``nn.ModuleList``s plus its other children) dealt out to the devices in
forward order; a *placed* component moves whole to one device. Each unit's inputs go to
its device, which is made current while it runs, and outputs come back to the primary
device where code outside the unit reads them. Moved components are pinned in the
:class:`~thenoise.memory.MemoryManager`.
"""
from __future__ import annotations

import copy
import inspect
import logging
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from thenoise.inference import held_modules
from thenoise.memory import (RESERVE_BASE_BYTES, owned_tensors, tensor_bytes,
                             weight_bytes, with_index)
from thenoise.utils.device import get_available_memory

logger = logging.getLogger(__name__)

_GIB = 1 << 30


@dataclass
class DeviceSpec:
    """What ``--devices`` and ``--place`` asked for, checked."""
    devices: List[torch.device] = field(default_factory=list)   # the first is the primary
    caps: Dict[torch.device, int] = field(default_factory=dict)  # bytes a split may use
    placement: Dict[str, str] = field(default_factory=dict)      # component -> device | split

    @property
    def primary(self) -> Optional[torch.device]:
        return self.devices[0] if self.devices else None


def parse_devices(spec: Optional[str]) -> DeviceSpec:
    """``"cuda:0,cuda:1"`` -> devices; the first is the primary. ``cuda:0@6`` caps that
    device's share of a split at 6 GiB. Several devices must all be GPUs."""
    out = DeviceSpec()
    for item in (spec or "").split(","):
        item, _, cap = item.strip().partition("@")
        if not item.strip():
            continue
        try:
            dev = with_index(item.strip())
        except RuntimeError as error:
            raise ValueError(f"bad device {item.strip()!r} in --devices") from error
        if dev in out.devices:
            raise ValueError(f"device {dev} listed twice in --devices")
        if cap.strip():
            out.caps[dev] = int(float(cap) * _GIB)
        out.devices.append(dev)
    if len(out.devices) > 1 and any(d.type == "cpu" for d in out.devices):
        raise ValueError("--devices stacks GPUs; a model split onto cpu gives a wrong "
                         "image (autocast is per device type)")
    return out


def parse_placement(spec: Optional[str]) -> Dict[str, str]:
    """``"text_encoder=cuda:1,vae=cuda:0,dit=split"`` -> ``{component: device | 'split'}``."""
    out: Dict[str, str] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, where = item.partition("=")
        if not sep or not name.strip() or not where.strip():
            raise ValueError(f"bad --place entry {item!r}; expected component=device|split")
        out[name.strip().replace("-", "_")] = where.strip()
    return out


def parse_stack(devices: Optional[str], place: Optional[str]) -> DeviceSpec:
    """Both flags, parsed and checked against each other: what is refused here is
    refused before anything loads."""
    spec = parse_devices(devices)
    spec.placement = parse_placement(place)
    if spec.placement and not spec.devices:
        raise ValueError("--place needs --devices")
    for name, where in spec.placement.items():
        if where == "split":
            if len(spec.devices) < 2:
                raise ValueError(f"--place {name}=split needs two or more --devices")
            continue
        try:
            dev = with_index(where)
        except RuntimeError as error:
            raise ValueError(f"--place {name}: bad device {where!r}") from error
        if dev.type != "cpu" and dev not in spec.devices:
            raise ValueError(f"--place {name}={where}: {dev} is not in --devices")
        spec.placement[name] = str(dev)
    return spec


def check_dit_placement(spec: DeviceSpec, dit_roles: Iterable[str]) -> None:
    """Refuse a DiT placed on cpu: autocast is per device type, so it gives a wrong image."""
    for role in dit_roles:
        where = spec.placement.get(role, "split")
        if where != "split" and torch.device(where).type == "cpu":
            raise ValueError(f"--place {role}=cpu: a DiT on cpu gives a wrong image")


def default_placement(dit_roles: Iterable[str], components: Iterable[str],
                      devices: Sequence[torch.device]) -> Dict[str, str]:
    """Split every DiT and put the text encoder on the last device; the VAE stays on the
    primary device with the latents."""
    plan: Dict[str, str] = {}
    if len(devices) < 2:
        return plan
    dit_roles = set(dit_roles)
    for name in components:
        if name in dit_roles:
            plan[name] = "split"
        elif name == "text_encoder":
            plan[name] = str(devices[-1])
    return plan


def _bytes_on(module: nn.Module, device: torch.device) -> int:
    return sum(tensor_bytes(t) for t in owned_tensors(module) if t.device == device)


def _has_module_list(module: nn.Module) -> bool:
    return any(isinstance(m, nn.ModuleList) for m in module.modules())


def split_units(module: nn.Module) -> List[Tuple[str, nn.Module, bool]]:
    """``(name, unit, is_block)`` in registration order: ModuleList members are blocks,
    children holding a ModuleList are opened up, other children are one unit each."""
    units: List[Tuple[str, nn.Module, bool]] = []

    def walk(prefix: str, m: nn.Module) -> None:
        for name, child in m.named_children():
            path = f"{prefix}{name}"
            if isinstance(child, nn.ModuleList):
                for i, block in enumerate(child):
                    units.append((f"{path}.{i}", block, True))
            elif _has_module_list(child) and weight_bytes(child) > 0:
                walk(path + ".", child)
            else:
                units.append((path, child, False))

    walk("", module)
    return units


def assign(units: Sequence[Tuple[str, nn.Module, bool]], devices: Sequence[torch.device],
           budgets: Sequence[int]) -> List[torch.device]:
    """Deal blocks to devices in order, in proportion to each device's budget (capped by
    it; the last device takes any remainder). Non-block units stay on the primary device,
    so their bytes come out of its budget first."""
    assert len(devices) == len(budgets) and devices
    sizes = [weight_bytes(u) for _, u, _ in units]
    block_total = sum(s for s, (_, _, b) in zip(sizes, units) if b)
    glue = sum(s for s, (_, _, b) in zip(sizes, units) if not b)
    budgets = [max(0, budgets[0] - glue), *budgets[1:]]
    budget_sum = sum(budgets)
    if block_total > budget_sum:
        logger.warning("split: %.1f GiB of blocks over %.1f GiB of budget; the last device "
                       "takes the rest and may run out", block_total / _GIB, budget_sum / _GIB)
    targets = [min(b, block_total * b / (budget_sum or 1)) for b in budgets]
    out: List[torch.device] = []
    di, used = 0, 0
    for size, (_name, _unit, is_block) in zip(sizes, units):
        if not is_block:
            out.append(devices[0])
            continue
        while di < len(devices) - 1 and used + size / 2 > targets[di]:
            di, used = di + 1, 0
        out.append(devices[di])
        used += size
    if block_total and logger.isEnabledFor(logging.INFO):
        per: Dict[str, int] = {}
        for size, dev in zip(sizes, out):
            per[str(dev)] = per.get(str(dev), 0) + size
        logger.info("split: %s", ", ".join(f"{d} {b / _GIB:.1f} GiB" for d, b in per.items()))
    return out


@dataclass
class _Shared:
    """Per-forward cache of copied tensors. The source is kept with its copy so its id
    cannot be reused by another tensor within the forward."""
    copies: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def clear(self) -> None:
        self.copies.clear()


def _to(obj: Any, device: torch.device, cache: Optional[_Shared]) -> Any:
    """``obj`` with its tensors on ``device``; containers are rebuilt, never edited."""
    if isinstance(obj, torch.Tensor):
        if obj.device == device:
            return obj
        if cache is not None:
            key = (id(obj), device)
            hit = cache.copies.get(key)
            if hit is not None and hit[0] is obj:
                return hit[1]
            moved = obj.to(device, non_blocking=True)
            cache.copies[key] = (obj, moved)
            return moved
        return obj.to(device, non_blocking=True)
    if isinstance(obj, tuple):
        if hasattr(obj, "_fields"):  # namedtuple
            return type(obj)(*(_to(o, device, cache) for o in obj))
        return tuple(_to(o, device, cache) for o in obj)
    if isinstance(obj, list):
        return [_to(o, device, cache) for o in obj]
    if isinstance(obj, dict):
        items = {k: _to(v, device, cache) for k, v in obj.items()}
        if type(obj) is dict:
            return items
        try:
            return type(obj)(**items)
        except Exception:
            moved = copy.copy(obj)
            for k, v in items.items():
                moved[k] = v
            return moved
    return obj


def _current(device: torch.device):
    """Make ``device`` the current one inside a ``with``; restored on any exit."""
    return torch.cuda.device(device) if device.type == "cuda" else nullcontext()


def _clear(cache: Optional[_Shared]) -> None:
    if cache is not None:
        cache.clear()


class _Depth(threading.local):
    """How deep this thread is inside one placed component's entry points."""
    level = 0


def _stream(gen, device: torch.device, home: Optional[torch.device], clear: Optional[_Shared],
            depth: Optional[_Depth] = None):
    """A generator advanced on ``device``, each item handed back on ``home``."""
    try:
        while True:
            if depth is not None:
                depth.level += 1
            try:
                with _current(device):
                    item = next(gen)
            except StopIteration:
                return
            finally:
                if depth is not None:
                    depth.level -= 1
            _clear(clear)
            yield item if home is None else _to(item, home, None)
    finally:
        gen.close()


def _entry(fn, device: torch.device, home: Optional[torch.device],
           share: Optional[_Shared] = None, clear: Optional[_Shared] = None,
           depth: Optional[_Depth] = None):
    """``fn`` run on ``device``: inputs copied there (through ``share``), the device
    current meanwhile, the result (each item, for a generator) brought to ``home``.
    ``clear`` is emptied around the call and after each item. With ``depth``, a call made
    from inside another entry point of the same component runs as is: only the outermost
    call moves results home."""
    def run(*args, **kwargs):
        if depth is not None and depth.level:
            return fn(*args, **kwargs)
        _clear(clear)
        if depth is not None:
            depth.level += 1
        try:
            with _current(device):
                out = fn(*_to(args, device, share), **_to(kwargs, device, share))
        finally:
            if depth is not None:
                depth.level -= 1
            _clear(clear)
        if inspect.isgenerator(out):
            return _stream(out, device, home, clear, depth)
        return out if home is None else _to(out, home, None)
    return run


class _Patch:
    """A method replaced on one instance."""
    def __init__(self, obj: Any, name: str, make) -> None:
        self.obj, self.name = obj, name
        setattr(obj, name, make(getattr(obj, name)))

    def remove(self) -> None:
        self.obj.__dict__.pop(self.name, None)


@dataclass
class Spread:
    """What was patched onto one component, so it can be undone."""
    handles: List[_Patch]

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _budgets(module: nn.Module, devices: Sequence[torch.device],
             caps: Mapping[torch.device, int]) -> List[int]:
    """What each device can take: its free memory (counting what ``module`` already holds
    there) less a stage's base reserve, up to its cap."""
    out = []
    for dev in devices:
        free = get_available_memory(dev)
        if free is not None:
            free += _bytes_on(module, dev)
        # Unknown headroom is not a capacity claim: use one model's weights as the share.
        budget = weight_bytes(module) if free is None else max(0, free - RESERVE_BASE_BYTES)
        out.append(min(budget, caps.get(dev, budget)))
    return out


def spread(module: Any, devices: Sequence[torch.device],
           caps: Optional[Mapping[torch.device, int]] = None) -> Spread:
    """Split ``module`` over ``devices`` (first = primary) and wrap its units."""
    held = held_modules(module)
    if len(held) != 1:
        raise ValueError(f"cannot split a component holding {len(held)} modules")
    module = held[0]
    primary = devices[0]
    units = split_units(module)
    budgets = _budgets(module, devices, caps or {})
    where = assign(units, devices, budgets)
    cache = _Shared()
    for _n, p in module.named_parameters(recurse=False):
        p.data = p.data.to(primary)
    for _n, b in module.named_buffers(recurse=False):
        b.data = b.data.to(primary)
    patches: List[_Patch] = []
    for i, ((_uname, unit, is_block), dev) in enumerate(zip(units, where)):
        unit.to(dev)
        nxt = where[i + 1] if i + 1 < len(where) else None
        home = primary if (not is_block or nxt != dev) and dev != primary else None
        patches.append(_Patch(unit, "forward", partial(_entry, device=dev, home=home, share=cache)))
    patches.append(_Patch(module, "forward",
                          partial(_entry, device=primary, home=primary, clear=cache)))
    return Spread(patches)


def _public_methods(module: nn.Module) -> List[str]:
    """Names of the public methods the module's class adds to ``nn.Module``."""
    base = set(dir(nn.Module))
    return [name for name in dir(type(module))
            if not name.startswith("_") and name not in base and name != "forward"
            and callable(getattr(type(module), name, None))
            and not isinstance(getattr(type(module), name), type)]


def place(module: Any, device: torch.device, primary: torch.device) -> Spread:
    """Move a component whole to ``device``. ``forward`` and the public methods its class
    adds take their inputs there, run with it current, and return results on ``primary``.

    The component's own ``device`` follows its weights, so code inside it that asks is
    told the truth and needs no per-submodule hooks.
    """
    patches: List[_Patch] = []
    depth = _Depth()        # one per component: its entry points call each other
    for held in held_modules(module):
        held.to(device)
        for name in ("forward", *_public_methods(held)):
            patches.append(_Patch(held, name, partial(_entry, device=device, home=primary,
                                                      depth=depth)))
    return Spread(patches)


def apply(model: Any, spec: DeviceSpec, plan: Dict[str, str]) -> Dict[str, Spread]:
    """Apply ``plan`` to a loaded model's registered components; pin what moved."""
    memory = model.memory
    primary = spec.primary
    done: Dict[str, Spread] = {}
    # Whole components move first: a split sizes itself on the memory they leave free.
    for name, where in sorted(plan.items(), key=lambda kv: kv[1] == "split"):
        module = memory.component(name)
        if module is None:
            raise ValueError(f"--place: model has no component {name!r}")
        if where == "split":
            done[name] = spread(module, spec.devices, caps=spec.caps)
        else:
            dev = with_index(where)
            if dev == primary:
                continue
            done[name] = place(module, dev, primary)
        memory.pin(name)
        logger.info("%s -> %s", name, where)
    return done


__all__ = ["DeviceSpec", "parse_devices", "parse_placement", "parse_stack",
           "check_dit_placement", "default_placement", "split_units", "assign",
           "spread", "place", "apply", "Spread"]
