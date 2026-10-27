"""Device placement / residency manager for a model's swappable components.

A ``DiffusionModel`` asks its ``MemoryManager`` where to build each weight set
(``device_for``) and registers the result. A stage says what it runs on and how big an
image it moves; the manager decides where everything else goes and what to keep free:

    with memory.stage("vae", pixels=width * height) as s:
        pixels = s.run(model.decode, latents)

Two rules hold the device coherent:

* **A held component never moves**, so a stage cannot lose the weights under itself;
  everything else on the device is offered up when a stage needs the room.
* **Leaving a stage moves nothing.** A component gives up space only when the next request
  cannot be satisfied without it, so a move is paid for once.

Nothing else is pinned, whichever device a component was built on; when
``offload_device == load_device`` (resident mode) every move is a no-op. Registering
freezes the module, stating the inference-only contract at the choke point every
adapter's weights pass through.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Set, Union

import torch
from torch import nn

from thenoise.inference import freeze
from thenoise.utils.device import (
    clean_memory_on_device,
    get_device_memory,
    get_free_memory,
)

logger = logging.getLogger(__name__)

# A stage's room beside its weights: a fixed part (tiles, halos, feature caches) plus a
# part per pixel, the sampling loop being the same order as a decode. Coarse because the
# peaks are: two APUs of one generation measured 1.2-1.4x apart, nothing here on CUDA.
_RESERVE_BASE_BYTES = 2 << 30
_RESERVE_BYTES_PER_MEGAPIXEL = 1 << 30


def reserve_bytes(pixels: int) -> int:
    """Room a stage moving ``pixels`` keeps free for what its kernels allocate."""
    return _RESERVE_BASE_BYTES + int(_RESERVE_BYTES_PER_MEGAPIXEL * pixels / 1e6)


def _peak_bucket(pixels: int) -> int:
    """The megapixel bucket, powers of two from one up: a measured transient is
    comparable only with runs of a similar size.
    """
    megapixels = max(1, -(-pixels // 1_000_000))
    return 1 << (megapixels - 1).bit_length()


@dataclass
class _Stage:
    """The handle a stage runs its work through."""

    manager: "MemoryManager"

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """``fn(*args, **kwargs)``, squeezing the device and trying again if it ran out.

        Nothing freed means the device cannot hold this stage at all and the error stands.
        ``fn`` runs at most twice and must be a function of its inputs.
        """
        try:
            return fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            if self.manager.squeeze() == 0:
                raise
            return fn(*args, **kwargs)


class MemoryManager:
    """Placement + residency for a model's swappable components."""

    def __init__(
        self,
        load_device: Union[str, torch.device],
        offload_device: Union[str, torch.device],
    ):
        self._load = torch.device(load_device)
        self._offload = torch.device(offload_device)
        self._components: Dict[str, Optional[nn.Module]] = {}
        self._weights: Dict[str, int] = {}
        self._resident: Set[str] = set()
        self._held: Set[str] = set()
        self._peaks: Dict[tuple, Dict[int, int]] = {}
        self._depth = 0

    # ------------------------------------------------------------- identity
    @property
    def load_device(self) -> torch.device:
        """The compute device components run on."""
        return self._load

    @property
    def offload_device(self) -> torch.device:
        """The device idle weights live on."""
        return self._offload

    @property
    def offloads(self) -> bool:
        """True when offloading actually moves anything (the devices differ)."""
        return self._load != self._offload

    # ----------------------------------------------------------- registry
    def register(self, name: str, module: Optional[nn.Module]) -> None:
        """Register ``module`` under ``name``, freezing it and noting where it came up.

        Residency is recorded, not decided: a component already on the load device needs
        no upload. Nothing is pinned — any of them can be evicted later.
        """
        if module is not None:
            freeze(module)
        self._components[name] = module
        self._weights[name] = self._weight_bytes(module)
        if module is not None and self._module_device(module) == self._load:
            self._resident.add(name)

    def resident(self) -> Set[str]:
        """Names of components currently resident on the load device."""
        return set(self._resident)

    def weight_bytes(self, name: str) -> int:
        """What ``name`` occupies on the device it is loaded onto (0 if unmeasurable)."""
        return self._weights.get(name, 0)

    # ---------------------------------------------------------- admission
    def device_for(self, nbytes: int) -> torch.device:
        """The device to build ``nbytes`` of weights on: the load device while that leaves
        the stage floor free, the offload device once it does not.

        ``nbytes`` is a hint (a checkpoint's file size), asked before the bytes exist.
        The floor is a stage's fixed part: the only part evicting later cannot claw back.
        """
        if not self.offloads:
            return self._load
        free = get_free_memory(self._load)
        if free is None:
            return self._load                      # no figure, no policy
        target = self._load if free - nbytes >= _RESERVE_BASE_BYTES else self._offload
        logger.info(
            "%.1f GiB of weights -> %s (%.1f GiB free on %s, %.1f GiB kept for a stage)",
            nbytes / (1 << 30), target, free / (1 << 30),
            self._load, _RESERVE_BASE_BYTES / (1 << 30),
        )
        return target

    # ------------------------------------------------------- measured headroom
    def _reserve_for(self, names: tuple, pixels: int) -> int:
        """What ``stage`` keeps free: the fleet's figure, or this device's own if larger."""
        observed = self._peaks.get(names, {})
        bucket = _peak_bucket(pixels)
        # transient grows with the image, so guess an unrun size by the next bucket up
        return max(reserve_bytes(pixels),
                   observed.get(bucket, 0),
                   observed.get(bucket * 2, 0))

    def _learn(self, names: tuple, pixels: int, baseline: int) -> None:
        """Remember the peak a stage allocated above what was live on entry, for its size
        and itself. The window spans a retry, so a stage that had to squeeze learns what
        sufficed.
        """
        transient = self._peak_end(self._load, baseline)
        bucket = _peak_bucket(pixels)
        peaks = self._peaks.setdefault(names, {})
        if transient <= peaks.get(bucket, 0):
            return
        peaks[bucket] = transient
        fleet = reserve_bytes(pixels)
        if transient > fleet:
            logger.warning(
                "stage %s used %.1f GiB at %d Mpx, over its %.1f GiB reserve; "
                "reserving the measured figure for that size from now on",
                names or ("<room>",), transient / (1 << 30), bucket, fleet / (1 << 30),
            )

    @staticmethod
    def _peak_begin(device: torch.device) -> Optional[int]:
        """Restart the device's peak counter and report the bytes live now, or None where
        there is nothing to learn.
        """
        if device.type != "cuda":
            return None
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError:
            return None
        return torch.cuda.memory_allocated(device)

    @staticmethod
    def _peak_end(device: torch.device, baseline: int) -> int:
        """Bytes above ``baseline`` that were live at the peak since ``_peak_begin``."""
        return torch.cuda.max_memory_allocated(device) - baseline

    # ------------------------------------------------------------ placement
    @contextmanager
    def stage(self, *names: str, pixels: int) -> Iterator[_Stage]:
        """Run a stage needing ``names`` on the device, with room for ``pixels``.

        The one thing a caller says about placement. ``pixels`` sizes the reserve, so a
        stage that named a component and sized nothing could fill the device to the last
        byte; with no names it is a claim on room alone, which is all a consumer that owns
        its own weights has to say. Nothing moves on the way out.

        Wrap the work in the yielded handle's ``run`` for a squeeze-and-retry. The stage
        is measured on the way out, so the next one of its size reserves what this used.
        """
        self._held.update(names)
        depth, self._depth = self._depth, self._depth + 1
        baseline = self._peak_begin(self._load) if depth == 0 else None
        try:
            self._place(names, self._reserve_for(names, pixels))
            yield _Stage(self)
        finally:
            self._held.difference_update(names)
            self._depth = depth
            if depth == 0 and baseline is not None:
                self._learn(names, pixels, baseline)

    def offload(self, *names: str) -> None:
        """Move each named component to the offload device and free compute memory.

        A held component stays put. In resident mode this only marks them resident.
        """
        if self._load == self._offload:
            self._resident.update(names)
            return
        moved = False
        for name in names:
            if name not in self._resident or name in self._held:
                continue
            module = self._components.get(name)
            if module is None:
                continue
            module.to(self._offload)
            self._resident.discard(name)
            moved = True
        if moved:
            clean_memory_on_device(self._load)

    def squeeze(self) -> int:
        """Give back every swappable component nobody is holding; return the bytes freed.

        Zero means nothing was left to take, so a retry would fail the same way.
        """
        before = self.free_bytes()
        self.offload(*(self._resident - self._held))
        after = self.free_bytes()
        if before is None or after is None:
            return 0
        return max(after - before, 0)

    # -------------------------------------------------------------- headroom
    def free_bytes(self) -> Optional[int]:
        """Bytes of the load device not held by live tensors, None if it will not say.

        Allocator bookkeeping, not the driver's: a block another stage released counts
        as available before ``empty_cache`` hands it back.
        """
        if self._load.type != "cuda":
            return None
        total = get_device_memory(self._load)
        if total is None:
            return None
        return total - torch.cuda.memory_allocated(self._load)

    def _place(self, names: Sequence[str], reserve: int) -> None:
        """Put ``names`` on the compute device, with ``reserve`` left over afterwards."""
        waiting = [n for n in names
                   if n not in self._resident and self._components.get(n) is not None]
        self._make_room(reserve + sum(self.weight_bytes(n) for n in waiting))
        for name in waiting:
            module = self._components[name]
            module.to(self._load)
            self._resident.add(name)

    def _make_room(self, nbytes: int) -> None:
        """Displace swappable components, largest first, until ``nbytes`` are free."""
        if not self.offloads:
            return
        free = self.free_bytes()
        if free is None or free >= nbytes:
            return
        victims = self._resident - self._held
        for _, name in sorted((self.weight_bytes(n), n) for n in victims)[::-1]:
            self.offload(name)
            free = self.free_bytes()
            if free is None or free >= nbytes:
                break

    @staticmethod
    def _weight_bytes(module: Optional[nn.Module]) -> int:
        """Resident size of ``module``'s weights and buffers, 0 when it reports none.

        Quantized checkpoints are counted as they sit in memory, not as their files,
        which is what a placement decision needs.
        """
        total = 0
        for kind in ("parameters", "buffers"):   # a wrapper may expose neither
            for tensor in getattr(module, kind, lambda: ())():
                total += tensor.numel() * tensor.element_size()
        return total

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _module_device(module: nn.Module) -> Optional[torch.device]:
        """Current device of ``module`` (an ``nn.Module`` or a wrapper).

        Wrappers expose a ``device`` attribute/property; plain modules are probed
        via their first parameter or buffer."""
        if hasattr(module, "device"):
            return module.device
        for p in module.parameters():
            if p is not None:
                return p.device
        for b in module.buffers():
            if b is not None:
                return b.device
        return None


__all__ = ["MemoryManager"]
