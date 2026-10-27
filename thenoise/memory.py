"""Device placement / residency manager for a model's swappable components.

A ``DiffusionModel`` owns exactly one ``MemoryManager`` and registers its heavy
components (DiT, text encoder, VAE) with it. A stage says what it runs on and how much
room it needs; the manager decides where everything else is:

    with memory.stage("vae", activations=model.activation_bytes(width, height)):
        pixels = model.decode(latents)

Three rules hold the device coherent:

* **A held component never moves**, so a stage cannot lose the weights under itself;
  everything else on the device is offered up when a stage needs the room.
* **Leaving a stage moves nothing.** A component gives up space only when the next request
  cannot be satisfied without it, so a move is paid for once.
* **A component that came up on the compute device is fixed there** (``register``) — the
  adapters build their VAEs that way and their DiTs on the offload device, which is the one
  fact separating small always-needed weights from bulk. Fixity stops a VAE being traded
  away for one claim and uploaded again for the next.

``squeeze`` is that policy stated by a stage that ran out anyway: hand over the device
except what I am holding. When ``offload_device == load_device`` (resident mode) every move
is a no-op.

Only weights are tracked; activations are heard of as a stage's figure and owned by the
pipeline cache. Registering freezes the module, stating the inference-only contract at the
choke point every adapter's weights pass through.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Sequence, Set, Union

import torch
from torch import nn

from thenoise.inference import freeze
from thenoise.utils.device import clean_memory_on_device, get_device_memory


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
        self._fixed: Set[str] = set()
        self._held: Set[str] = set()

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
        """Register ``module`` under ``name``, freezing it and noting its residency.

        Where a component came up decides whether it is fixed, and this is the one place
        that knows which is which.
        """
        if module is not None:
            freeze(module)
        self._components[name] = module
        self._weights[name] = self._weight_bytes(module)
        if module is not None and self._module_device(module) == self._load:
            self._resident.add(name)
            self._fixed.add(name)

    def resident(self) -> Set[str]:
        """Names of components currently resident on the load device."""
        return set(self._resident)

    def weight_bytes(self, name: str) -> int:
        """What ``name`` occupies on the device it is loaded onto (0 if unmeasurable)."""
        return self._weights.get(name, 0)

    # ------------------------------------------------------------ placement
    @contextmanager
    def stage(self, *names: str, activations: int) -> Iterator[None]:
        """Run a stage needing ``names`` on the device, plus ``activations`` bytes of room.

        Required ``activations`` is deliberate: weights are knowable, what they allocate
        while running is not, and a stage that named a component and claimed nothing
        would fill the device to the last byte. With no names it is a claim on room
        alone, which is all a consumer that owns its weights has to say.
        """
        self._held.update(names)
        try:
            self._place(names, activations)
            yield
        finally:
            self._held.difference_update(names)

    def ensure(self, *names: str) -> None:
        """Move each named component to the load device (no-op if already there).

        Placement without a hold; ``stage`` is what a caller with a lifetime wants.
        """
        self._place(names, 0)

    def offload(self, *names: str) -> None:
        """Move each named component to the offload device and free compute memory.

        Held and fixed components stay put. In resident mode this only marks them
        resident.
        """
        if self._load == self._offload:
            self._resident.update(names)
            return
        moved = False
        for name in names:
            if name not in self._resident or name in self._held or name in self._fixed:
                continue
            module = self._components.get(name)
            if module is None:
                continue
            module.to(self._offload)
            self._resident.discard(name)
            moved = True
        if moved:
            clean_memory_on_device(self._load)

    def squeeze(self, *, held: Union[str, Sequence[str]] = ()) -> int:
        """Give back every swappable component nobody is holding; return the bytes freed.

        Zero means nothing was left to take, so the caller knows a retry fails alike.
        """
        keep = {held} if isinstance(held, str) else set(held)
        before = self.free_bytes()
        self.offload(*(self._resident - self._held - self._fixed - keep))
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

    def _place(self, names: Sequence[str], activations: int) -> None:
        """Put ``names`` on the compute device, with room for ``activations`` too."""
        waiting = [n for n in names
                   if n not in self._resident and self._components.get(n) is not None]
        self._make_room(activations + sum(self.weight_bytes(n) for n in waiting))
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
        victims = self._resident - self._held - self._fixed
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
