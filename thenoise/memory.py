"""Device placement / residency manager for a model's swappable components.

A ``DiffusionModel`` owns exactly one ``MemoryManager``. Each heavy component (the
DiT, the text encoder) is registered with it; the pipeline ``ensure``s the
components a stage needs and ``offload``s the rest, so on a dGPU only the weights
in active use are resident. It is a strict single-transition state machine per
request, with no LRU and no keep-resident hint.

When ``offload_device == load_device`` (resident mode — enough unified memory for
every component) every move is a no-op. Only *weights / parameters* are managed
here; intermediate activations belong to the pipeline cache.

Registering a component also freezes it (eval mode + ``requires_grad_(False)``), so
the engine's inference-only contract is stated at the one choke point every
adapter's weights pass through.
"""
from __future__ import annotations

from typing import Dict, Optional, Set, Union

import torch
from torch import nn

from thenoise.inference import freeze
from thenoise.utils.device import clean_memory_on_device


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
        self._resident: Set[str] = set()

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

        A module already on the load device (e.g. the always-resident VAE) is
        recorded as resident. The ``.eval()`` of freezing recurses into
        submodules, which covers wrappers.
        """
        if module is not None:
            freeze(module)
        self._components[name] = module
        if module is not None and self._module_device(module) == self._load:
            self._resident.add(name)

    def resident(self) -> Set[str]:
        """Names of components currently resident on the load device."""
        return set(self._resident)

    # ------------------------------------------------------------ placement
    def ensure(self, *names: str) -> None:
        """Move each named component to the load device (no-op if already there)."""
        for name in names:
            if name in self._resident:
                continue
            module = self._components.get(name)
            if module is None:
                continue
            module.to(self._load)
            self._resident.add(name)

    def offload(self, *names: str) -> None:
        """Move each named component to the offload device and free compute memory.

        In resident mode this is a no-op that simply marks the components resident.
        """
        if self._load == self._offload:
            self._resident.update(names)
            return
        for name in names:
            if name not in self._resident:
                continue
            module = self._components.get(name)
            if module is None:
                continue
            module.to(self._offload)
            self._resident.discard(name)
        clean_memory_on_device(self._load)

    def offload_all(self) -> None:
        """Offload every registered component (use with care: includes the VAE)."""
        self.offload(*self._components)

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
