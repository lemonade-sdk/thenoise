"""Shared LoRA lifecycle for image and video model adapters."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

from thenoise.errors import ModelReloadNeeded
from thenoise.utils.lora import LoRAApplyResult, apply_lora_to_model, undo_lora_on_model
from thenoise.utils.lora_roles import DEFAULT_LORA_ROLE
from thenoise.utils.model_dir import ensure_safetensors, list_safetensors, resolve_in_dir

logger = logging.getLogger(__name__)


class LoRASupport:
    """One LoRA switching and error-recovery contract across all model types.

    LoRAs are switched per *role* — the name a model gives each of its LoRA-able
    components, which is the name it registers them for residency under. A model with
    one component uses the default and needs no role in its requests.
    """

    lora_fusions: Dict[str, Tuple[str, ...]] = {}

    def _lora_key_map(self, key: str) -> str:
        """Convert trainer-specific target names to this model's module paths."""
        return key

    def _init_lora_state(self) -> None:
        # Per role: undoing must use the record of what was applied to the module in
        # hand, and an out-of-memory on one component says nothing about another.
        self._active_loras: Dict[str, Tuple[str, Optional[LoRAApplyResult]]] = {}
        self._lora_unfit_roles: set[str] = set()

    # --------------------------------------------------------------- LoRA
    def _parse_lora_spec(self, spec: str) -> Tuple[str, float]:
        """Parse a 'filename:weight' spec into (filename, weight), appending .safetensors."""
        if ":" in spec:
            filename, weight_str = spec.rsplit(":", 1)
            weight = float(weight_str)
        else:
            filename = spec
            weight = 1.0

        filename = ensure_safetensors(filename)

        return filename, weight

    def _resolve_lora_path(self, filename: str) -> str:
        """Resolve a LoRA filename against lora_dir, rejecting path traversal."""
        return resolve_in_dir(self.lora_dir, filename)

    def _get_lora_sd(self, filename: str) -> Dict[str, torch.Tensor]:
        """Load a LoRA state dict from disk."""
        filepath = self._resolve_lora_path(filename)

        logger.info("Loading LoRA: %s", filepath)
        return load_file(filepath, device=str(self.device))

    def _make_lora_spec_hash(self, lora_specs: Optional[List[str]]) -> str:
        """Stable key for a set of LoRA specs."""
        if not lora_specs:
            return "__none__"
        return "|".join(sorted(lora_specs))

    def switch_loras(
        self,
        lora_specs: Optional[List[str]],
        module: torch.nn.Module,
        role: str = DEFAULT_LORA_ROLE,
    ) -> None:
        """Switch the LoRAs on one component (in-place, under the lock).

        No-op when the requested config matches what that role already has.

        Running out of memory part-way leaves some layers merged with nothing recorded to
        undo them, and undoing the prefix is no repair: a baked delta subtracted from a
        rounded weight does not return the weight. That component refuses from then on
        until the model is rebuilt, with ``ModelReloadNeeded``.
        """
        if role in self._lora_unfit_roles:
            raise ModelReloadNeeded(
                f"{self.name}: a LoRA switch ran out of memory on '{role}' on this"
                " instance, so it will not touch that component again. Reload the model."
            )

        new_spec = self._make_lora_spec_hash(lora_specs)
        active = self._active_loras.get(role)
        if active is not None and active[0] == new_spec:
            return

        try:
            if active is not None and active[1] is not None:
                logger.debug("Undoing previous LoRA config on '%s'", role)
                undo_lora_on_model(module, active[1])

            if lora_specs and self.lora_dir is not None:
                lora_sds = []
                multipliers = []
                for spec in lora_specs:
                    filename, weight = self._parse_lora_spec(spec)
                    lora_sds.append(self._get_lora_sd(filename))
                    multipliers.append(weight)

                result = apply_lora_to_model(
                    module, lora_sds, multipliers,
                    dit_path=self.dit_path,
                    key_map=self._lora_key_map,
                    fusions=self.lora_fusions,
                )
                active_names = ", ".join(
                    self._parse_lora_spec(spec)[0] for spec in lora_specs
                )
                logger.info("Applied LoRA(s) on '%s': %s", role, active_names)
            else:
                result = None
                logger.debug("Using base model on '%s' (no LoRA)", role)
        except torch.cuda.OutOfMemoryError:
            self._active_loras.pop(role, None)
            self._lora_unfit_roles.add(role)
            logger.error("LoRA switch ran out of memory on %s ('%s')", self.name, role)
            raise ModelReloadNeeded(
                f"{self.name}: applying the LoRAs on '{role}' ran out of memory part-way,"
                " so that component holds weights that are neither the base model nor any"
                " LoRA set. Reload the model."
            ) from None

        self._active_loras[role] = (new_spec, result)

    def list_loras(self) -> List[str]:
        """Available LoRA names relative to lora_dir, with .safetensors stripped."""
        return list_safetensors(self.lora_dir)
