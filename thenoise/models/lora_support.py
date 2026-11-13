"""Shared LoRA lifecycle for image and video model adapters."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

from thenoise.errors import ModelReloadNeeded
from thenoise.utils.lora import LoRAApplyResult, apply_lora_to_model, undo_lora_on_model
from thenoise.utils.model_dir import ensure_safetensors, list_safetensors, resolve_in_dir

logger = logging.getLogger(__name__)


class LoRASupport:
    """One LoRA switching and error-recovery contract across all model types."""

    lora_fusions: Dict[str, Tuple[str, ...]] = {}

    def _lora_key_map(self, key: str) -> str:
        """Convert trainer-specific target names to this model's module paths."""
        return key

    def _init_lora_state(self) -> None:
        self._active_lora_result: Optional[LoRAApplyResult] = None
        self._active_lora_spec: Optional[str] = None
        self._lora_state_unfit = False

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
        dit: torch.nn.Module,
    ) -> None:
        """Switch active LoRAs on the DiT (in-place, under the lock).

        No-op when the requested config matches the current one.

        Running out of memory part-way leaves some layers merged with nothing recorded to
        undo them, and undoing the prefix is no repair: a baked delta subtracted from a
        rounded weight does not return the weight. The instance refuses from then on until
        it is rebuilt, with ``ModelReloadNeeded``.
        """
        if self._lora_state_unfit:
            raise ModelReloadNeeded(
                f"{self.name}: a LoRA switch ran out of memory on this instance, so it"
                " will not touch the DiT again. Reload the model."
            )

        new_spec = self._make_lora_spec_hash(lora_specs)
        if new_spec == self._active_lora_spec:
            return

        try:
            if self._active_lora_result is not None:
                logger.debug("Undoing previous LoRA config")
                undo_lora_on_model(dit, self._active_lora_result)
                self._active_lora_result = None

            if lora_specs and self.lora_dir is not None:
                lora_sds = []
                multipliers = []
                for spec in lora_specs:
                    filename, weight = self._parse_lora_spec(spec)
                    lora_sds.append(self._get_lora_sd(filename))
                    multipliers.append(weight)

                self._active_lora_result = apply_lora_to_model(
                    dit, lora_sds, multipliers,
                    dit_path=self.dit_path,
                    key_map=self._lora_key_map,
                    fusions=self.lora_fusions,
                )
                active_names = ", ".join(
                    self._parse_lora_spec(s)[0] for s in lora_specs
                )
                logger.info("Applied LoRA(s): %s", active_names)
            else:
                logger.debug("Using base model (no LoRA)")
        except torch.cuda.OutOfMemoryError:
            self._active_lora_result = None
            self._active_lora_spec = None
            self._lora_state_unfit = True
            logger.error("LoRA switch ran out of memory on %s", self.name)
            raise ModelReloadNeeded(
                f"{self.name}: applying the LoRAs ran out of memory part-way, so the DiT"
                " holds weights that are neither the base model nor any LoRA set. Reload"
                " the model."
            ) from None

        self._active_lora_spec = new_spec

    def list_loras(self) -> List[str]:
        """Available LoRA names relative to lora_dir, with .safetensors stripped."""
        return list_safetensors(self.lora_dir)
