"""Shared lifecycle contract for video model adapters.

The image DiffusionModel contract assumes 4D latents and an image pipeline.
Share device residency, model-supplied defaults and teardown here; frame grids,
reference conditioning and sampling remain model-specific. Multi-keyframe or
synchronized audio workflows require orchestration extensions, not Wan defaults.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar

import torch

from thenoise.memory import MemoryManager
from thenoise.models.lora_support import LoRASupport
from thenoise.models.config import ModelConfig
from thenoise.utils.checkpoint import checkpoint_bytes
from thenoise.utils.device import release_device_memory
from thenoise.utils.device_patches import apply_device_patches
from thenoise.utils.dynamo import apply_recompile_limit
from thenoise.video.jobs import Cancellation
from thenoise.video.spec import ResolvedVideo, VideoSpec


class VideoModel(LoRASupport, ABC):
    VIDEO: ClassVar[bool] = True
    CAPABILITIES: ClassVar[dict[str, bool]] = {"video": True}
    DEFAULT_PREFS: ClassVar[dict[str, Any]]
    name: ClassVar[str]

    def __init__(self, *, config: ModelConfig):
        self.dit_path = config.dit_path
        self.vae_path = config.vae_path
        self.text_encoder_path = config.text_encoder_path
        self.lora_dir = getattr(config, "lora_dir", None)
        self._init_lora_state()
        self.device = torch.device(config.device)
        self.offload_device = torch.device(config.offload_device)
        self.memory = MemoryManager(self.device, self.offload_device)

        apply_recompile_limit()
        apply_device_patches(self.device)

    @staticmethod
    @abstractmethod
    def detect(checkpoint) -> bool:
        """Recognize the model from an open checkpoint."""

    def _load_target(self, path: str) -> str:
        """The device to build the weights at ``path`` on, the manager's decision."""
        return str(self.memory.device_for(checkpoint_bytes(path)))

    @classmethod
    def resolve_video(cls, spec: VideoSpec) -> ResolvedVideo:
        """Resolve optional request values using this model's preferences."""
        return spec.resolve(cls.DEFAULT_PREFS)

    @abstractmethod
    def load_weights(self) -> None:
        """Load all model components before the server starts accepting requests."""

    @abstractmethod
    def generate(
        self, spec: ResolvedVideo, token: Cancellation,
        progress: Callable[[str, int, int], None], output: Path,
    ) -> None:
        """Render one resolved video to the destination."""

    @abstractmethod
    def _clear_weights(self) -> None:
        """Drop model-owned references to modules and tokenizer state."""

    def unload(self) -> None:
        """Drop both model and residency references before freeing the device."""
        self._clear_weights()
        self._init_lora_state()
        # The old manager still references registered modules and residency
        # state. Rebuild it so a subsequent load is not marked resident.
        self.memory = MemoryManager(self.device, self.offload_device)
        release_device_memory(self.device)
