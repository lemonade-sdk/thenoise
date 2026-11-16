"""What a video adapter shares with every model that renders pixels, and what only it
has: a request resolved onto a frame grid, a schedule more than one transformer may
answer, and a codec handing back frames instead of one image.
"""
from __future__ import annotations

import dataclasses
import io
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import torch

from thenoise.memory import MemoryManager
from thenoise.models.pixel_base import PixelModel
from thenoise.samplers import Step
from thenoise.utils.image_tensor import load_image
from thenoise.video.spec import ResolvedVideo, VideoSpec


@dataclass(frozen=True)
class Segment:
    """One span of the schedule, answered by one component resident for it.

    A two-expert model splits the schedule where its experts hand over; whether both
    stay resident is the memory manager's, not this struct's.
    """

    #: Residency and LoRA stage name of this span's component, as in ``dit_paths``.
    stage: str
    module: torch.nn.Module
    steps: tuple[Step, ...]
    guidance_scale: float


class VideoModel(PixelModel):
    MEDIUM = "video"

    CAPABILITIES = {**PixelModel.CAPABILITIES, "image": None,
                    "video": {"t2v": False, "i2v": False, "flf2v": False}}

    #: Shared video prefs; ``test_video_shared.py`` checks each variant covers them all.
    DEFAULT_PREFS = {**PixelModel.DEFAULT_PREFS,
                     "negative_prompt": "", "output_format": "webm"}

    #: Per-run state the controller sets between steps, never across a swap.
    _segment: Optional[Segment] = None

    # ------------------------------------------------------------ request
    @property
    def default_prefs(self) -> dict:
        """An unset request field falls back to the checkpoint markers, then here."""
        return {**self.DEFAULT_PREFS, **self.checkpoint_prefs}

    def resolve_video(self, spec: VideoSpec) -> ResolvedVideo:
        """Resolve optional request values, then the size onto this model's grid."""
        # A reference answers an unnamed size, as it does for a still edit.
        if spec.image is not None and self.capability("video")["i2v"]:
            width, height = self.size_from_reference(
                spec.width, spec.height, load_image(io.BytesIO(spec.image)))
            spec = spec.model_copy(update={"width": width, "height": height})
        resolved = spec.resolve(self.default_prefs)
        width, height = self.resolve_size(resolved.width, resolved.height)
        if (width, height) != (resolved.width, resolved.height):
            resolved = dataclasses.replace(resolved, width=width, height=height)
        return resolved

    # ------------------------------------------------------------ schedule
    def transition_segment(self, latents: torch.Tensor, previous: Segment,
                           next_segment: Segment, params, token) -> torch.Tensor:
        """Optional model-specific latent transition between complete segments.

        Called after the previous stage releases its weights and before the next
        stage is held. Wan's high/low experts share latents unchanged. AV models
        may use a separate staged spatial upscaler here.
        """
        return latents

    def clear_video_state(self) -> None:
        """Release transient, request-owned reference state on success or failure."""

    def segments(self, params, schedule: Sequence[Step],
                 guidance_scale: float) -> list[Segment]:
        """The schedule split by the component that answers each span.

        The controller stages each segment, switches LoRAs on its module and runs the
        solver over its steps, so a model with more than one transformer says only where
        the spans are.
        """
        return [Segment("dit", self.dit, tuple(schedule), guidance_scale)]
    def begin_segment(self, segment: Segment) -> None:
        """Tell ``denoise_step`` which segment it is inside; called before each one."""
        self._segment = segment

    # ------------------------------------------------------------ codec
    def audio_components(self) -> tuple[str, ...]:
        """Extra MemoryManager roles needed to decode audio, if this codec has any."""
        return ()

    def decode_audio(self, latents: torch.Tensor):
        """Optional ``(waveform, sample_rate)`` for the AV muxer."""
        return None

    @abstractmethod
    def decode_frames(self, latents: torch.Tensor) -> Iterator[torch.Tensor]:
        """Canonical clip latent -> pixels ``[C, H, W]`` in [-1, 1], yielded one frame at
        a time so a long clip is never live all at once."""

    # ------------------------------------------------------------ teardown
    @abstractmethod
    def _clear_weights(self) -> None:
        """Drop model-owned references to modules and tokenizer state."""

    def unload(self) -> None:
        """Drop both model and residency references before the device is freed."""
        self._clear_weights()
        self._init_lora_state()
        # Rebuilt so a later load is not marked resident by the old manager.
        self.memory = MemoryManager(self.device, self.offload_device)
