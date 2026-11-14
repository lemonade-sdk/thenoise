"""What a video adapter shares with every image model, and what only it has.

Derives from the image contract, so device residency, component placement, the preference
chain, capabilities and LoRA switching arrive once (``DiffusionModel``). What is added here
is temporal: a request resolved onto a frame grid, a schedule that more than one
transformer may answer, and a codec that hands back frames instead of one image.

A video adapter exposes its main transformer as ``dit``, exactly as an image adapter does,
and any second expert under the role name its checkpoint paths use. Those names —
``dit``, ``dit_low`` — are the same strings in ``dit_paths``, in the ``MemoryManager``
registry and in :attr:`Segment.stage`, which is what lets the controller stage, switch and
(one day) per-role LoRA a component without knowing what model it is driving.
"""
from __future__ import annotations

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import torch

from thenoise.memory import MemoryManager
from thenoise.models.base import DiffusionModel
from thenoise.samplers import Step
from thenoise.video.spec import ResolvedVideo, VideoSpec


@dataclass(frozen=True)
class Segment:
    """One span of the schedule, answered by one component that is resident for it.

    A single-transformer model has one segment. A two-expert model splits the schedule
    where its experts hand over, each half carrying the guidance its expert was trained
    with. Nothing here says the two may not share a device: whether both stay resident is
    the memory manager's, and only the split into stages is the model's.
    """

    stage: str
    module: torch.nn.Module
    steps: tuple[Step, ...]
    guidance_scale: float


class VideoModel(DiffusionModel):
    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "video": True}

    #: The video preferences every adapter shares. What a release decides for itself —
    #: frame count, frame rate, size, steps — belongs to that release and is absent here;
    #: `tests/test_video_shared.py` asserts every released variant supplies every
    #: preference a request can leave unset.
    DEFAULT_PREFS = {**DiffusionModel.DEFAULT_PREFS,
                     "negative_prompt": "", "output_format": "webm"}

    #: The segment `denoise_step` is currently answering. Like the KV caches, per-run
    #: state set by the controller between steps, never across a swap.
    _segment: Optional[Segment] = None

    # ------------------------------------------------------------ request
    @property
    def default_prefs(self) -> dict:
        """An unset request field falls back to the checkpoint markers, then here."""
        return {**self.DEFAULT_PREFS, **self.checkpoint_prefs}

    def resolve_video(self, spec: VideoSpec) -> ResolvedVideo:
        """Resolve optional request values, then the size onto this model's grid."""
        resolved = spec.resolve(self.default_prefs)
        width, height = self.resolve_size(resolved.width, resolved.height)
        if (width, height) != (resolved.width, resolved.height):
            resolved = dataclasses.replace(resolved, width=width, height=height)
        return resolved

    # ------------------------------------------------------------ schedule
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
    @abstractmethod
    def decode_frames(self, latents: torch.Tensor) -> Iterator[torch.Tensor]:
        """Canonical clip latent -> pixels ``[C, H, W]`` in [-1, 1], one frame at a time.

        A generator, consumed as the container takes frames, so a long clip is never live
        all at once.
        """

    # ------------------------------------------------------------ teardown
    @abstractmethod
    def _clear_weights(self) -> None:
        """Drop model-owned references to modules and tokenizer state."""

    def unload(self) -> None:
        """Drop both model and residency references before the device is freed."""
        self._clear_weights()
        self._init_lora_state()
        # The old manager still references registered modules and residency state.
        # Rebuild it so a subsequent load is not marked resident.
        self.memory = MemoryManager(self.device, self.offload_device)
