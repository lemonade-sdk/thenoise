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
from thenoise.models.config import VideoSamplingParams
from thenoise.models.pixel_base import PixelModel
from thenoise.samplers import Step
from thenoise.utils.attention import DEFAULT_ATTENTION
from thenoise.utils.image_tensor import load_image
from thenoise.utils.media import media_size
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
    #: What the latents this span walks are a picture of: a flat latent cannot tell a
    #: codec whose stream carries more than one modality.
    params: VideoSamplingParams


@dataclass(frozen=True)
class Waveform:
    """PCM a model decoded for a clip: ``[channels, samples]``, float in [-1, 1].

    The rate is the codec's, so it travels with the audio: the muxer must not infer it
    from the clip length.
    """

    samples: torch.Tensor
    sample_rate: int


@dataclass(frozen=True)
class Condition:
    """A condition a clip run reads alongside its own frames, already through a codec.

    ``frame``, negative counted from the end, pins it there; without one it keeps the grid
    it arrived with and takes time ahead of the clip's timeline.
    """

    video: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    frame: Optional[int] = None


class VideoModel(PixelModel):
    MEDIUM = "video"

    CAPABILITIES = {**PixelModel.CAPABILITIES, "image": None,
                    "video": {"t2v": False, "i2v": False, "flf2v": False,
                                  "ref2v": False}}

    #: Whether the architecture has anywhere in its packing for reference conditions,
    #: which are not the frames a clip starts or ends on. Nothing in a checkpoint says
    #: which release it is, so this is answered by the architecture, not by a filename.

    #: The prefs every video variant must answer, whatever its own defaults are.
    DEFAULT_PREFS = {**PixelModel.DEFAULT_PREFS,
                     "negative_prompt": "", "output_format": "webm"}

    #: Per-run state, set between steps and never carried across a component swap.
    _segment: Optional[Segment] = None
    _attention: str = DEFAULT_ATTENTION

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
        elif (spec.width is None and spec.height is None and spec.references
              and self.capability("video").get("ref2v")):
            # Nothing keys a size but the first thing the clip was given to look like --
            # and a reference with no pictures in it keys nothing.
            seen = next((size for size in (media_size(guide.media)
                                           for guide in spec.references) if size), None)
            if seen is not None:
                width, height = self.size_from_reference(None, None, seen)
                spec = spec.model_copy(update={"width": width, "height": height})
        resolved = spec.resolve(self.default_prefs)
        self.resolve_attention(resolved.attention)
        width, height = self.resolve_size(resolved.width, resolved.height)
        if (width, height) != (resolved.width, resolved.height):
            resolved = dataclasses.replace(resolved, width=width, height=height)
        return resolved

    # ------------------------------------------------------------ schedule
    def transition_segment(self, latents: torch.Tensor, previous: Segment,
                           next_segment: Segment, params, token) -> torch.Tensor:
        """Optional latent transition between complete segments, run after the previous
        stage releases its weights and before the next is held.
        """
        return latents

    def clear_video_state(self) -> None:
        """Release transient, request-owned reference state on success or failure."""

    def segments(self, params, schedule: Sequence[Step],
                 guidance_scale: float) -> list[Segment]:
        """The schedule split by the component that answers each span: a model with more
        than one transformer says only where the spans are.
        """
        return [Segment("dit", self.dit, tuple(schedule), guidance_scale, params)]

    def begin_segment(self, segment: Segment) -> None:
        """Called before each segment with the one the run is now inside."""
        self._segment = segment
        self._attention = segment.params.attention

    # ------------------------------------------------------------ codec
    def audio_components(self) -> tuple[str, ...]:
        """Extra MemoryManager roles needed to decode audio, if this codec has any."""
        return ()

    def decoded_pixels(self, spec: ResolvedVideo) -> int:
        """Pixels one call to :meth:`decode_frames` holds live at its peak.

        A codec that answers a frame at a time needs a frame of headroom. One that has to
        run over the whole clip before it can hand back the first frame needs the clip,
        and a stage sized by the frame alone then survives only by being retried.
        """
        return spec.width * spec.height

    def decode_audio(self, latents: torch.Tensor,
                     params: VideoSamplingParams) -> Optional["Waveform"]:
        """Canonical clip latent -> the sound to mux beside its frames, if this codec has one."""
        return None

    @abstractmethod
    def decode_frames(self, latents: torch.Tensor,
                      params: VideoSamplingParams) -> Iterator[torch.Tensor]:
        """Canonical clip latent -> pixels ``[C, H, W]`` in [-1, 1], yielded one frame at
        a time so a long clip is never live all at once.
        """

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
