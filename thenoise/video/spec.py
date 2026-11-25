"""The video request contract, independent of FastAPI and model libraries."""
from __future__ import annotations

import math
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from thenoise.utils.lora_roles import LoraSpecs, lora_specs_by_role

#: What a request can be checked against before any model exists. A value it leaves unset
#: arrives from the model instead and is checked after the merge, in `resolve`.
_Positive = Annotated[int, Field(gt=0)]
_PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def frame_grid(wanted: float, *, compression: int, upward: bool) -> int:
    """The frame count a causal video codec can pack, nearest to the one that was asked for.

    A codec packs ``(F-1)/compression`` temporal cells around a first frame, so only
    ``compression*n+1`` frames exist at all. Explicit frame counts never shrink; a
    duration takes the nearest valid count, ties away from zero.
    """
    if not math.isfinite(wanted) or wanted <= 0:
        raise ValueError("video duration / frame count must be positive and finite")
    cells = (wanted - 1) / compression
    return compression * max(1, math.ceil(cells) if upward else math.floor(cells + 0.5)) + 1


def mode_of(image: bytes | None, end_image: bytes | None,
            references: tuple[bytes, ...] = ()) -> str:
    """``t2v``, ``i2v``, ``flf2v`` or ``ref2v``, from the conditions a request carries.

    References outrank keyframes: a clip given something to look like is a reference run whether
    or not it also names its first frame. Which architectures have anywhere to put one is the
    ``ref2v`` capability, so this answers from what arrived rather than from a mode a caller also
    has to set and keep in agreement.
    """
    if references:
        return "ref2v"
    if image is None:
        return "t2v"
    return "flf2v" if end_image is not None else "i2v"


@dataclass(frozen=True)
class ResolvedVideo:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    frame_count: int
    fps: int
    steps: int
    guidance_scale: float
    flow_shift: float
    seed: int
    output_format: str
    image: bytes | None
    #: The clip's last frame, which only a concatenating reference has anywhere to put.
    end_image: bytes | None
    sampler: str
    #: LoRA specs by role, so a model with two LoRA-able components can be asked for a
    #: different set on each; every model has at least the default role to answer for.
    lora_specs: dict[str, list[str]]
    explicit_frames: bool = False
    #: What the clip is asked to look like, as opposed to the frame it starts or ends on.
    references: tuple[bytes, ...] = ()

    @property
    def mode(self) -> str:
        return mode_of(self.image, self.end_image, self.references)

    def public(self) -> dict:
        return {
            "mode": self.mode, "width": self.width, "height": self.height,
            "frame_count": self.frame_count, "fps": self.fps,
            "duration": self.frame_count / self.fps, "steps": self.steps,
            "guidance_scale": self.guidance_scale, "seed": self.seed,
            "output_format": self.output_format,
            "sampler": self.sampler,
            "lora_specs": self.lora_specs,
        }


#: The containers the muxer writes.
OUTPUT_FORMATS = ("mp4", "webm")


class VideoSpec(BaseModel):
    """What a video request asks for, before any model has been consulted.

    A field left as ``None`` was left unset and takes the selected model's preference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    negative_prompt: str | None = None
    width: _Positive | None = None
    height: _Positive | None = None
    frames: _Positive | None = None
    duration: _PositiveFinite | None = None
    fps: _Positive | None = None
    steps: _Positive | None = None
    guidance_scale: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    flow_shift: _PositiveFinite | None = None
    seed: int | None = None
    output_format: Literal["mp4", "webm"] | None = None
    image: Annotated[bytes, Field(min_length=1)] | None = None
    end_image: Annotated[bytes, Field(min_length=1)] | None = None
    references: list[Annotated[bytes, Field(min_length=1)]] = []
    sampler: Annotated[str, Field(min_length=1)] | None = None
    lora_specs: LoraSpecs = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value

    @property
    def mode(self) -> str:
        return mode_of(self.image, self.end_image, tuple(self.references))

    @field_validator("seed")
    @classmethod
    def _seed_torch_generator_range(cls, value: int | None) -> int | None:
        if value is not None and not -(2**63) <= value < 2**64:
            raise ValueError("seed is outside torch.Generator's supported range")
        return value

    def resolve(self, defaults: Mapping[str, Any]) -> ResolvedVideo:
        """Fill what the request left unset, then merge into one set of numbers.

        Anything here can have come from a checkpoint marker rather than from the
        request, so it is checked rather than trusted.
        """
        def pref(key: str):
            value = getattr(self, key)
            return defaults[key] if value is None else value

        negative_prompt = pref("negative_prompt")
        if self.end_image is not None and self.image is None:
            raise ValueError("end_image needs image: a clip cannot end before it starts")
        width, height = pref("width"), pref("height")
        fps, steps = pref("fps"), pref("steps")
        guidance_scale, flow_shift = pref("guidance_scale"), pref("flow_shift")
        output_format, sampler = pref("output_format"), pref("sampler")
        if min(width, height, fps, steps) <= 0:
            raise ValueError("width, height, fps and steps must be positive")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if not math.isfinite(flow_shift) or flow_shift <= 0:
            raise ValueError("flow_shift must be finite and positive")
        if not sampler:
            raise ValueError("sampler must not be empty")
        if output_format not in OUTPUT_FORMATS:
            raise ValueError("output_format must be mp4 or webm")
        frames = (self.frames if self.frames is not None else
                  max(1, math.floor(self.duration * fps + 0.5))
                  if self.duration is not None else defaults["frames"])
        if frames < 1:
            raise ValueError("frame count must be positive")
        seed = secrets.randbits(32) if self.seed is None or self.seed == -1 else self.seed
        return ResolvedVideo(
            prompt=self.prompt, negative_prompt=negative_prompt,
            width=width, height=height, frame_count=frames, fps=fps,
            steps=steps, guidance_scale=guidance_scale, flow_shift=flow_shift,
            seed=seed,
            output_format=output_format, image=self.image, end_image=self.end_image,
            sampler=sampler, lora_specs=lora_specs_by_role(self.lora_specs),
            explicit_frames=self.frames is not None,
            references=tuple(self.references),
        )
