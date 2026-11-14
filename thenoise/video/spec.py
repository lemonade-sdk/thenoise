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
    seed: int
    output_format: str
    image: bytes | None
    sampler: str
    #: LoRA specs by role, so a model with two LoRA-able components can be asked for a
    #: different set on each; every model has at least the default role to answer for.
    lora_specs: dict[str, list[str]]
    explicit_frames: bool = False

    @property
    def mode(self) -> str:
        return "i2v" if self.image is not None else "t2v"

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
    seed: int | None = None
    output_format: Literal["mp4", "webm"] | None = None
    image: Annotated[bytes, Field(min_length=1)] | None = None
    sampler: Annotated[str, Field(min_length=1)] | None = None
    lora_specs: LoraSpecs = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value

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
        width, height = pref("width"), pref("height")
        fps, steps = pref("fps"), pref("steps")
        guidance_scale = pref("guidance_scale")
        output_format, sampler = pref("output_format"), pref("sampler")
        if min(width, height, fps, steps) <= 0:
            raise ValueError("width, height, fps and steps must be positive")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if not sampler:
            raise ValueError("sampler must not be empty")
        if output_format not in ("mp4", "webm"):
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
            steps=steps, guidance_scale=guidance_scale, seed=seed,
            output_format=output_format, image=self.image,
            sampler=sampler, lora_specs=lora_specs_by_role(self.lora_specs),
            explicit_frames=self.frames is not None,
        )
