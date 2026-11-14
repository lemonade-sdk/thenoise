"""The video request contract, independent of FastAPI and model libraries."""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Any, Mapping, Optional


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
    image: Optional[bytes]
    sampler: str
    lora_specs: Optional[list[str]]
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


@dataclass(frozen=True)
class VideoSpec:
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frames: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    output_format: Optional[str] = None
    image: Optional[bytes] = None
    sampler: Optional[str] = None
    lora_specs: Optional[list[str]] = None

    def resolve(self, defaults: Mapping[str, Any]) -> ResolvedVideo:
        """Resolve omitted options from the selected model's preferences."""
        def pref(key: str):
            value = getattr(self, key)
            return defaults[key] if value is None else value

        negative_prompt = pref("negative_prompt")
        width, height = pref("width"), pref("height")
        fps, steps = pref("fps"), pref("steps")
        guidance_scale = pref("guidance_scale")
        output_format, sampler = pref("output_format"), pref("sampler")

        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if fps <= 0 or steps <= 0:
            raise ValueError("fps and steps must be positive")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if not sampler:
            raise ValueError("sampler must not be empty")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if self.frames is not None and self.frames <= 0:
            raise ValueError("frames must be positive")
        if self.duration is not None and (not math.isfinite(self.duration) or self.duration <= 0):
            raise ValueError("duration must be finite and positive")
        explicit_frames = self.frames is not None
        frames = (self.frames if explicit_frames else
                  max(1, math.floor(self.duration * fps + 0.5))
                  if self.duration is not None else defaults["frames"])
        if output_format not in ("mp4", "webm"):
            raise ValueError("output_format must be mp4 or webm")
        if self.seed is not None and not -(2**63) <= self.seed < 2**64:
            raise ValueError("seed is outside torch.Generator's supported range")
        seed = secrets.randbits(32) if self.seed is None or self.seed == -1 else self.seed
        return ResolvedVideo(
            prompt=self.prompt, negative_prompt=negative_prompt,
            width=width, height=height, frame_count=frames, fps=fps,
            steps=steps, guidance_scale=guidance_scale, seed=seed,
            output_format=output_format, image=self.image,
            sampler=sampler, lora_specs=self.lora_specs,
            explicit_frames=explicit_frames,
        )
