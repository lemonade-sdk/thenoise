"""The video request contract, independent of FastAPI and model libraries."""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Optional


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
    image: Optional[bytes] = None
    sampler: str = "unipc"

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
        }


@dataclass(frozen=True)
class VideoSpec:
    prompt: str
    negative_prompt: str = ""
    width: int = 1280
    height: int = 704
    frames: Optional[int] = None
    duration: Optional[float] = None
    fps: int = 24
    steps: int = 30
    guidance_scale: float = 5.0
    seed: Optional[int] = None
    output_format: str = "webm"
    image: Optional[bytes] = None
    sampler: str = "unipc"

    def resolve(self) -> ResolvedVideo:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.fps <= 0 or self.steps <= 0:
            raise ValueError("fps and steps must be positive")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if not self.sampler:
            raise ValueError("sampler must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.frames is not None and self.frames <= 0:
            raise ValueError("frames must be positive")
        if self.duration is not None and (not math.isfinite(self.duration) or self.duration <= 0):
            raise ValueError("duration must be finite and positive")
        width, height = self.width, self.height
        frames = (self.frames if self.frames is not None else
                  max(1, math.floor(self.duration * self.fps + 0.5))
                  if self.duration is not None else 33)
        if self.output_format not in ("mp4", "webm"):
            raise ValueError("output_format must be mp4 or webm")
        if self.seed is not None and not -(2**63) <= self.seed < 2**64:
            raise ValueError("seed is outside torch.Generator's supported range")
        seed = secrets.randbits(32) if self.seed is None or self.seed == -1 else self.seed
        return ResolvedVideo(
            prompt=self.prompt, negative_prompt=self.negative_prompt,
            width=width, height=height, frame_count=frames, fps=self.fps,
            steps=self.steps, guidance_scale=self.guidance_scale, seed=seed,
            output_format=self.output_format, image=self.image,
            sampler=self.sampler,
        )
