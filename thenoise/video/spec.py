"""The video request contract, independent of FastAPI and model libraries."""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Optional

def _frame_grid(wanted: float, *, upward: bool) -> int:
    if not math.isfinite(wanted) or wanted <= 0:
        raise ValueError("video duration / frame count must be positive and finite")
    # Wan 2.2 packs (F-1)/4 temporal cells. Explicit frames never shrink;
    # an approximate duration chooses the nearest valid frame count.
    cells = (wanted - 1) / 4
    # Match the API's ties-away-from-zero frame rounding.
    n = math.ceil(cells) if upward else math.floor(cells + 0.5)
    return 4 * max(1, n) + 1


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
    sampling_method: str = "unipc"
    flow_shift: float = 5.0

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
            "sampling_method": self.sampling_method, "flow_shift": self.flow_shift,
        }


@dataclass(frozen=True)
class VideoSpec:
    prompt: str
    negative_prompt: str = ""
    width: int = 1280
    height: int = 704
    video_frames: Optional[int] = None
    duration: Optional[float] = None
    fps: int = 24
    steps: int = 30
    cfg_scale: float = 5.0
    seed: Optional[int] = None
    output_format: str = "webm"
    image: Optional[bytes] = None
    sampling_method: str = "unipc"
    flow_shift: float = 5.0

    def resolve(self) -> ResolvedVideo:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 1 <= self.fps <= 60:
            raise ValueError("fps must be between 1 and 60")
        if not 1 <= self.steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0 or self.cfg_scale > 30:
            raise ValueError("cfg_scale must be finite and between 0 and 30")
        if self.sampling_method not in ("unipc", "euler"):
            raise ValueError("sampling_method must be unipc or euler")
        if not math.isfinite(self.flow_shift) or not 0 < self.flow_shift <= 20:
            raise ValueError("flow_shift must be finite and in (0, 20]")
        if not 128 <= self.width <= 2048 or not 128 <= self.height <= 2048:
            raise ValueError("width and height must be between 128 and 2048")
        width, height = self.width // 32 * 32, self.height // 32 * 32
        if self.video_frames is not None:
            frames = _frame_grid(self.video_frames, upward=True)
        elif self.duration is not None:
            frames = _frame_grid(self.duration * self.fps, upward=False)
        else:
            frames = 33
        if self.output_format not in ("mp4", "webm"):
            raise ValueError("output_format must be mp4 or webm")
        if self.image is not None and len(self.image) > 15 * 1024 * 1024:
            raise ValueError("image exceeds 15 MiB")
        if self.seed is not None and not -1 <= self.seed < 2**32:
            raise ValueError("seed must be -1 or between 0 and 2**32-1")
        seed = secrets.randbits(32) if self.seed is None or self.seed == -1 else self.seed
        return ResolvedVideo(
            prompt=self.prompt, negative_prompt=self.negative_prompt,
            width=width, height=height, frame_count=frames, fps=self.fps,
            steps=self.steps, guidance_scale=self.cfg_scale, seed=seed,
            output_format=self.output_format, image=self.image,
            sampling_method=self.sampling_method, flow_shift=self.flow_shift,
        )
