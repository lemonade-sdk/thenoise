"""The mesh request contract, independent of FastAPI and model libraries."""
from __future__ import annotations

import math
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from thenoise.utils.lora_roles import LoraSpecs, lora_specs_by_role

_Positive = Annotated[int, Field(gt=0)]
_PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]

#: The containers a mesh is written into.
OUTPUT_FORMATS = ("glb", "obj", "stl")


def mode_of(image: bytes | None) -> str:
    """``i2d`` or ``t2d``, from the picture a request carries."""
    return "i2d" if image is not None else "t2d"


@dataclass(frozen=True)
class ResolvedMesh:
    prompt: str
    negative_prompt: str
    image: bytes | None
    latents: int
    steps: int
    guidance_scale: float
    sampler: str
    seed: int
    output_format: str
    octree_resolution: int
    bounds: float
    mc_level: float
    lora_specs: dict[str, list[str]]

    @property
    def mode(self) -> str:
        return mode_of(self.image)

    def public(self) -> dict:
        return {
            "mode": self.mode, "latents": self.latents, "steps": self.steps,
            "guidance_scale": self.guidance_scale, "sampler": self.sampler,
            "seed": self.seed, "output_format": self.output_format,
            "octree_resolution": self.octree_resolution, "bounds": self.bounds,
            "mc_level": self.mc_level, "lora_specs": self.lora_specs,
        }


class MeshSpec(BaseModel):
    """What a mesh request asks for; a field left ``None`` takes the model's preference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str = ""
    negative_prompt: str | None = None
    image: Annotated[bytes, Field(min_length=1)] | None = None
    latents: _Positive | None = None
    steps: _Positive | None = None
    guidance_scale: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    sampler: Annotated[str, Field(min_length=1)] | None = None
    seed: int | None = None
    output_format: Literal["glb", "obj", "stl"] | None = None
    octree_resolution: _Positive | None = None
    bounds: _PositiveFinite | None = None
    mc_level: Annotated[float, Field(allow_inf_nan=False)] | None = None
    lora_specs: LoraSpecs = None

    @property
    def mode(self) -> str:
        return mode_of(self.image)

    @field_validator("seed")
    @classmethod
    def _seed_torch_generator_range(cls, value: int | None) -> int | None:
        if value is not None and not -(2**63) <= value < 2**64:
            raise ValueError("seed is outside torch.Generator's supported range")
        return value

    def resolve(self, defaults: Mapping[str, Any]) -> ResolvedMesh:
        """Fill what the request left unset from the model's defaults, then check."""
        def pref(key: str, default_key: str | None = None):
            value = getattr(self, key)
            return defaults[default_key or key] if value is None else value

        latents, steps = pref("latents", "num_latents"), pref("steps")
        guidance_scale, sampler = pref("guidance_scale"), pref("sampler")
        output_format = pref("output_format")
        octree_resolution = pref("octree_resolution")
        bounds, mc_level = pref("bounds"), pref("mc_level")
        if latents < 1 or steps < 1:
            raise ValueError("latents and steps must be positive")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if not sampler:
            raise ValueError("sampler must not be empty")
        if output_format not in OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {', '.join(OUTPUT_FORMATS)}")
        if octree_resolution < 8:
            raise ValueError("octree_resolution must be at least 8")
        if bounds <= 0 or not math.isfinite(bounds):
            raise ValueError("bounds must be finite and positive")
        if not math.isfinite(mc_level):
            raise ValueError("mc_level must be finite")
        return ResolvedMesh(
            prompt=self.prompt, negative_prompt=pref("negative_prompt"),
            image=self.image, latents=latents, steps=steps,
            guidance_scale=guidance_scale, sampler=sampler,
            seed=secrets.randbits(32) if self.seed is None or self.seed == -1 else self.seed,
            output_format=output_format, octree_resolution=octree_resolution,
            bounds=bounds, mc_level=mc_level,
            lora_specs=lora_specs_by_role(self.lora_specs),
        )


__all__ = ["OUTPUT_FORMATS", "MeshSpec", "ResolvedMesh", "mode_of"]
