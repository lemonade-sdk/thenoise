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
_Picture = Annotated[bytes, Field(min_length=1)]

OUTPUT_FORMATS = ("glb", "obj", "stl")
DENSITY_PRESETS = (96, 192, 384, 768)
MAX_OCTREE_RESOLUTION = 768
DEFAULT_MESH_SCALE = 1.0


def mode_of(images: tuple[bytes, ...]) -> str:
    """``t2d``, ``i2d`` or ``mv2d``, from how many pictures a request carries."""
    if not images:
        return "t2d"
    return "i2d" if len(images) == 1 else "mv2d"


@dataclass(frozen=True)
class ResolvedMesh:
    images: tuple[bytes, ...]
    latents: int
    steps: int
    guidance_scale: float
    sampler: str
    attention: str
    seed: int
    output_format: str
    octree_resolution: int
    bounds: float
    mc_level: float
    scale: float
    lora_specs: dict[str, list[str]]

    @property
    def mode(self) -> str:
        return mode_of(self.images)

    def public(self) -> dict:
        return {
            "mode": self.mode, "latents": self.latents, "steps": self.steps,
            "guidance_scale": self.guidance_scale, "sampler": self.sampler,
            "attention": self.attention,
            "seed": self.seed, "output_format": self.output_format,
            "octree_resolution": self.octree_resolution, "bounds": self.bounds,
            "mc_level": self.mc_level, "scale": self.scale,
            "lora_specs": self.lora_specs,
        }


class MeshSpec(BaseModel):
    """What a mesh request asks for; a field left ``None`` takes the model's preference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: _Picture | list[_Picture] | None = None
    latents: _Positive | None = None
    steps: _Positive | None = None
    guidance_scale: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    sampler: Annotated[str, Field(min_length=1)] | None = None
    #: ``auto``, ``full`` or ``optimized``.
    attention: str | None = None
    seed: int | None = None
    output_format: Literal["glb", "obj", "stl"] | None = None
    octree_resolution: _Positive | None = None
    bounds: _PositiveFinite | None = None
    mc_level: Annotated[float, Field(allow_inf_nan=False)] | None = None
    scale: _PositiveFinite | None = None
    lora_specs: LoraSpecs = None

    @property
    def images(self) -> tuple[bytes, ...]:
        """The pictures as a tuple, however many the request sent."""
        if self.image is None:
            return ()
        return tuple(self.image) if isinstance(self.image, list) else (self.image,)

    @property
    def mode(self) -> str:
        return mode_of(self.images)

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
        attention = pref("attention")
        output_format = pref("output_format")
        octree_resolution = pref("octree_resolution")
        bounds, mc_level = pref("bounds"), pref("mc_level")
        scale = self.scale
        if scale is None:
            scale = defaults.get("scale", DEFAULT_MESH_SCALE)
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
        if octree_resolution > MAX_OCTREE_RESOLUTION:
            raise ValueError(f"octree_resolution must be at most {MAX_OCTREE_RESOLUTION}, "
                             f"whose grid is {(MAX_OCTREE_RESOLUTION + 1) ** 3} samples")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        return ResolvedMesh(
            images=self.images, latents=latents, steps=steps,
            guidance_scale=guidance_scale, sampler=sampler, attention=attention,
            seed=secrets.randbits(32) if self.seed is None or self.seed == -1 else self.seed,
            output_format=output_format, octree_resolution=octree_resolution,
            bounds=bounds, mc_level=mc_level, scale=scale,
            lora_specs=lora_specs_by_role(self.lora_specs),
        )


__all__ = ["DEFAULT_MESH_SCALE", "DENSITY_PRESETS", "MAX_OCTREE_RESOLUTION",
           "OUTPUT_FORMATS", "MeshSpec", "ResolvedMesh", "mode_of"]
