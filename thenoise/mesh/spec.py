"""The mesh request contract, independent of FastAPI and model libraries."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from thenoise.utils.lora_roles import LoraSpecs, lora_specs_by_role

from thenoise.utils.spec import (Picture, Positive, PositiveFinite, Seed, prefer,
                                resolve_seed)

OUTPUT_FORMATS = ("glb", "obj", "stl")
DENSITY_PRESETS = (96, 192, 384, 768)
MAX_OCTREE_RESOLUTION = 768
DEFAULT_MESH_SCALE = 1.0
TOKEN_PRESETS = (2048, 4096, 8192)
BOUNDS_PRESETS = (0.9, 1.01, 1.2, 1.5)
CUT_LEVEL_PRESETS = (1 / 1024, 1 / 512, 1 / 256, 1 / 128)
MIN_BOUNDS, MAX_BOUNDS = 0.05, 8.0
MIN_CUT_LEVEL, MAX_CUT_LEVEL = -0.5, 0.5
MAX_LATENTS = 16384


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

    image: Picture | list[Picture] | None = None
    latents: Positive | None = None
    steps: Positive | None = None
    guidance_scale: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    sampler: Annotated[str, Field(min_length=1)] | None = None
    #: ``auto``, ``full`` or ``optimized``.
    attention: str | None = None
    seed: Seed = None
    output_format: Literal["glb", "obj", "stl"] | None = None
    octree_resolution: Positive | None = None
    bounds: PositiveFinite | None = None
    mc_level: Annotated[float, Field(allow_inf_nan=False)] | None = None
    scale: PositiveFinite | None = None
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

    def resolve(self, defaults: Mapping[str, Any]) -> ResolvedMesh:
        """Fill what the request left unset from the model's defaults, then check."""
        pref = prefer(self, defaults)

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
        if latents > MAX_LATENTS:
            raise ValueError(f"latents must be at most {MAX_LATENTS}, got {latents}: the "
                             f"attention inside it is quadratic")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if not sampler:
            raise ValueError("sampler must not be empty")
        if output_format not in OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {', '.join(OUTPUT_FORMATS)}")
        if octree_resolution < 8:
            raise ValueError("octree_resolution must be at least 8")
        if not MIN_BOUNDS <= bounds <= MAX_BOUNDS:
            raise ValueError(f"bounds must be between {MIN_BOUNDS} and {MAX_BOUNDS}, "
                             f"got {bounds}: a shape fills about ±1")
        if not MIN_CUT_LEVEL <= mc_level <= MAX_CUT_LEVEL:
            raise ValueError(f"mc_level must be between {MIN_CUT_LEVEL} and "
                             f"{MAX_CUT_LEVEL}, got {mc_level}: the codec's field crosses "
                             "near 0")
        if octree_resolution > MAX_OCTREE_RESOLUTION:
            raise ValueError(f"octree_resolution must be at most {MAX_OCTREE_RESOLUTION}, "
                             f"whose grid is {(MAX_OCTREE_RESOLUTION + 1) ** 3} samples")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        return ResolvedMesh(
            images=self.images, latents=latents, steps=steps,
            guidance_scale=guidance_scale, sampler=sampler, attention=attention,
            seed=resolve_seed(self.seed),
            output_format=output_format, octree_resolution=octree_resolution,
            bounds=bounds, mc_level=mc_level, scale=scale,
            lora_specs=lora_specs_by_role(self.lora_specs),
        )


__all__ = ["BOUNDS_PRESETS", "CUT_LEVEL_PRESETS", "DEFAULT_MESH_SCALE", "DENSITY_PRESETS",
           "MAX_BOUNDS", "MAX_CUT_LEVEL", "MAX_LATENTS", "MAX_OCTREE_RESOLUTION",
           "MIN_BOUNDS", "MIN_CUT_LEVEL", "OUTPUT_FORMATS",
           "TOKEN_PRESETS", "MeshSpec", "ResolvedMesh", "mode_of"]
