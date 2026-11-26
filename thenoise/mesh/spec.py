"""The mesh request contract, independent of FastAPI and model libraries."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from thenoise.mesh.finish.options import MeshFinishOptions, options_from
from thenoise.mesh.finish.smooth import SPOT_LIMIT, SPOT_RADIUS_LIMIT
from thenoise.utils.lora_roles import LoraSpecs, lora_specs_by_role
from thenoise.utils.options import wire_fields

from thenoise.utils.spec import (Picture, Positive, PositiveFinite, Seed, prefer,
                                resolve_seed)

OUTPUT_FORMATS = ("glb", "obj", "stl")
DEFAULT_MESH_SCALE = 1.0

_LIMITS = {
    "latents": (1, 16384, "the attention inside them is quadratic"),
    "octree_resolution": (8, 768, "each step is eight times the memory of the one below"),
    "bounds": (0.05, 8.0, "a shape fills about ±1 of the volume"),
    "mc_level": (-0.5, 0.5, "the codec's field crosses near zero"),
    "scale": (0.001, 1000.0, "a mesh is read at about ±1, so past this someone is "
                             "reading a unit, not sizing a shape"),
}

KNOB_STEPS = {
    "octree_resolution": (("draft", 0.25), ("fine", 0.5), ("standard", 1), ("ultra", 2)),
    "latents": (("lean", 0.5), ("standard", 1), ("detailed", 2)),
    "bounds": (("tight", 0.9), ("standard", 1), ("roomy", 1.2), ("wide", 1.5)),
    "mc_level": (("sharp", 0.5), ("standard", 1), ("fatten", 2), ("extra fatten", 4)),
    "scale": (("×0.5", 0.5), ("×1", 1), ("×2", 2), ("×4", 4)),
}

KNOB_PREFS = {"latents": "num_latents"}

_WHOLE_KNOBS = ("latents", "octree_resolution")


def limits(knob: str) -> tuple[float, float, str]:
    """``knob``'s edges and the reason for them: the check, the refusal and the menu."""
    try:
        return _LIMITS[knob]
    except KeyError:
        raise KeyError(f"no limit is declared for {knob!r}; known: "
                       f"{', '.join(sorted(_LIMITS))}") from None


def knob_menu(defaults: Mapping[str, Any]) -> dict[str, list[dict]]:
    """Each step's word with the model's own value scaled by its ratio, a step whose number
    a request would be refused for left out whole."""
    menu = {}
    for knob, steps in KNOB_STEPS.items():
        own = defaults.get(KNOB_PREFS.get(knob, knob))
        if own is None:
            continue
        low, high, _ = limits(knob)
        offered = []
        for name, ratio in steps:
            value = (round(own * ratio) if knob in _WHOLE_KNOBS
                     else round(own * ratio, 10))
            if low <= value <= high and value not in [one["value"] for one in offered]:
                offered.append({"value": value, "name": name})
        menu[knob] = offered
    return menu


SHAPE_MODE = "shape"


def mode_of(images: tuple[bytes, ...]) -> str:
    """``t2d``, ``i2d`` or ``mv2d``, from how many pictures a request carries."""
    if not images:
        return "t2d"
    return "i2d" if len(images) == 1 else "mv2d"


@dataclass(frozen=True)
class ResolvedMesh:
    images: tuple[bytes, ...]
    from_shape: str | None
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
    finish: MeshFinishOptions
    smooth_spots: tuple[tuple[float, float, float, float], ...]

    @property
    def mode(self) -> str:
        """``shape`` stands in for the image modes: their conditioning is already spent."""
        return SHAPE_MODE if self.from_shape else mode_of(self.images)

    def public(self) -> dict:
        return {
            "mode": self.mode, "from_shape": self.from_shape,
            "latents": self.latents, "steps": self.steps,
            "guidance_scale": self.guidance_scale, "sampler": self.sampler,
            "attention": self.attention,
            "seed": self.seed, "output_format": self.output_format,
            "octree_resolution": self.octree_resolution, "bounds": self.bounds,
            "mc_level": self.mc_level, "scale": self.scale,
            "lora_specs": self.lora_specs, "finish": asdict(self.finish),
            "smooth_spots": len(self.smooth_spots),
        }


class _MeshBody(BaseModel):
    """The shape knobs; one left ``None`` takes the model's preference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: Picture | list[Picture] | None = None
    from_shape: Annotated[str, Field(min_length=1)] | None = None
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
    smooth_spots: list[tuple[float, float, float, float]] | None = None

    @property
    def images(self) -> tuple[bytes, ...]:
        """The pictures as a tuple, however many the request sent."""
        if self.image is None:
            return ()
        return tuple(self.image) if isinstance(self.image, list) else (self.image,)

    @property
    def mode(self) -> str:
        return mode_of(self.images)

    @field_validator("smooth_spots")
    @classmethod
    def _spots_are_places(cls, value):
        if value is None:
            return None
        if len(value) > SPOT_LIMIT:
            raise ValueError(f"smooth_spots holds at most {SPOT_LIMIT} spots, "
                             f"got {len(value)}")
        for x, y, z, radius in value:
            if not all(math.isfinite(one) for one in (x, y, z, radius)):
                raise ValueError("smooth_spots wants finite x, y, z and radius")
            if not 0.0 < radius <= SPOT_RADIUS_LIMIT:
                raise ValueError(f"a spot's radius must be within {SPOT_RADIUS_LIMIT}, "
                                 f"got {radius}")
        return [tuple(one) for one in value]

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
        if self.from_shape and self.images:
            raise ValueError("from_shape reuses a shape whose image is already spent: "
                             "send either image or from_shape, not both")
        if steps < 1:
            raise ValueError("steps must be positive")
        for knob, value in (("latents", latents), ("bounds", bounds),
                            ("mc_level", mc_level), ("scale", scale),
                            ("octree_resolution", octree_resolution)):
            low, high, why = limits(knob)
            if not low <= value <= high:
                raise ValueError(f"{knob} must be between {low} and {high}, "
                                 f"got {value}: {why}")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if not sampler:
            raise ValueError("sampler must not be empty")
        if output_format not in OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {', '.join(OUTPUT_FORMATS)}")
        if self.smooth_spots and options_from(self).smooth <= 0.0:
            raise ValueError("smooth_spots says where to smooth; send smooth above zero with it")
        return ResolvedMesh(
            images=self.images, from_shape=self.from_shape,
            latents=latents, steps=steps,
            guidance_scale=guidance_scale, sampler=sampler, attention=attention,
            seed=resolve_seed(self.seed),
            output_format=output_format, octree_resolution=octree_resolution,
            bounds=bounds, mc_level=mc_level, scale=scale,
            lora_specs=lora_specs_by_role(self.lora_specs),
            finish=options_from(self),
            smooth_spots=tuple(tuple(one) for one in (self.smooth_spots or ())),
        )


MeshSpec = create_model("MeshSpec", __base__=_MeshBody, **wire_fields(MeshFinishOptions))
MeshSpec.__doc__ = "What a mesh request asks for, from cutting the surface to writing it."


__all__ = ["DEFAULT_MESH_SCALE", "KNOB_STEPS", "OUTPUT_FORMATS", "SHAPE_MODE", "MeshSpec",
           "ResolvedMesh", "knob_menu", "limits", "mode_of"]
