"""The mesh pipeline controller."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable

from thenoise.inference import inference
from thenoise.jobs import Cancellation, stepping
from thenoise.models.config import EncodePromptArgs, MeshDecodeParams, MeshSamplingParams
from thenoise.samplers import check_sampler, create_sampler
from thenoise.stages import Stages
from thenoise.utils.image_tensor import load_image
from thenoise.utils.lora_roles import check_lora_roles
from thenoise.utils.pipeline_cache import PipelineCache, content_digest

from .finish import finish_mesh
from .spec import MeshSpec, ResolvedMesh
from .write import write_mesh

logger = logging.getLogger(__name__)


class MeshPipelineController:
    """The shape pipeline: picture, denoise, volume, finish, file. Nothing past
    sampling caches."""

    def __init__(self, model):
        self.model = model
        self._cache = PipelineCache()

    def list_loras(self) -> list[str]:
        return self.model.list_loras()

    def resolve(self, spec: MeshSpec) -> ResolvedMesh:
        """Refuse what this model cannot serve, then apply its defaults."""
        model = self.model
        capacity = model.capability("mesh").get(spec.mode)
        if not capacity:
            if spec.mode == "mv2d":
                raise ValueError(f"{model.name} conditions on one picture: it has no "
                                 f"multi-view tower, and {len(spec.images)} were sent")
            raise ValueError(f"model '{model.name}' does not support {spec.mode}")
        if spec.mode == "mv2d" and len(spec.images) > capacity:
            raise ValueError(f"{model.name} takes at most {capacity} views, "
                             f"got {len(spec.images)}")
        resolved = spec.resolve(model.default_prefs)
        check_sampler(resolved.sampler)
        model.resolve_attention(resolved.attention)
        check_lora_roles(resolved.lora_specs, model.lora_components())
        return resolved

    def render(self, spec: ResolvedMesh, token: Cancellation,
               progress: Callable[[str, int, int], None], output: Path) -> None:
        """Condition -> denoise -> volume -> file, checking ``token`` throughout."""
        model = self.model
        images = [load_image(io.BytesIO(one)) for one in spec.images]
        with inference():
            if spec.lora_specs and not model.lora_dir:
                raise ValueError("mesh LoRAs require --lora-dir")
            params = MeshSamplingParams(
                height=0, width=0, steps=spec.steps, seed=spec.seed,
                guidance_scale=spec.guidance_scale, sampler=spec.sampler,
                attention=spec.attention, num_latents=spec.latents,
            )
            stages = Stages(model, self._cache)
            # A shape has no pixel grid of its own: its token count sizes the reserve.
            work = spec.latents

            cond = stages.run(
                model.encode_prompt,
                EncodePromptArgs(prompt="", guidance_scale=spec.guidance_scale,
                                 image=images or None),
                resident=("conditioner",),
                pixels=sum(image.width * image.height for image in images) or work,
                cache="prompt", key=self._cache_key(spec),
            )
            token.check()

            latents = model.init_latents(params)

            def wear_loras():
                # Empty specs included: a LoRA the last shape wore must come off this one.
                for role, component in model.lora_components().items():
                    model.switch_loras(spec.lora_specs.get(role), component, role)

            x = stages.run(self._denoise, latents, model.fuse_text(cond), params, spec,
                           token, progress, resident=("dit",), pixels=work,
                           prepare=wear_loras)
            token.check()

            mesh = stages.run(
                model.decode_mesh, x,
                MeshDecodeParams(octree_resolution=spec.octree_resolution,
                                 bounds=spec.bounds, mc_level=spec.mc_level),
                resident=("codec",), pixels=work,
            )
            mesh = finish_mesh(mesh.scaled(spec.scale), spec.finish, token.check,
                               progress)
            write_mesh(mesh, output, spec.output_format)
            logger.info("Mesh: %d vertices, %d triangles written at resolution %d",
                        mesh.vertices.shape[0], mesh.faces.shape[0],
                        spec.octree_resolution)

    @staticmethod
    def _cache_key(spec: ResolvedMesh) -> tuple:
        """The pictures decide the conditioning, and the guidance decides whether the
        unguided branch is built at all, so both belong in the key."""
        return ("prompt", spec.guidance_scale,
                content_digest(load_image(io.BytesIO(one)) for one in spec.images))

    def _denoise(self, latents, cond, params, spec: ResolvedMesh, token: Cancellation,
                 progress):
        """One solver run, from a fresh solver: a retried stage must not carry history."""
        return create_sampler(params.sampler, self.model).sample(
            latents, list(self.model.schedule(params)), cond, spec.guidance_scale,
            params.seed, step_callback=stepping(token, progress, "sampling", 0,
                                                spec.steps))


__all__ = ["MeshPipelineController"]
