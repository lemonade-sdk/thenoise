"""The mesh pipeline controller."""
from __future__ import annotations

import io
import logging
import time
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
    """The shape pipeline: picture, denoise, volume, file, each stage behind the cache."""

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
                raise ValueError(f"{model.name} conditions on one image: it has no "
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
               progress: Callable[[str, int, int], None], output: Path) -> str:
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
            # A shape moves no pixel grid of its own: its token count sizes the claim.
            work = spec.latents

            progress("conditioning")
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

            key = self._sampling_key(spec)
            # A sampling hit implies a prompt hit: the sampling key carries it.
            again = self._cache.hit("sampling", key)
            began = time.perf_counter()
            x = stages.run(self._denoise, latents, model.fuse_text(cond), params, spec,
                           token, progress, resident=("dit",), pixels=work,
                           prepare=wear_loras,
                           cache="sampling", key=key)
            how = ("cut from the latents this server still held, nothing resampled"
                   if again else f"shaped in {time.perf_counter() - began:.1f} s")
            token.check()

            progress("cutting")
            mesh = stages.run(
                model.decode_mesh, x,
                MeshDecodeParams(octree_resolution=spec.octree_resolution,
                                 bounds=spec.bounds, mc_level=spec.mc_level),
                resident=("codec",), pixels=work,
            )
            logger.info("Cut: %d triangles; finish %s", mesh.faces.shape[0], spec.finish)
            mesh = finish_mesh(mesh.scaled(spec.scale), spec.finish, token.check,
                               progress)
            drawn = write_mesh(mesh, output, spec.output_format)
            logger.info("Mesh: %d vertices, %d triangles written at resolution %d",
                        drawn.vertices.shape[0], drawn.faces.shape[0],
                        spec.octree_resolution)
        return how

    @staticmethod
    def _cache_key(spec: ResolvedMesh) -> tuple:
        """The pixels condition; the guidance decides whether an unguided branch exists."""
        return ("prompt", spec.guidance_scale,
                content_digest(load_image(io.BytesIO(one)) for one in spec.images))

    @classmethod
    def _sampling_key(cls, spec: ResolvedMesh) -> tuple:
        """Another seed, sampler, LoRA or token count is another shape, so all four are keyed."""
        worn = tuple(sorted((role, tuple(files))
                            for role, files in spec.lora_specs.items()))
        return (*cls._cache_key(spec), spec.seed, spec.steps, spec.sampler,
                spec.latents, worn)

    def _denoise(self, latents, cond, params, spec: ResolvedMesh, token: Cancellation,
                 progress):
        """One solver run, from a fresh solver: a retried stage must not carry history."""
        return create_sampler(params.sampler, self.model).sample(
            latents, list(self.model.schedule(params)), cond, spec.guidance_scale,
            params.seed, step_callback=stepping(token, progress, "sampling", 0,
                                                spec.steps))


__all__ = ["MeshPipelineController"]
