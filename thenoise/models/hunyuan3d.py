"""Hunyuan3D shape adapter: one file, a reference picture in, a mesh out."""
from __future__ import annotations

import logging
from typing import Callable

import torch

from thenoise.dit.hunyuan3d import (
    Hunyuan3DCodecParams,
    Hunyuan3DConditionerParams,
    Hunyuan3DShapeParams,
    checkpoint_shapes,
    codec_params,
    conditioner_params,
    is_hunyuan3d_shape,
    load_hunyuan3d_image_encoder,
    load_hunyuan3d_shape_codec,
    load_hunyuan3d_shape_dit,
    shape_params,
)
from thenoise.mesh.extract import Mesh, extract_surface
from thenoise.mesh.octree import sample_sdf
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    VISION_REF_METHOD,
    normalize_keys,
)
from thenoise.models.config import (
    EncodePromptArgs,
    MeshDecodeParams,
    MeshSamplingParams,
    ModelConfig,
)
from thenoise.samplers import sigma_steps
from thenoise.utils.image_tensor import ReferenceSizing

logger = logging.getLogger(__name__)


class Hunyuan3DShapeModel(DiffusionModel):
    """Hunyuan3D 2.1 image-to-shape: DINOv2 context, flow-matched DiT, SDF codec."""

    name = "hunyuan3d-shape"

    MEDIUM = "mesh"
    REQUIRED_COMPONENTS = ()
    REF_METHODS = (VISION_REF_METHOD,)

    CAPABILITIES = {**DiffusionModel.CAPABILITIES,
                    "mesh": {"i2d": True, "t2d": False}}

    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 30,
        "guidance_scale": 5.0,
        "sampler": "euler",
        "negative_prompt": "",
        "num_latents": 4096,
        "octree_resolution": 384,
        "bounds": 1.01,
        "mc_level": 0.0,
    }

    #: Whole DINOv2 patches, aspect preserved: the tower interpolates its position grid.
    REFERENCE_SIZING = ReferenceSizing(fit="area", cap=518 * 518, align=14)

    @staticmethod
    def detect(f) -> bool:
        """True for the shared shape file: DiT, conditioner and codec in one."""
        return is_hunyuan3d_shape(normalize_keys(f.keys()))

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)
        shapes = checkpoint_shapes(config.dit_path)
        self.shape_params: Hunyuan3DShapeParams = shape_params(shapes)
        self.conditioner_params: Hunyuan3DConditionerParams = conditioner_params(shapes)
        self.codec_params: Hunyuan3DCodecParams = codec_params(shapes)

        target = self._load_target(config.dit_path)
        logger.info("Loading Hunyuan3D shape DiT from %s", config.dit_path)
        self.dit = load_hunyuan3d_shape_dit(config.dit_path, self.shape_params,
                                            device=target, dtype=config.dtype)
        logger.info("Loading its DINOv2 conditioner (%d positions)",
                    self.conditioner_params.num_tokens)
        self.conditioner = load_hunyuan3d_image_encoder(
            config.dit_path, self.conditioner_params, device=target, dtype=config.dtype)
        logger.info("Loading its shape codec (%d latent layers)", self.codec_params.layers)
        self.codec = load_hunyuan3d_shape_codec(config.dit_path, self.codec_params,
                                                device=target, dtype=config.dtype)

        self.memory.register("dit", self.dit)
        self.memory.register("conditioner", self.conditioner)
        self.memory.register("codec", self.codec)
        logger.info("Hunyuan3D shape model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        """The reference picture through the tower; this model reads no text at all."""
        images = self.conditioned_images(args)
        if not images:
            raise ValueError(f"{self.name} conditions on a picture: an image is required")
        cond = self.conditioner(images, self.dtype)
        null = (self.conditioner.unconditional(cond) if args.guidance_scale > 1.0
                else None)
        return Conditioning(cond=cond, null=null)

    def init_latents(self, params: MeshSamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        shape = (1, params.num_latents, self.shape_params.in_channels)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def schedule(self, params: MeshSamplingParams) -> list[Step]:
        """A uniform sigma grid unless the request brought one; ``t`` is the sigma itself."""
        grid = self.pref("sigmas") or torch.linspace(1.0, 0.0, params.steps + 1).tolist()
        return sigma_steps(grid, torch.device(self.device), torch.float32)

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        predicted = self._velocity(latents, t, cond.cond)
        if guidance_scale > 1.0 and cond.null is not None:
            unguided = self._velocity(latents, t, cond.null)
            return unguided + guidance_scale * (predicted - unguided)
        return predicted

    def _velocity(self, latents: torch.Tensor, sigma: torch.Tensor,
                  context: torch.Tensor) -> torch.Tensor:
        """The DiT's ``dx/dt`` as ``dx/dsigma``, which is what ``x -= delta * v`` descends."""
        t = (1.0 - sigma).expand(latents.shape[0]).to(dtype=latents.dtype)
        return -self.dit(latents, t, context)

    # ------------------------------------------------------------ the mesh half
    def sdf_function(self, latents: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """One SDF query to ``[M]``, over tokens refined once for the whole volume."""
        tokens = self.codec.tokens(latents / self.codec.scale_factor)

        def sdf(points: torch.Tensor) -> torch.Tensor:
            return self.codec.sdf(tokens, points[None])[0]

        return sdf

    def decode_mesh(self, latents: torch.Tensor, params: MeshDecodeParams) -> Mesh:
        """Latents to triangles: refine once, sample the narrow band, extract its surface."""
        volume = sample_sdf(self.sdf_function(latents), params.octree_resolution,
                            device=self.device, bounds=params.bounds,
                            level=params.mc_level, chunk=params.chunk)
        return extract_surface(volume)


__all__ = ["Hunyuan3DShapeModel"]
