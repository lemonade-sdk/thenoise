"""Hunyuan3D 2.1 shape adapter."""
from __future__ import annotations

import logging
from typing import Callable

import torch

from thenoise.dit.hunyuan3d21 import (
    SHAPE_SIGNATURE,
    Hunyuan3D21CodecParams,
    Hunyuan3D21ConditionerParams,
    Hunyuan3D21ShapeParams,
    codec_params,
    conditioner_params,
    load_hunyuan3d21_image_encoder,
    load_hunyuan3d21_shape_codec,
    load_hunyuan3d21_shape_dit,
    shape_params,
)
from thenoise.mesh.extract import Mesh, extract_surface
from thenoise.mesh.octree import sample_sdf
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    VISION_REF_METHOD,
)
from thenoise.models.config import (
    EncodePromptArgs,
    MeshDecodeParams,
    MeshSamplingParams,
    ModelConfig,
)
from thenoise.samplers import sigma_steps
from thenoise.utils.image_tensor import cover_crop

logger = logging.getLogger(__name__)

REFERENCE_SIDE = 518

NEAR_BAND = 0.95


class Hunyuan3D21ShapeModel(DiffusionModel):
    name = "hunyuan3d21"

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
        "output_format": "glb",
        "num_latents": 4096,
        "octree_resolution": 384,
        "bounds": 1.01,
        "mc_level": 1 / 512,
    }

    SIGNATURE = SHAPE_SIGNATURE

    def prepare_reference(self, image: Image.Image) -> Image.Image:
        """Short side to :data:`REFERENCE_SIDE`, then centre-crop the overhang: the frame the
        tower builds itself with ``Resize``+``CenterCrop``, so no aspect ratio arrives
        stretched or forces its position grid to interpolate."""
        return cover_crop(image, REFERENCE_SIDE, REFERENCE_SIDE)

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)
        header = config.header()
        shapes = header.shapes
        self.shape_params: Hunyuan3D21ShapeParams = shape_params(shapes)
        self.conditioner_params: Hunyuan3D21ConditionerParams = conditioner_params(shapes)
        self.codec_params: Hunyuan3D21CodecParams = codec_params(shapes)

        target = self._load_target(config.dit_path)
        logger.info("Loading Hunyuan3D shape DiT from %s", header.path)
        self.dit = load_hunyuan3d21_shape_dit(header, self.shape_params,
                                            device=target, dtype=config.dtype)
        logger.info("Loading its DINOv2 conditioner (%d positions)",
                    self.conditioner_params.num_tokens)
        self.conditioner = load_hunyuan3d21_image_encoder(
            header, self.conditioner_params, device=target, dtype=config.dtype)
        logger.info("Loading its shape codec (%d latent layers)", self.codec_params.layers)
        self.codec = load_hunyuan3d21_shape_codec(header, self.codec_params,
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
        """The DiT counts up from noise (``x_t = t x0 + (1-t) noise``) and predicts
        ``x0 - noise``; this engine counts down and wants ``noise - x0``, so the clock it
        is fed and the flow it answers both come inverted."""
        t = (1.0 - sigma).expand(latents.shape[0])
        return -self.dit(latents, t, context)

    # ------------------------------------------------------------ the mesh half
    def sdf_function(self, latents: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """One SDF query to ``[M]``, over tokens refined once for the whole volume.

        The codec answers an inside-positive occupancy logit; the mesh layer wants an SDF,
        negative inside, and that is also what decides which way the triangles face.
        """
        tokens = self.codec.tokens(latents / self.codec.scale_factor)

        def sdf(points: torch.Tensor) -> torch.Tensor:
            return -self.codec.sdf(tokens, points[None])[0]

        return sdf

    def decode_mesh(self, latents: torch.Tensor, params: MeshDecodeParams) -> Mesh:
        """Latents to triangles: refine once, sample the narrow band, extract its surface."""
        volume = sample_sdf(self.sdf_function(latents), params.octree_resolution,
                            device=self.device, bounds=params.bounds,
                            level=params.mc_level, chunk=params.chunk,
                            band=NEAR_BAND)
        mesh = extract_surface(volume)
        if mesh.is_empty:
            seen = latents.float()
            raise RuntimeError(
                f"no corner changed sign across level {params.mc_level}: the field spanned "
                f"{volume.span()} over {volume.queries} queries of "
                f"±{params.bounds}, latents mean {seen.mean():.3f} std {seen.std():.3f}")
        return mesh


__all__ = ["Hunyuan3D21ShapeModel"]
