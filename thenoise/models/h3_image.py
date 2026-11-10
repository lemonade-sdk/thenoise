"""H3-Image adapter — 50-layer packed DiT + truncated Qwen3-VL-32B + a 24ch 16x codec.

A distilled model with a fixed recipe: six steps along the sigma grid baked into the
checkpoint, guidance folded into the weights, and an output head that only makes sense
on that grid (see :mod:`thenoise.dit.h3_image.sampling`). Sizes are the model's own
canvas: 2048x2048 pixels of area, any aspect, rounded to 32.

Editing feeds a reference in twice, deliberately: the language model sees it as vision
tokens inside the prompt (short edge 1024), and the DiT sees its latent as rows that
are *not* denoised — pinned at a timestep of 0.999 and mixed with the noise that label
implies, so it reads as an untouchable hint rather than as picture-to-be-cleaned. Both
paths must agree on the reference's aspect ratio, so a single sizing rule
(:attr:`REFERENCE_SIZING`) drives the latent, the default canvas and, at a fixed short
edge, the conditioning copy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from thenoise.dit.h3_image import sampling as h3_sampling
from thenoise.dit.h3_image.encoder import load_h3_image_text_encoder
from thenoise.dit.h3_image.layout import PATCH_SIZE, Layout, pad_to_patch
from thenoise.dit.h3_image.models import H3ImageParams, detect_params, load_h3_image_dit
from thenoise.models.base import Conditioning, DiffusionModel, normalize_keys
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.samplers import Step
from thenoise.upscale import LatentUpscaler, VAEPixelUpscaler
from thenoise.utils.image_tensor import ReferenceSizing, resize_to_short_edge
from thenoise.utils.math import round_up
from thenoise.vae import load_h3_image_vae

logger = logging.getLogger(__name__)

#: Short edge of the picture the language model is shown. The DiT is shown a larger
#: one (up to :attr:`REFERENCE_SIZING`'s cap): the vision tower compresses 32x32
#: pixels into a token, so a bigger copy buys conditioning detail the text stream
#: cannot use, at a token cost the packed sequence can.
VISION_SHORT_EDGE = 1024

#: The model's own canvas multiple.
CANVAS_MULTIPLE = 32


@dataclass
class H3ImageConditioning(Conditioning):
    """``Conditioning`` plus the per-branch modality tags the DiT modulates by.

    ``tags`` marks each token of ``cond``: 0 where it is a vision placeholder (a
    reference's expanded patches and the markers framing them), 1 everywhere else. They
    are per-branch because the two prompts have the images at different offsets.
    """

    tags: Optional[torch.Tensor] = None
    null_tags: Optional[torch.Tensor] = None


class H3ImageModel(DiffusionModel):
    name = "h3_image"

    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "width": 2048,
        "height": 2048,
        "steps": 6,
        "guidance_scale": 1.0,
        "sampler": "euler",
        "ref_method": "index_timestep_zero",
    }

    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": True}
    KV_CACHED_SLICE = "suffix"

    REFERENCE_SIZING = ReferenceSizing(
        fit="short_edge", floor=1024, cap=2048, align=CANVAS_MULTIPLE
    )

    REF_METHODS = ("index_timestep_zero",)

    @staticmethod
    def detect(f) -> bool:
        """The packed H3 DiT without an audio stream: no ``audio_patch_proj``."""
        keys = set(normalize_keys(f.keys()))
        return (
            "video_patch_proj.weight" in keys
            and "condition_proj.weight" in keys
            and "audio_patch_proj.weight" not in keys
        )

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.params: H3ImageParams = detect_params(config.dit_path)
        logger.info("Loading H3-Image DiT from %s", config.dit_path)
        self.dit = load_h3_image_dit(
            config.dit_path, self.params,
            device=self._load_target(config.dit_path), dtype=config.dtype,
        )

        logger.info("Loading H3-Image text encoder from %s", config.text_encoder_path)
        self.text_encoder = load_h3_image_text_encoder(
            config.text_encoder_path,
            dtype=config.dtype,
            device=self._load_target(config.text_encoder_path),
        )

        logger.info("Loading H3-Image VAE from %s", self.vae_path)
        self.vae = load_h3_image_vae(self.vae_path, device=self._load_target(self.vae_path))

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("H3-Image model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(self, args: EncodePromptArgs) -> H3ImageConditioning:
        """Prompt (and references, as labelled vision blocks) -> raw conditioning.

        The negative branch sees the same pictures: they are the edit's subject, not
        something to negate.
        """
        images = [resize_to_short_edge(img, VISION_SHORT_EDGE, align=CANVAS_MULTIPLE)
                  for img in _as_list(args.image)]

        cond, cond_mask, tags = self.text_encoder(args.prompt, images)
        null = null_mask = null_tags = None
        if args.guidance_scale > 1.0:
            null, null_mask, null_tags = self.text_encoder(args.negative_prompt, images)
        return H3ImageConditioning(
            cond=cond, cond_mask=cond_mask, null=null, null_mask=null_mask,
            tags=tags, null_tags=null_tags,
        )

    def fuse_text(self, cond: Conditioning) -> H3ImageConditioning:
        """Project and refine the prompt once per run, with the DiT resident.

        The DiT's own ``condition_proj`` plus its token refiner: the prompt does not
        depend on the timestep, so paying for two attention blocks per step would buy
        nothing.
        """
        dev = torch.device(self.device)
        return H3ImageConditioning(
            cond=self.dit.refine_text(cond.cond.to(dev, self.dtype)),
            cond_mask=cond.cond_mask,
            null=(None if cond.null is None
                  else self.dit.refine_text(cond.null.to(dev, self.dtype))),
            null_mask=cond.null_mask,
            tags=getattr(cond, "tags", None),
            null_tags=getattr(cond, "null_tags", None),
        )

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        shape = (
            1, self.vae.z_dim,
            params.height // self.vae.spatial_compression,
            params.width // self.vae.spatial_compression,
        )
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
        ref: Optional[List[torch.Tensor]] = None,
        ref_method: str = "index",
    ) -> torch.Tensor:
        """Stash the packed sequence this run steps through; the latent is unchanged.

        The DiT takes the canonical latent and patches it internally, so the internal
        form here is the latent itself — what has to be built once per run is the layout
        (segment spans and rope coordinates), the reference latents, and the run's K/V
        caches.
        """
        dev = torch.device(self.device)
        x = latents.to(dev, self.dtype)
        refs = [pad_to_patch(r.to(dev, self.dtype), PATCH_SIZE)[0] for r in _as_list(ref)]

        lat_h, lat_w = x.shape[-2], x.shape[-1]
        shapes = [(r.shape[-2], r.shape[-1]) for r in refs]
        self._refs = refs
        self._seed = params.seed
        self._layout = Layout(cond.cond.shape[1], lat_h, lat_w, shapes)
        self._null_layout = (
            None if cond.null is None
            else Layout(cond.null.shape[1], lat_h, lat_w, shapes)
        )

        self.dit.pe_embedder.clear()
        self.dit.prepare_rope(self._layout, self.dtype)
        if self._null_layout is not None:
            self.dit.prepare_rope(self._null_layout, self.dtype, name="uncond")

        self.start_kv_caches(
            params,
            has_reference=bool(refs),
            has_uncond=params.guidance_scale > 1.0 and self._null_layout is not None,
        )
        return x

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        self.end_kv_caches()
        self.dit.pe_embedder.clear()
        return latents

    def begin_sampling(self, schedule: List[Step]) -> None:
        """The grid the PDD head bank averages over: this run's own sigmas."""
        sigmas = [float(step.t) for step in schedule]
        if schedule:
            sigmas.append(float(schedule[-1].t) - float(schedule[-1].delta))
        self._sample_sigmas = torch.tensor(sigmas, dtype=torch.float32)

    def schedule(self, params: SamplingParams) -> list[Step]:
        """Uniform in the distilled grid's index — the grid itself at the default steps."""
        ladder = h3_sampling.sigma_ladder(self.params.pdd_sigmas, params.steps)
        return [Step(t=ladder[i], delta=ladder[i] - ladder[i + 1])
                for i in range(params.steps)]

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma along the distilled grid, not along a straight line."""
        return h3_sampling.percent_to_sigma(self.params.pdd_sigmas, percent)

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One packed forward; ``t`` is the sigma itself."""
        sigma = float(t)
        v = self.dit(
            latents, sigma, cond.cond, self._layout, self._refs,
            tags=getattr(cond, "tags", None), seed=self._seed,
            sample_sigmas=self._sample_sigmas, kv=self.kv_cache("cond"),
        )
        if guidance_scale > 1.0 and cond.null is not None:
            neg = self.dit(
                latents, sigma, cond.null, self._null_layout, self._refs,
                tags=getattr(cond, "null_tags", None), seed=self._seed,
                sample_sigmas=self._sample_sigmas, rope="uncond",
                kv=self.kv_cache("uncond"),
            )
            return neg + guidance_scale * (v - neg)
        return v

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Round onto the canvas grid; any aspect is valid."""
        return round_up(width, CANVAS_MULTIPLE), round_up(height, CANVAS_MULTIPLE)

    def default_edit_size(self, width: int, height: int) -> Tuple[int, int]:
        """An unnamed edit is sampled at its first reference's own fitted size."""
        return self.REFERENCE_SIZING.target_size(width, height)

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    # -------------------------------------------------------------- upscaling
    def _create_upscaler(self) -> LatentUpscaler:
        """Weight-free VAE round trip: no latent upscaler has been trained for 24 channels."""
        return VAEPixelUpscaler(self.vae, scale=self.UPSCALE_SCALE)


def _as_list(value) -> list:
    """``None`` / one item / a sequence of them -> a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


__all__ = ["H3ImageConditioning", "H3ImageModel"]
