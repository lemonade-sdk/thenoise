"""Fibo 2.0 adapter — Bria's unified generate-and-edit model.

One set of weights does text-to-image and instruction editing, so every checkpoint is
edit-capable and the only difference between the two releases is how few steps they
want. The codec is the Flux.2 AE, so latents are the canonical 128-channel ones at 16
pixels per cell — the DiT reads and writes them directly — and the latent upscaler is
the one that codec uses.

The conditioner (``dit/fibo2/encoder``) wants PIL images, not just text: an edit shows
Qwen3-VL the images it is about to change, and a caption that asks for text in the
image shows it that text rendered. The same images also reach the DiT as reference
latents, each on its own RoPE plane and modulated at ``t = 0``, which is why the
negative branch carries no vision tokens of its own.

Per-run state is per CFG branch: each branch's prompt length decides its own gist
count, so neither branch can be padded and neither can share the other's rotary table.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from thenoise.dit.fibo2.encoder import load_fibo2_text_encoder
from thenoise.dit.fibo2.models import Fibo2Params, gist_count, sequence_positions
from thenoise.dit.fibo2.sampling import get_sigmas, shift
from thenoise.dit.fibo2.sizes import trained_size
from thenoise.dit.fibo2.utils import detect_params, is_fibo2, load_fibo2_dit
from thenoise.models.base import Conditioning, DiffusionModel, Step, normalize_keys
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, SesquiLSRUpscaler
from thenoise.utils.image_tensor import cover_crop
from thenoise.utils.math import round_up
from thenoise.vae import load_flux2_vae

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image

logger = logging.getLogger(__name__)


class Fibo2Model(DiffusionModel):
    name = "fibo2"

    # Edit Turbo's recipe. Dev is the non-distilled parent of the same architecture
    # and wants ``--steps 50 --guidance-scale 5``; nothing in either file says which
    # it is, so the few-step one is the default and the flag is the escape hatch.
    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 4,
        "guidance_scale": 1.0,
        "sampler": "euler",
        "ref_method": "index_timestep_zero",
    }

    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": True}
    KV_CACHED_SLICE = "suffix"

    REF_METHODS = ("index_timestep_zero",)

    #: Fused sub-projections a LoRA may train apart, in each fused matrix's row order.
    lora_fusions = {
        "qkv": ("to_q", "to_k", "to_v"),
        "to_kv": ("to_k", "to_v"),
        "w13": ("w1", "w3"),
    }

    @staticmethod
    def detect(f) -> bool:
        """The Perceiver's queries, over a single-stream block and a bare head."""
        return is_fibo2(normalize_keys(f.keys()))

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.params: Fibo2Params = detect_params(config.dit_path)

        logger.info("Loading Fibo 2.0 DiT from %s", config.dit_path)
        self.dit = load_fibo2_dit(
            config.dit_path, self.params, device=self._load_target(config.dit_path), dtype=config.dtype
        )

        logger.info("Loading Fibo 2.0 text encoder (Qwen3-VL-4B) from %s", config.text_encoder_path)
        self.text_encoder = load_fibo2_text_encoder(
            config.text_encoder_path, dtype=config.dtype, device=self._load_target(config.text_encoder_path)
        )

        logger.info("Loading Fibo 2.0 VAE (Flux.2) from %s", self.vae_path)
        self.vae = load_flux2_vae(self.vae_path, device=self._load_target(self.vae_path))

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("Fibo 2.0 model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        """Prompt (and the images to edit) -> the six text bundles the DiT reads.

        The negative branch is text alone: an edit's images reach it through the
        reference latents, and the null caption it encodes was trained without any.
        """
        cond = self.text_encoder(args.prompt, args.image)
        null = None
        if args.guidance_scale > 1.0:
            null = self.text_encoder(args.negative_prompt)
        return Conditioning(cond=cond, null=null)

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
        """Each branch's rotary table and gist tokens, built ONCE before the loop.

        The latent needs no packing. Everything that depends on the prompt length —
        the gist count, and with it the whole ``[gist, image, references]`` layout —
        is fixed here, per branch, so the per-step ``denoise_step`` stays a pure DiT
        forward. Safe under the lock.
        """
        if ref and ref_method not in self.REF_METHODS:
            raise ValueError(
                f"unsupported ref_latents_method {ref_method!r}; Fibo 2.0 always "
                "conditions its reference tokens at timestep zero, on a rope plane of "
                "their own"
            )
        dev = torch.device(self.device)
        x = latents.to(device=dev, dtype=self.dtype)
        height, width = x.shape[-2], x.shape[-1]
        self._refs = [r.to(device=dev, dtype=self.dtype) for r in ref or ()]

        self.dit.pe_embedder.clear()
        self._branches: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        context = cond.cond.to(device=dev, dtype=self.dtype)
        self._branches["cond"] = (context, self._prepare_branch("cond", context, height, width))
        if cond.null is not None:
            null = cond.null.to(device=dev, dtype=self.dtype)
            self._branches["uncond"] = (null, self._prepare_branch("uncond", null, height, width))

        self.start_kv_caches(
            params,
            has_reference=bool(self._refs),
            has_uncond=params.guidance_scale > 1.0 and "uncond" in self._branches,
        )
        return x

    def _prepare_branch(
        self, branch: str, context: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """One branch's rotary table under ``branch``; returns its gist tokens.

        The table stays in float32, which is the precision the reference builds it in
        and reads it at.
        """
        refs = [(r.shape[-2], r.shape[-1]) for r in self._refs]
        positions = sequence_positions(
            gist_count(context.shape[1], self.params.max_gist_tokens),
            height, width, refs, self.params.max_gist_tokens, device=self.device,
        )
        self.dit.pe_embedder.store(branch, positions)
        return self.dit.gist_tokens(context)

    def schedule(self, params: SamplingParams) -> list[Step]:
        ts = get_sigmas(params.steps, torch.device(self.device))
        # Step.t is the flow timestep (1 -> 0); delta = t_i - t_{i+1}.
        return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(params.steps)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One DiT forward per branch, combined into the velocity.

        The model's raw output *is* the velocity the flow ODE integrates, which is
        what the shared sampler expects back.
        """
        dev = torch.device(self.device)
        t_full = torch.full((len(latents),), float(t), dtype=latents.dtype, device=dev)
        with torch.autocast(device_type=dev.type, dtype=self.dtype):
            pos = self._dit_forward(latents, t_full, "cond")
            if guidance_scale > 1.0 and "uncond" in self._branches:
                neg = self._dit_forward(latents, t_full, "uncond")
                v = neg + guidance_scale * (pos - neg)
            else:
                v = pos
        return v

    def _dit_forward(
        self, x: torch.Tensor, t_full: torch.Tensor, branch: str
    ) -> torch.Tensor:
        """One forward on one conditioning branch: its own text, gist and positions."""
        context, gist = self._branches[branch]
        return self.dit(
            x, t_full, context, gist, self.dit.pe_embedder[branch],
            self._refs, self.kv_cache(branch),
        )

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Drop the run's caches and per-branch state before anything decodes."""
        self.end_kv_caches()
        self._branches = {}
        self._refs = []
        return latents

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Input pixels (``[C,H,W]`` in [-1, 1]) -> the canonical latent.

        The canonical latent is also what the DiT reads: a reference goes over unpacked,
        on the RoPE plane its order among them gives it.
        """
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))


    def fit_references(self, images: List[Image.Image],
                       target: Tuple[int, int]) -> List[Image.Image]:
        """A lone reference is laid over the canvas; several each keep their own shape.

        An edit lines up with a single reference pixel for pixel — at the same size the
        two share every ``(h, w)`` coordinate and differ only in the plane they ride — so
        covering the canvas and cropping the overhang is what keeps the result framed
        like the image it came from. Several references cannot share one canvas aspect,
        so they take the shared per-image fit instead.
        """
        if len(images) == 1:
            return [cover_crop(images[0], *target)]
        return super().fit_references(images, target)

    def default_edit_size(self, width: int, height: int) -> Tuple[int, int]:
        """The grid the model was trained at for the reference's aspect ratio."""
        return trained_size(width / height)

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # One token per latent cell, so the VAE's own compression is the grid step.
        align = self.vae.spatial_compression
        return round_up(width, align), round_up(height, align)

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma under the model's flow shift (ER-SDE needs sigma_0 < 1)."""
        return shift(1.0 - percent)

    # -------------------------------------------------------------- upscaling
    def _create_upscaler(self) -> LatentUpscaler:
        return SesquiLSRUpscaler("flux2", device=self.device, dtype=self.dtype)


__all__ = ["Fibo2Model"]
