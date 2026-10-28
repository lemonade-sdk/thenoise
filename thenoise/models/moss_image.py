"""MOSS-Image adapter — joint-stream DiT + Qwen3.5-9B + the Flux.2 AE.

OpenMOSS' text-to-image model. What is new here is the DiT (``dit/moss_image``) and
its conditioner; the codec is the Flux.2 AE, so latents are the canonical 128-channel
ones at 16 pixels per cell and the latent upscaler is the one that codec uses.

Text-to-image only: the architecture has no reference-latent path at all — every
token of one image goes through the same joint stream. The checkpoints are not
distilled, so the step count is a plain quality/time trade.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

from thenoise.dit.moss_image.encoder import load_moss_image_text_encoder
from thenoise.dit.moss_image.models import MossImageParams, joint_positions
from thenoise.dit.moss_image.sampling import get_schedule
from thenoise.dit.moss_image.utils import detect_params, load_moss_image_dit
from thenoise.models.base import Conditioning, DiffusionModel, Step, normalize_keys
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, SesquiLSRUpscaler
from thenoise.utils.math import round_up
from thenoise.vae import load_flux2_vae

logger = logging.getLogger(__name__)


class MossImageModel(DiffusionModel):
    name = "moss_image"

    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 25,
        "guidance_scale": 3.0,
    }

    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": False, "kv_cache": False}

    REFINE_DENOISE = 0.1

    # A LoRA trained on separate q/k/v projections is stacked onto the fused matrix
    # before matching, in that matrix's row order.
    lora_fusions = {"qkv": ("q", "k", "v")}

    #: Joint blocks the **unconditional** branch does not run: the release recipe
    #: drops one from the negative forward, which measurably sharpens the CFG
    #: direction. Guidance <= 1 never runs that forward, so it is unaffected.
    UNCOND_SKIP_BLOCKS = frozenset({22})

    #: Shared modulation plus fused up/gate FFN: together, on their own here.
    _FINGERPRINT = ("shared_block_modulation.", "layers.0.proj.ffn_up_gate.")

    #: Per CFG branch: the joint ``[image, text]`` table, then the text-only one the
    #: front-end reads. Branches cannot share one because the image sits after the
    #: text on the rope ``f`` axis.
    _BRANCH_TABLES = {"cond": ("joint", "txt"), "uncond": ("joint_uncond", "txt_uncond")}

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a MOSS-Image DiT."""
        keys = list(normalize_keys(f.keys()))
        return all(
            any(key.startswith(prefix) for key in keys) for prefix in MossImageModel._FINGERPRINT
        )

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.params: MossImageParams = detect_params(config.dit_path)

        logger.info("Loading MOSS-Image DiT from %s", config.dit_path)
        self.dit = load_moss_image_dit(
            config.dit_path, self.params, device=self.offload_device, dtype=config.dtype
        )

        logger.info(
            "Loading MOSS-Image text encoder (Qwen3.5-9B) from %s", config.text_encoder_path
        )
        self.text_encoder = load_moss_image_text_encoder(
            config.text_encoder_path, dtype=config.dtype, device=self.offload_device
        )

        logger.info("Loading MOSS-Image VAE (Flux.2) from %s", self.vae_path)
        self.vae = load_flux2_vae(self.vae_path, device=self.device)

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("MOSS-Image model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        """Encode the prompt, and the negative under CFG — each at its own length."""
        cond = self.text_encoder(args.prompt)
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
        ref: Optional[torch.Tensor] = None,
        ref_method: str = "index",
    ) -> torch.Tensor:
        """Build each branch's rotary tables and refined text, ONCE.

        The latent needs no packing: the DiT reads the canonical 4D latent and answers
        in it. The front-end output is prepared here because no timestep touches it.
        """
        dev = torch.device(self.device)
        x = latents.to(device=dev, dtype=self.dtype)
        height, width = x.shape[-2], x.shape[-1]

        self._txt = cond.cond.to(device=dev, dtype=self.dtype)
        self._un_txt = None if cond.null is None else cond.null.to(device=dev, dtype=self.dtype)

        self.dit.pe_embedder.clear()
        self._txt_states: Dict[str, Optional[torch.Tensor]] = {
            "cond": self._prepare_branch("cond", self._txt, height, width)
        }
        self._txt_states["uncond"] = (
            None if self._un_txt is None else self._prepare_branch("uncond", self._un_txt, height, width)
        )
        return x

    def _prepare_branch(self, branch: str, ctx: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """One branch's tables and refined tokens, from its own prompt length.

        A padded branch would move the whole image, which sits at ``f = text_length``.
        """
        joint_name, txt_name = self._BRANCH_TABLES[branch]
        ids = joint_positions(height, width, ctx.shape[1], device=self.device)
        self.dit.pe_embedder.store(joint_name, ids, dtype=self.dtype)
        self.dit.pe_embedder.store(txt_name, ids[height * width:], dtype=self.dtype)
        return self.dit.text_states(ctx, self.dit.pe_embedder[txt_name])

    def schedule(self, params: SamplingParams) -> list[Step]:
        image_ts = get_schedule(params.steps)
        # Step.t is the flow timestep (1 -> 0); delta = t_i - t_{i+1}.
        return [Step(t=image_ts[i], delta=image_ts[i] - image_ts[i + 1])
                for i in range(params.steps)]

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
            if guidance_scale > 1.0 and self._txt_states["uncond"] is not None:
                neg = self._dit_forward(latents, t_full, "uncond")
                v = neg + guidance_scale * (pos - neg)
            else:
                v = pos
        return v

    def _dit_forward(self, x: torch.Tensor, t_full: torch.Tensor, branch: str) -> torch.Tensor:
        """One forward on one conditioning branch: its own text, positions, depth."""
        joint, _ = self._BRANCH_TABLES[branch]
        return self.dit(
            x,
            t_full,
            self._txt_states[branch],
            self.dit.pe_embedder[joint],
            self.UNCOND_SKIP_BLOCKS if branch == "uncond" else (),
        )

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # One token per latent cell, so the VAE's own compression is the grid step.
        align = self.vae.spatial_compression
        return round_up(width, align), round_up(height, align)

    # -------------------------------------------------------------- upscaling
    def _create_upscaler(self) -> LatentUpscaler:
        return SesquiLSRUpscaler("flux2", device=self.device, dtype=self.dtype)


__all__ = ["MossImageModel"]
