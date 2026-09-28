"""Ming-Image 0.1 adapter — the Lumina DiT with two conditioning tensors and RGBA.

The kernels are Z-Image's shape (``t = 1 - sigma``, ``v = -out``) plus a second,
already-DiT-wide conditioning tensor per branch. :meth:`_encode_prompt` is the one
hole left (phase 3 of the plan); everything downstream of it is live.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from thenoise.dit.lumina.keys import has_learned_pad_tokens, is_s3dit, lumina_key_map
from thenoise.dit.ming_image import sampling as ming_sampling
from thenoise.dit.ming_image.utils import load_ming_dit
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    normalize_keys,
)
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, SesquiLSRUpscaler
from thenoise.utils.lora import FUSE_QKV
from thenoise.utils.math import round_up
from thenoise.vae import load_ming_vae

logger = logging.getLogger(__name__)


@dataclass
class MingConditioning(Conditioning):
    """``Conditioning`` plus each branch's ``direct_context``: ``[1, n_text, dim]``,
    already at DiT width, ``None`` when the branch does not exist.
    """

    cond_extra: Optional[torch.Tensor] = None
    null_extra: Optional[torch.Tensor] = None


def _branch(extra: Optional[torch.Tensor]) -> Optional[list]:
    """The DiT's per-sample list form of a branch's extra tensor (``None`` kept)."""
    return None if extra is None else [extra[0]]


class MingImageModel(DiffusionModel):
    name = "ming_image"

    # The reference's defaults, at the 1024 bucket it was trained on; the schedule's
    # shift is resolution dependent (2048 is the second bucket).
    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 12,
        "guidance_scale": 1.0,
        "sampler": "euler",
    }

    # Both exports ship one fused ``attn.qkv``, so LoRAs named after the separate
    # ``to_q/to_k/to_v`` have to fold onto it.
    lora_fusions = FUSE_QKV

    def _lora_key_map(self, key: str) -> str:
        """Resolve LoRAs named after either generation: ``to_out.0``/``norm_q``, or
        this tree's ``out``/``q_norm``.
        """
        return lumina_key_map(key)

    @staticmethod
    def detect(f) -> bool:
        """The S3-DiT signature with the sibling's pad tokens ABSENT — the only
        separator there is, since Ming-Image zero-fills and masks its padding.
        """
        keys = list(normalize_keys(f.keys()))
        return is_s3dit(keys) and not has_learned_pad_tokens(keys)

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        logger.info("Loading Ming-Image DiT from %s", config.dit_path)
        self.dit = load_ming_dit(config.dit_path, device=self.offload_device, dtype=config.dtype)
        self.dit.eval().requires_grad_(False)

        # Phase 3: the BailingMM2 conditioner. The empty slot keeps the pipeline's
        # ``ensure("text_encoder")`` a no-op until there is a module to move.
        self.text_encoder = None

        logger.info("Loading Ming-Image VAE from %s", config.vae_path)
        self.vae = load_ming_vae(self.vae_path, device=self.device, dtype=config.dtype)
        self.vae.eval().requires_grad_(False)

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("Ming-Image model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        cond, cond_extra = self._encode_prompt(args.prompt)
        null = null_extra = None
        if args.guidance_scale > 1.0:
            null, null_extra = self._encode_prompt(args.negative_prompt)
        return MingConditioning(
            cond=cond, cond_extra=cond_extra, null=null, null_extra=null_extra
        )

    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(cap_feats [1, n, cap_feat_dim], direct_context [1, m, dim])`` — phase 3."""
        raise NotImplementedError(
            "Ming-Image text encoding is not implemented yet (phase 3 of "
            "docs/ming-image-plan.md): the DiT, VAE and schedule are ready, the "
            "BailingMM2 conditioner that produces cap_feats + direct_context is not."
        )

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        # The DiT patchifies the VAE latent itself, so the VAE's geometry is the noise's.
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
    ) -> torch.Tensor:
        # The DiT expects an F (frame) axis: [B, C, H, W] -> [B, C, 1, H, W].
        latents = latents.unsqueeze(2)
        # One RoPE build per branch: each branch's positions depend on ITS caption
        # length (extra included), and neither may clear the other's table.
        self.dit.prepare_rope(
            [latents[0]], [cond.cond[0]], _branch(getattr(cond, "cond_extra", None))
        )
        if cond.null is not None:
            self.dit.prepare_rope(
                [latents[0]],
                [cond.null[0]],
                _branch(getattr(cond, "null_extra", None)),
                key="_neg",
                clear=False,
            )
        return latents

    def schedule(self, params: SamplingParams) -> list[Step]:
        dev = torch.device(self.device)
        sigmas = ming_sampling.get_sigmas(params.steps, params.height, params.width, dev)
        # ``Step.t`` carries the sigma grid; the model timestep ``t = 1 - sigma`` is
        # derived in ``denoise_step`` (see Z-Image's identical split for why).
        return [
            Step(t=sigmas[i], delta=sigmas[i] - sigmas[i + 1])
            for i in range(params.steps)
        ]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        dev = torch.device(self.device)
        x_list = [latents[0]]  # [C, 1, H, W]
        cap = [cond.cond[0]]   # [n, cap_feat_dim]
        extra = _branch(getattr(cond, "cond_extra", None))
        # ``t`` is sigma (see ``schedule``); the DiT's model timestep is ``1 - sigma``.
        t_full = torch.full((1,), 1.0 - float(t), device=dev, dtype=latents.dtype)

        with torch.no_grad():
            # Sign convention as in Z-Image: the reference integrates the negated DiT
            # output and our Euler loop subtracts, so ``v = -out``.
            pos = self.dit(x_list, t_full, cap, extra)[0].unsqueeze(0)  # [1, C, 1, H, W]
            v_pos = -pos
            if guidance_scale > 1.0 and cond.null is not None:
                neg = self.dit(
                    x_list,
                    t_full,
                    [cond.null[0]],
                    _branch(getattr(cond, "null_extra", None)),
                    rope_key="_neg",
                )[0].unsqueeze(0)
                v_uncond = -neg
                v = v_uncond + guidance_scale * (v_pos - v_uncond)
            else:
                v = v_pos
        return v

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # Drop the F axis back to the canonical 4D latent.
        return latents.squeeze(2)

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # 2x2 patches on an 8x-compressed latent: pixel dims must be multiples of 16.
        align = self.vae.spatial_compression * self.dit.patch_size
        return round_up(width, align), round_up(height, align)

    # -------------------------------------------------------------- upscaling
    def _create_upscaler(self) -> LatentUpscaler:
        """The Wan2.1 Sesqui network over Ming-Image's affine 16ch latent: same VAE
        architecture, only the normalisation differs. Borrowed, not yet measured.
        """
        return SesquiLSRUpscaler("ming", device=self.device, dtype=self.dtype)


__all__ = ["MingConditioning", "MingImageModel"]
