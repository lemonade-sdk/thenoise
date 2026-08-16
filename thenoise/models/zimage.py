"""Z-Image (S3-DiT) adapter — supports the distilled Z-Image-Turbo checkpoint.

Turbo is an 8-NFE flow model with guidance disabled (CFG off). It shares the
canonical 4D latent format ([B, 16, H//8, W//8]) with the other models but uses
the Flux VAE (decoder) and a Qwen3 caption encoder instead of the Qwen-Image VAE.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch

from thenoise.dit.zimage import sampling as zimage_sampling
from thenoise.dit.zimage.utils import (
    find_zimage_tokenizer_dir,
    load_zimage_dit,
    load_zimage_text_encoder,
)
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    normalize_keys,
)
from thenoise.utils.math import round_up
from thenoise.vae import load_flux_vae

logger = logging.getLogger(__name__)


class ZImageModel(DiffusionModel):
    name = "zimage"

    # Distilled Turbo defaults: 8 NFEs, no CFG (guidance 1 = "off", ComfyUI's
    # convention), 1024x1024.
    DEFAULT_STEPS = 8
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    # Z-Image's flow-matching Euler schedule is exactly the shared euler sampler.
    SAMPLER = "euler"

    MAX_SEQUENCE_LENGTH = 512

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is the Z-Image S3-DiT.

        Z-Image's distinctive blocks are the caption embedder (``cap_embedder.``),
        the plain patch embedder (``x_embedder.``) and the context refiner
        (``context_refiner.``). Keys are normalized first so repackaged checkpoints
        (``model.diffusion_model.`` / ``net.``) resolve identically.
        """
        keys = list(normalize_keys(f.keys()))
        has_cap = any(k.startswith("cap_embedder.") for k in keys)
        has_context = any(k.startswith("context_refiner.") for k in keys)
        has_x_embed = any(k.startswith("x_embedder.") for k in keys)
        return has_cap and has_context and has_x_embed

    def __init__(
        self,
        *,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_dir: Optional[str] = None,
        upscaler_dir: Optional[str] = None,
    ):
        super().__init__(
            dit_path=dit_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            device=device,
            dtype=dtype,
            lora_dir=lora_dir,
            upscaler_dir=upscaler_dir,
        )

        logger.info("Loading Z-Image DiT from %s", dit_path)
        self.dit = load_zimage_dit(dit_path, device=device, dtype=dtype)
        self.dit.eval().requires_grad_(False)

        logger.info("Loading Z-Image text encoder from %s", text_encoder_path)
        self.text_encoder, self.tokenizer = load_zimage_text_encoder(
            text_encoder_path,
            dtype=dtype,
            device=device,
            tokenizer_dir=find_zimage_tokenizer_dir(text_encoder_path),
        )
        self.text_encoder.eval().requires_grad_(False)

        # Flux VAE (decoder-only).
        self.vae = load_flux_vae(self.vae_path, device=self.device, disable_mmap=True, dtype=self.dtype)
        self.vae.eval().requires_grad_(False)

        logger.info("Z-Image model ready on %s (%s)", device, dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        guidance_scale: float,
    ) -> Conditioning:
        cond = self._encode_prompt(prompt)
        null = None
        if guidance_scale > 1.0:
            null = self._encode_prompt(negative_prompt)
        return Conditioning(cond=cond, null=null)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Apply the Qwen chat template, tokenize, and return the caption embeddings.

        Mirrors the Z-Image pipeline: uses ``hidden_states[-2]`` and keeps only the
        valid (non-padded) tokens. Returns ``[1, n_valid, cap_feat_dim]``.
        """
        dev = torch.device(self.device)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(dev)
        mask = inputs.attention_mask.to(dev).bool()

        with torch.no_grad():
            out = self.text_encoder(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
            emb = out.hidden_states[-2]  # [1, seq, 2560]
            valid = emb[0][mask[0]]  # [n_valid, 2560]
        return valid.unsqueeze(0).to(self.dtype)

    def init_latents(self, height: int, width: int, seed: int) -> torch.Tensor:
        dev = torch.device(self.device)
        shape = (1, self.dit.in_channels, height // self._VAE_SCALE, width // self._VAE_SCALE)
        generator = torch.Generator(device=dev).manual_seed(seed)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        # The DiT expects an F (frame) axis: [B, C, H, W] -> [B, C, 1, H, W].
        return latents.unsqueeze(2)

    def schedule(self, steps: int, height: int, width: int) -> list[Step]:
        dev = torch.device(self.device)
        sigmas = zimage_sampling.get_sigmas(steps, dev)
        # Step.t carries the *sigma* grid (1 -> 1/steps -> 0). Both solvers consume
        # it as sigma: the Euler loop integrates ``x -= delta * v`` (delta = sigma_i -
        # sigma_{i+1}) and ER-SDE reconstructs its sigmas from Step.t. The model's
        # actual timestep ``t = 1 - sigma`` is derived in ``denoise_step``.
        return [
            Step(t=sigmas[i], delta=sigmas[i] - sigmas[i + 1])
            for i in range(steps)
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
        cap = [cond.cond[0]]   # [n_valid, cap_feat_dim]
        # ``t`` is sigma (see ``schedule``); the DiT's model timestep is ``1 - sigma``.
        t_full = torch.full((1,), 1.0 - float(t), device=dev, dtype=latents.dtype)

        with torch.no_grad():
            # The scheduler integrates ``x + dt * noise_pred`` with ``noise_pred =
            # -model_out``; our shared euler loop is ``x -= delta * v``, so the
            # velocity v must be the NEGATED DiT output.
            pos = self.dit(x_list, t_full, cap)[0].unsqueeze(0)  # [1, C, 1, H, W]
            v_pos = -pos
            if guidance_scale > 1.0 and cond.null is not None:
                neg = self.dit(x_list, t_full, [cond.null[0]])[0].unsqueeze(0)
                # CFG over velocities: v = v_uncond + g * (v_pos - v_uncond),
                # where v_pos = -pos (conditional) and v_uncond = -neg.
                v_uncond = -neg
                v = v_uncond + guidance_scale * (v_pos - v_uncond)
            else:
                v = v_pos
        return v

    def finalize_latent(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Drop the F axis back to canonical 4D: [B, C, 1, H, W] -> [B, C, H, W].
        return latents.squeeze(2)

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # The latent grid is patchified in 2x2 blocks on an 8x-VAE-compressed latent
        # (Flux VAE), so pixel dims must be multiples of 8 * 2 = 16. Round up.
        align = self._VAE_SCALE * self.dit.patch_size
        return round_up(width, align), round_up(height, align)

    def _upscale_format(self) -> str:
        """Flux VAE -> affine shift/scale latent format."""
        return "flux"

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma (used by the ER-SDE solver to nudge sigma_0 below 1).

        The shifted schedule's first sigma lands exactly on 1.0, where the ER-SDE
        solver's ``sigma/(1-sigma)`` blows up; nudge it to just below 1.
        """
        return 1.0 - percent
