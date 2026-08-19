"""Flux.2 (Flux Klein) adapter — supports the 4B and 9B Klein variants.

Flux Klein is a flow-matching MMDiT operating on the Flux.2 *packed* 128-channel
latent ``[B, 128, H//16, W//16]`` (the Flux.2 VAE packs a 32ch latent 2x2 and
normalizes it via BatchNorm). The canonical latent format here is therefore the
normalized packed 128ch latent, and both the DiT and the VAE operate on it
directly (the adapter packs/unpacks around the denoise loop only).

The Klein DiT variant (4B / 9B) is read from the checkpoint's ``img_in`` width and
selects the matching Qwen3 text encoder (4B / 8B). Distilled vs base behavior is
driven by ``guidance_scale``: distilled models default to guidance 1.0 (single
forward, no CFG); base models pass a guidance > 1.0 to enable CFG (two forwards).

The default schedule is Euler (the Flux.2 flow ODE); ER-SDE is also usable.
"""
from __future__ import annotations

import logging

import torch

from thenoise.dit.flux2.models import Flux2Params
from thenoise.dit.flux2.sampling import get_schedule, prc_img, prc_txt, scatter_ids
from thenoise.dit.flux2.utils import (
    detect_klein_params,
    find_flux2_tokenizer_dir,
    load_flux2_dit,
    load_qwen3_embedder,
)
from thenoise.models.base import Conditioning, DiffusionModel, Step, normalize_keys
from thenoise.models.config import ModelConfig, SamplingParams
from thenoise.utils.math import round_up
from thenoise.vae import load_flux2_vae

logger = logging.getLogger(__name__)


class FluxKleinModel(DiffusionModel):
    name = "flux_klein"

    # Distilled defaults (the common inference use): 4 NFEs, CFG off (guidance 1.0).
    # Base models should pass --steps 50 --guidance-scale 4.
    DEFAULT_STEPS = 4
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    # Flux.2 flow-matching Euler schedule.
    SAMPLER = "euler"

    # Packed-latent geometry (Flux.2 VAE): 128ch at 16x spatial compression.
    LATENT_CHANNELS = 128
    _PACK = 16  # pixel / packed-latent ratio

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a Flux.2 (Flux Klein) DiT.

        Flux.2's distinctive signature is the pair of separate double-stream
        modulations (``double_stream_modulation_img.`` / ``_txt.``) plus the
        single-stream modulation — unique to the Flux.2 family. Keys are normalized
        first so repackaged checkpoints (``model.diffusion_model.`` / ``net.``)
        resolve identically.
        """
        keys = list(normalize_keys(f.keys()))
        has_img = any(k.startswith("double_stream_modulation_img.") for k in keys)
        has_txt = any(k.startswith("double_stream_modulation_txt.") for k in keys)
        has_single = any(k.startswith("single_stream_modulation.") for k in keys)
        return has_img and has_txt and has_single

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        # Determine the Klein variant (4B / 9B) from the DiT checkpoint; this also
        # selects the matching Qwen3 text encoder (4B / 8B).
        self.params: Flux2Params = detect_klein_params(config.dit_path)
        self.is_8b = self.params.context_in_dim == 12288
        logger.info("Loading Flux Klein DiT (%s) from %s", self.variant_label, config.dit_path)
        self.dit = load_flux2_dit(config.dit_path, self.params, device=config.device, dtype=config.dtype)
        self.dit.eval().requires_grad_(False)

        logger.info("Loading Flux Klein text encoder (Qwen3-%s) from %s", self.text_label, config.text_encoder_path)
        self.text_encoder = load_qwen3_embedder(
            config.text_encoder_path,
            is_8b=self.is_8b,
            dtype=config.dtype,
            device=config.device,
            tokenizer_dir=find_flux2_tokenizer_dir(config.text_encoder_path),
        )

        # Flux.2 VAE (decoder-only).
        self.vae = load_flux2_vae(self.vae_path, device=self.device, disable_mmap=True, dtype=self.dtype)
        self.vae.eval().requires_grad_(False)

        logger.info("Flux Klein model (%s) ready on %s (%s)", self.variant_label, config.device, config.dtype)

    @property
    def variant_label(self) -> str:
        return "9B" if self.is_8b else "4B"

    @property
    def text_label(self) -> str:
        return "8B" if self.is_8b else "4B"

    # ------------------------------------------------------------ kernels
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        guidance_scale: float,
    ) -> Conditioning:
        cond = self.text_encoder(prompt)  # [1, 512, ctx_dim]
        null = None
        if guidance_scale > 1.0:
            null = self.text_encoder(negative_prompt)
        return Conditioning(cond=cond, null=null)

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        shape = (1, self.LATENT_CHANNELS, params.height // self._PACK, params.width // self._PACK)
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
    ) -> torch.Tensor:
        """Pack the canonical latent into DiT tokens and stash conditioning, ONCE.

        ``prc_img`` converts ``[B, 128, H//16, W//16]`` -> ``[B, seq, 128]`` tokens
        plus ``[B, seq, 4]`` position ids. The text embeddings and their (fixed)
        position ids are stashed so the per-step ``denoise_step`` stays a pure DiT
        forward. Safe under the lock.
        """
        dev = torch.device(self.device)
        x, x_ids = prc_img(latents.to(device=dev, dtype=self.dtype))
        self._img_ids = x_ids

        self._txt = cond.cond.to(device=dev, dtype=self.dtype)
        _, self._txt_ids = prc_txt(self._txt)

        if cond.null is not None:
            self._un_txt = cond.null.to(device=dev, dtype=self.dtype)
            _, self._un_txt_ids = prc_txt(self._un_txt)
        else:
            self._un_txt = self._un_txt_ids = None

        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        image_seq_len = (params.width // self._PACK) * (params.height // self._PACK)
        ts = get_schedule(params.steps, image_seq_len)
        # Step.t is the flow timestep (1 -> 0); delta = t_i - t_{i+1}. The shared
        # Euler loop integrates ``x -= delta * velocity``, matching the Flux.2
        # update ``x += (t_{i+1} - t_i) * v`` when velocity = model output.
        return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(params.steps)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One Flux.2 DiT forward (+ CFG), returning the velocity (model output).

        The Flux.2 flow ODE integrates ``x += (t_prev - t_curr) * v``, which is
        exactly the shared Euler update ``x -= delta * v`` when ``v`` is the model's
        raw output (no negation, unlike Z-Image).
        """
        dev = torch.device(self.device)
        t_full = torch.full((len(latents),), float(t), dtype=latents.dtype, device=dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=self.dtype):
            pos = self.dit(x=latents, x_ids=self._img_ids, timesteps=t_full, ctx=self._txt, ctx_ids=self._txt_ids)
            if guidance_scale > 1.0 and self._un_txt is not None:
                neg = self.dit(x=latents, x_ids=self._img_ids, timesteps=t_full, ctx=self._un_txt, ctx_ids=self._un_txt_ids)
                v = neg + guidance_scale * (pos - neg)
            else:
                v = pos
        return v

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Unpack the DiT tokens back to the canonical packed latent."""
        x = torch.cat(scatter_ids(latents, self._img_ids)).squeeze(2)  # [B, 128, H//16, W//16]
        return x

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # The packed latent is H//16 x W//16, so pixel dims must be multiples of 16.
        align = self._PACK
        return round_up(width, align), round_up(height, align)

    def _upscale_format(self) -> str:
        """Flux.2 VAE -> 128ch patched + BN-normalized latent format."""
        return "flux2"

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma (used by the ER-SDE solver to nudge sigma_0 below 1).

        The shifted schedule's first timestep lands exactly on 1.0, where the ER-SDE
        solver's ``sigma/(1-sigma)`` blows up; nudge it to just below 1.
        """
        return 1.0 - percent


__all__ = ["FluxKleinModel"]
