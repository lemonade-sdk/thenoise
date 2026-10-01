"""Mage-Flow adapter — 12-layer dual-stream DiT + Qwen3-VL-4B + Mage-VAE.

Native-resolution flow model: the latent is the VAE's raw 128-channel output at 16x
(one DiT token per latent cell, no packing), the schedule is a static shift, and sizes
are only rounded up to the VAE's 16-pixel cell. Both released checkpoints are the same
architecture, so both are edit-capable.

The codec is the Mage-VAE, but the Flux.2 AE is accepted in its place — the latent is
the same shape and the Mage space was anchored to Flux.2's — and which of the two was
passed is read off the file (see :func:`thenoise.vae.load_mage_family_vae`).

Defaults are the **Turbo** recipe, because that is what the download script ships and
no checkpoint carries a marker saying which variant it is — the CLI/API stays the
source of truth:

    base     --steps 30 --guidance-scale 5
    RL       --steps 20 --guidance-scale 5
    turbo    (the default)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from thenoise.dit.mage_flow.keys import MAGE_LAYERS, dit_block_count, is_qwen_image_family
from thenoise.dit.mage_flow.models import (
    MageFlowParams,
    latent_to_tokens,
    load_mage_flow_dit,
    tokens_to_latent,
)
from thenoise.dit.mage_flow import sampling as mage_sampling
from thenoise.dit.mage_flow.encoder import load_mage_flow_text_encoder
from thenoise.dit.mage_flow.utils import detect_params
from thenoise.dit.qwen_image.models import build_video_positions
from thenoise.models.base import Conditioning, DiffusionModel, Step, normalize_keys
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, VAEPixelUpscaler
from thenoise.utils.math import round_up
from thenoise.utils.positions import grid_positions
from thenoise.vae import load_mage_family_vae

logger = logging.getLogger(__name__)


class MageFlowModel(DiffusionModel):
    name = "mage_flow"

    # The Turbo recipe; the module docstring has the other variants.
    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 4,
        "guidance_scale": 1.0,
        "sampler": "euler",
    }

    # No KV cache: the single-row timestep embedding modulates the references at ``t``
    # like everything else, so their K/V are not step-invariant.
    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": False}

    @staticmethod
    def detect(f) -> bool:
        """The Qwen-Image block layout at Mage-Flow's depth (``dit/mage_flow/keys``)."""
        keys = list(normalize_keys(f.keys()))
        return is_qwen_image_family(keys) and dit_block_count(keys) == MAGE_LAYERS

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.params: MageFlowParams = detect_params(config.dit_path)
        logger.info("Loading Mage-Flow DiT from %s", config.dit_path)
        self.dit = load_mage_flow_dit(
            config.dit_path, self.params, device=self.offload_device, dtype=config.dtype
        )

        logger.info("Loading Mage-Flow text encoder (Qwen3-VL-4B) from %s", config.text_encoder_path)
        self.text_encoder = load_mage_flow_text_encoder(
            config.text_encoder_path, dtype=config.dtype, device=self.offload_device
        )

        logger.info("Loading Mage-Flow VAE from %s", self.vae_path)
        self.vae = load_mage_family_vae(self.vae_path, device=self.device, dtype=config.dtype)

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("Mage-Flow model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        """Prompt (and reference images) -> conditioning, via Qwen3-VL-4B.

        The references are vision tokens of the same turn for the negative branch too,
        so CFG compares "edit like this" against "don't" rather than image-conditioned
        against unconditioned.
        """
        images = args.image or None
        cond, cond_mask = self.text_encoder(args.prompt, images)

        null = null_mask = None
        if args.guidance_scale > 1.0:
            null, null_mask = self.text_encoder(args.negative_prompt, images)
        return Conditioning(cond=cond, cond_mask=cond_mask, null=null, null_mask=null_mask)

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
        """Tokens + conditioning + positions, stashed once before the loop.

        One RoPE build per run. The image entry is the target's positions followed by
        the references' and stays that way every step. The text entry is a single
        identity rotation: this model does not rotate its text tokens, and
        ``apply_rope`` broadcasts the one matrix over the whole prompt.
        """
        dev = torch.device(self.device)
        x = latent_to_tokens(latents.to(device=dev, dtype=self.dtype))
        self._txt = cond.cond.to(device=dev, dtype=self.dtype)
        self._null_txt = (
            cond.null.to(device=dev, dtype=self.dtype) if cond.null is not None else None
        )

        h, w = params.height // self.vae.spatial_compression, params.width // self.vae.spatial_compression
        img_pe = build_video_positions([(1, h, w)], device=dev)  # frame index 0 = the target

        if ref is not None:
            ref_tokens, ref_pe = [], []
            for i, ref_latent in enumerate(ref):
                tokens, positions = self.pack_reference_latent(
                    ref_latent, ref_method, ref_index=i + 1
                )
                ref_tokens.append(tokens)
                ref_pe.append(positions)
            self._ref_tokens = torch.cat(ref_tokens, dim=1)
            img_pe = torch.cat([img_pe, torch.cat(ref_pe, dim=1)], dim=1)
        else:
            self._ref_tokens = None

        self.dit.pe_embedder.clear()
        self.dit.pe_embedder.store("img", img_pe, dtype=self.dtype)
        self.dit.pe_embedder.store("txt", torch.zeros(1, 1, 3, device=dev), dtype=self.dtype)
        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        # Static shift: the grid is the same at every resolution. ``Step.t`` is the
        # shifted sigma, which is literally the timestep the DiT is fed.
        ts = mage_sampling.get_schedule(params.steps)
        return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(params.steps)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One DiT forward (+ plain CFG), returning the velocity of the target tokens.

        The CFG combination is left unnormalised: upstream's per-token renorm defaults
        to off.
        """
        dev = torch.device(self.device)
        t_full = torch.full((1,), float(t), dtype=latents.dtype, device=dev)
        img_pe = self.dit.pe_embedder["img"]
        txt_pe = self.dit.pe_embedder["txt"]

        with torch.autocast(device_type=dev.type, dtype=self.dtype):
            pos = self.dit(
                hidden_states=latents,
                encoder_hidden_states=self._txt,
                timestep=t_full,
                img_pe=img_pe,
                txt_pe=txt_pe,
                ref_tokens=self._ref_tokens,
            )
            if guidance_scale > 1.0 and self._null_txt is not None:
                neg = self.dit(
                    hidden_states=latents,
                    encoder_hidden_states=self._null_txt,
                    timestep=t_full,
                    img_pe=img_pe,
                    txt_pe=txt_pe,
                    ref_tokens=self._ref_tokens,
                )
                v = neg + guidance_scale * (pos - neg)
            else:
                v = pos
        return v

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Tokens -> canonical latent."""
        return tokens_to_latent(
            latents,
            params.height // self.vae.spatial_compression,
            params.width // self.vae.spatial_compression,
        )

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Round up to the pixel cell of one token: 512..2048 is a quality range, not a
        constraint, so no bucket quantisation and no clamping."""
        align = self._pixels_per_token
        return round_up(width, align), round_up(height, align)

    @property
    def _pixels_per_token(self) -> int:
        """The VAE's compression times the DiT's patch size."""
        return self.vae.spatial_compression * self.dit.patch_size

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "index",
        ref_index: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Canonical reference latent -> (tokens, positions) at frame index ``ref_index``.

        ``index_timestep_zero`` needs a second timestep row for the reference tokens
        and this DiT's embedding has one row for everything, so the method is rejected
        rather than silently run as ``index``.
        """
        if method != "index":
            raise ValueError(
                f"unsupported ref_latents_method {method!r} for mage_flow; only 'index' "
                "is available (the model has no timestep-zero reference conditioning, "
                "so neither is the KV cache)"
            )
        dev = torch.device(self.device)
        tokens = latent_to_tokens(latents.to(device=dev, dtype=self.dtype))
        h, w = latents.shape[-2], latents.shape[-1]
        pe = grid_positions(
            [1, h, w],
            start=[ref_index, 0, 0],
            centered=[False, True, True],
            dtype=torch.float32,
            device=dev,
        ).unsqueeze(0)
        return tokens, pe

    # -------------------------------------------------------------- upscaling
    def _create_upscaler(self) -> LatentUpscaler:
        # The weight-free VAE round trip. No trained transcoder exists for this space:
        # the Sesqui ``flux2`` weights are trained on Flux.2's BatchNorm-normalised
        # packed latent and this latent is the raw anchor space — same shape,
        # different statistics.
        return VAEPixelUpscaler(self.vae, scale=self.UPSCALE_SCALE)


__all__ = ["MageFlowModel"]
