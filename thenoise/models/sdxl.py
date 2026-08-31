"""SDXL adapter (vanilla SDXL and SDXL fine-tunes).

Stable Diffusion XL (and its fine-tune variants such as Illustrious-XL) is an
SDXL LDM UNet (transformer-depth ``[0,0,2,2,10,10]``), the SDXL dual-CLIP text
encoders (CLIP-L + CLIP-G -> 2048-dim cross-attention, CLIP-G pooled for the
ADM vector), and the SDXL VAE (4 channels, 8x compression). It is a discrete
model sampled with the shared ``euler`` sampler over a discrete DDIM-style
sigma grid.

The architecture is identical across all SDXL checkpoints, so any SDXL model
(vanilla ``stabilityai/stable-diffusion-xl-base-1.0``, anime fine-tunes like
Illustrious-XL / Juggernaut, etc.) loads with the same adapter. The prediction
type (epsilon vs v-prediction) is autodetected from marker tensors in the
checkpoint, exactly like ComfyUI's ``SDXL.model_type``; continuous-EDM (Playground
V2.5) and zsnr variants are detected but rejected until supported.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional

import torch

from thenoise.dit.sdxl.models import timestep_embedding
from thenoise.dit.sdxl.sampling import (
    discrete_timesteps,
    get_alphas_cumprod,
    get_sigmas,
    rescale_zero_terminal_snr_alphas_cumprod,
    sigmas_for_timesteps,
)
from thenoise.dit.sdxl.text import OpenClipTextTransformer
from thenoise.dit.sdxl.utils import (
    find_sdxl_tokenizer_dir,
    load_sdxl_dit,
    load_sdxl_text_encoders,
    load_sdxl_tokenizer,
)
from thenoise.dit.sdxl.lora import convert_hyper_sd_lora, lora_uses_diffusers_unet_keys
from thenoise.dit.sdxl.vae import AutoencoderKLSdxl, load_sdxl_vae
from thenoise.models.base import Conditioning, DiffusionModel, Step, normalize_keys
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.samplers.euler import EulerSampler
from thenoise.utils.math import round_up

logger = logging.getLogger(__name__)

#: Pooled CLIP-G projection width (1280) and size-embedding width (6 x 256).
POOLED_DIM = 1280
SIZE_EMBED_DIM = 1536


class SdxlModel(DiffusionModel):
    name = "sdxl"

    # SDXL defaults: ~28 euler steps, CFG ~5.5, 1024x1024. This matches the
    # widely-recommended settings for vanilla SDXL and anime fine-tunes; higher
    # steps / CFG tend to over-saturate and hurt prompt adherence rather than help.
    DEFAULT_STEPS = 28
    DEFAULT_GUIDANCE_SCALE = 5.5
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    # The shared euler sampler reproduces the discrete DDIM/euler update when
    # ``denoise_step`` returns the predicted noise and the schedule's deltas are
    # sigma differences.
    SAMPLER = "euler"
    # SDXL is a discrete model; only the euler solver is valid. A requested
    # ``er_sde`` (a stochastic flow-oriented solver) produces poor output on
    # SDXL, so it falls back to euler with a warning (see ``create_sampler``).
    SUPPORTED_SAMPLERS = ["euler"]

    #: Prediction types detected from the checkpoint's marker keys (ComfyUI's
    #: ``SDXL.model_type``). Epsilon and v-prediction share the discrete 1000-step
    #: sigma grid; they differ only in how the UNet output maps to a velocity.
    PREDICTION_EPSILON = "epsilon"
    PREDICTION_V_PREDICTION = "v_prediction"

    #: Prediction-type marker tensors carried by some SDXL checkpoints. ``v_pred``
    #: marks a v-prediction model; the rest flag continuous-EDM / zsnr variants
    #: that thenoise does not (yet) support.
    _PREDICTION_MARKERS = (
        "v_pred",
        "ztsnr",
        "edm_mean",
        "edm_std",
        "edm_vpred.sigma_max",
        "edm_vpred.sigma_min",
    )

    # 4-channel SDXL latent, 8x spatial compression.
    LATENT_CHANNELS = 4
    _VAE_SCALE = 8

    MAX_SEQUENCE_LENGTH = 77

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is an SDXL LDM UNet.

        The classic CompVis ``UNetModel`` layout is uniquely identified by the
        ``input_blocks`` / ``middle_block`` block lists together with the
        ``label_emb`` and ``time_embed`` conditioning MLPs. Keys are normalized
        first so repackaged checkpoints (``model.diffusion_model.``) resolve
        identically. Other registered models (flow DiTs) share none of these.
        """
        keys = list(normalize_keys(f.keys()))
        has_input = any(k.startswith("input_blocks.") for k in keys)
        has_middle = any(k.startswith("middle_block.") for k in keys)
        has_label = any(k.startswith("label_emb.") for k in keys)
        has_time = any(k.startswith("time_embed.") for k in keys)
        return has_input and has_middle and has_label and has_time

    @staticmethod
    def _read_dit_keys(dit_path: str) -> list[str]:
        """Read a dit file's tensor keys (header only, no tensor data)."""
        from thenoise.utils.safetensors import MemoryEfficientSafeOpen

        with MemoryEfficientSafeOpen(dit_path) as f:
            return list(f.keys())

    @staticmethod
    def prediction_type_from_keys(keys) -> str:
        """Prediction type from a checkpoint's tensor keys.

        Mirrors ComfyUI's ``SDXL.model_type``, which reads marker tensors from the
        state dict rather than the (identical) UNet architecture:
        ``edm_mean``/``edm_std`` -> EDM, ``edm_vpred.*`` -> continuous
        V_PREDICTION_EDM, ``v_pred`` -> v-prediction, otherwise epsilon. The
        ``ztsnr`` marker selects a zero-terminal-SNR schedule (see
        :meth:`zsnr_from_keys`) on top of whatever prediction type; it is no
        longer rejected. Continuous-EDM variants are detected but not yet
        supported, so they raise a clear error instead of rendering wrong.
        """
        keys = set(normalize_keys(keys))
        if "edm_mean" in keys and "edm_std" in keys:
            raise NotImplementedError(
                "SDXL checkpoint uses EDM prediction (Playground V2.5), which "
                "thenoise does not support yet"
            )
        if "edm_vpred.sigma_max" in keys:
            raise NotImplementedError(
                "SDXL checkpoint uses continuous V_PREDICTION_EDM, which thenoise "
                "does not support yet"
            )
        if "v_pred" in keys:
            return SdxlModel.PREDICTION_V_PREDICTION
        return SdxlModel.PREDICTION_EPSILON

    @staticmethod
    def zsnr_from_keys(keys) -> bool:
        """True if the checkpoint carries the ``ztsnr`` zero-terminal-SNR marker."""
        return "ztsnr" in set(normalize_keys(keys))

    @staticmethod
    def _resolve_zsnr(keys, override) -> bool:
        """Resolve the effective zsnr flag from marker auto-detection + CLI override.

        ``override`` is the tri-state ``sd_zsnr``: ``None`` (auto-detect from the
        ``ztsnr`` marker), ``True`` (force on), or ``False`` (force off — lets a
        marker-bearing checkpoint be sampled on the plain linear schedule for
        debugging, e.g. NoobAI's misleading ``ztsnr`` marker).

        When zsnr is auto-detected from the marker (no override), log the result
        and how to override it, since some checkpoints carry a misleading
        ``ztsnr`` marker (NoobAI vPred samples better on the plain schedule).
        """
        auto = SdxlModel.zsnr_from_keys(keys)
        if override is None:
            if auto:
                logger.info(
                    "This checkpoint has the zero-terminal-SNR (zsnr) marker, so "
                    "the zsnr schedule is ON. If the generated image comes out "
                    "garbled (for example a solid purple color), the marker may "
                    "be misleading; retry with --no-sd-zsnr to turn zsnr off "
                    "and use the normal schedule."
                )
            else:
                logger.info(
                    "This checkpoint does not carry the zero-terminal-SNR (zsnr) "
                    "marker, so the normal schedule is used. If the generated "
                    "image comes out garbled and you know this model was trained "
                    "with zsnr, retry with --sd-zsnr to turn it on."
                )
            return auto
        return override

    def __init__(
        self,
        *,
        config: ModelConfig,
    ):
        super().__init__(config=config)

        # Enable TF32 for float32 matmuls (the CLIP text towers / weight loading
        # run some fp32 ops); suppresses the TensorFloat32 UserWarning and is
        # faster with negligible precision loss for inference.
        torch.set_float32_matmul_precision("high")

        if config.checkpoint_path:
            # Single combined checkpoint: partition it in memory and load all
            # three components from the one file (no splitting needed).
            from thenoise.dit.sdxl.checkpoint import SDXLCheckpoint

            logger.info("Loading combined SDXL checkpoint %s", config.checkpoint_path)
            ckpt = SDXLCheckpoint(config.checkpoint_path, device=config.device, dtype=config.dtype)
            self.dit, self.clip_l, self.clip_g, self.vae = ckpt.load_components()
            # Prediction type / zsnr are autodetected from the checkpoint's marker
            # keys; a ``--sd-zsnr`` flag overrides/forces zsnr for models whose
            # marker was stripped.
            self.prediction_type = self.prediction_type_from_keys(ckpt.keys)
            self.zsnr = self._resolve_zsnr(ckpt.keys, config.sd_zsnr)
            self.tokenizer = load_sdxl_tokenizer()  # vendored config
        else:
            logger.info("Loading SDXL UNet from %s", config.dit_path)
            self.dit = load_sdxl_dit(config.dit_path, device=config.device, dtype=config.dtype)
            # Prediction type (epsilon / v-prediction) is autodetected from marker
            # tensors in the checkpoint, not the UNet architecture. The zsnr
            # schedule is likewise autodetected from the ``ztsnr`` marker or forced
            # by the ``--sd-zsnr`` flag.
            keys = self._read_dit_keys(config.dit_path)
            self.prediction_type = self.prediction_type_from_keys(keys)
            self.zsnr = self._resolve_zsnr(keys, config.sd_zsnr)

            logger.info(
                "Loading SDXL text encoders (CLIP-L + CLIP-G) from %s",
                config.text_encoder_path,
            )
            self.clip_l, self.clip_g = load_sdxl_text_encoders(
                config.text_encoder_path, device=config.device, dtype=config.dtype
            )
            self.tokenizer = load_sdxl_tokenizer(
                find_sdxl_tokenizer_dir(config.text_encoder_path)
            )

            logger.info("Loading SDXL VAE from %s", config.vae_path)
            self.vae = load_sdxl_vae(self.vae_path, device=self.device, disable_mmap=True)

        self.dit.eval().requires_grad_(False)
        self.clip_l.eval().requires_grad_(False)
        self.clip_g.eval().requires_grad_(False)
        self.vae.to(self.dtype).eval().requires_grad_(False)
        logger.info("SDXL prediction type: %s", self.prediction_type)

        # Discrete alphas_cumprod grid on-device, for per-step sigma lookup.
        # ComfyUI's EPS ``calculate_input`` scales the UNet input by
        # ``1/sqrt(sigma^2 + 1)`` (see ``BaseModel._apply_model``); without it
        # the noisiest input is ~sigma_max (~26) too large and denoise collapses.
        # zsnr checkpoints (or ``--sd-zsnr``) use a zero-terminal-SNR-rescaled grid.
        self._alphas_cumprod = get_alphas_cumprod(device=self.device)
        if self.zsnr:
            self._alphas_cumprod = rescale_zero_terminal_snr_alphas_cumprod(
                self._alphas_cumprod
            )

        # Per-step cached ADM vector (pooled text + size embedding), built in
        # ``prepare_latent`` from the request's ``Conditioning``.
        self._y = None
        self._y_uncond = None

        logger.info("SDXL model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(
        self,
        args: EncodePromptArgs,
    ) -> Conditioning:
        context, pooled = self._encode_prompt(args.prompt)
        null = None
        neg_pooled = None
        if args.guidance_scale > 1.0:
            neg_context, neg_pooled = self._encode_prompt(args.negative_prompt)
            null = neg_context
        return Conditioning(cond=context, null=null, pooled=pooled, neg_pooled=neg_pooled)

    def _get_lora_sd(self, filename: str) -> dict[str, torch.Tensor]:
        """Load a LoRA, converting Hyper-SD diffusers-keyed UNet LoRAs to LDM keys.

        Hyper-SD step-reduction LoRAs target the diffusers SDXL UNet layout
        (``lora_unet_down_blocks_*``); our UNet uses LDM keys. Auto-convert so
        ``--lora Hyper-SDXL-8steps-CFG-lora.safetensors --steps 8`` just works.
        """
        sd = super()._get_lora_sd(filename)
        if lora_uses_diffusers_unet_keys(sd):
            logger.info(
                "Converting diffusers-keyed UNet LoRA (%s) to LDM key naming", filename
            )
            sd = convert_hyper_sd_lora(sd)
        return sd

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cross_attn_context [1,77,2048], pooled [1,1280])``."""
        dev = torch.device(self.device)
        # ComfyUI tokenizes CLIP-L and CLIP-G with *different* padding: CLIP-L
        # pads with EOS (49407) but CLIP-G pads with 0. Sharing one padded
        # sequence (EOS) for both made the CLIP-G cross-attention context attend
        # to ~66 spurious EOS tokens, corrupting prompt adherence (incoherent
        # subjects / generic anime faces).
        raw = self.tokenizer(prompt, truncation=True, max_length=self.MAX_SEQUENCE_LENGTH)
        base = raw["input_ids"][: self.MAX_SEQUENCE_LENGTH]
        pad = self.MAX_SEQUENCE_LENGTH - len(base)
        ids_l = torch.tensor([base + [49407] * pad], device=dev)  # CLIP-L pad = EOS
        ids_g = torch.tensor([base + [0] * pad], device=dev)      # CLIP-G pad = 0

        with torch.no_grad():
            out_l = self.clip_l(ids_l, output_hidden_states=True)
            hidden_l = out_l.hidden_states[-2]  # [1, 77, 768] penultimate
            hidden_g, pooled = self.clip_g(ids_g)  # [1, 77, 1280], [1, 1280]
            context = torch.cat([hidden_l, hidden_g], dim=-1)  # [1, 77, 2048]
        # The CLIP-G pooled text vector is passed to the UNet's ``label_emb``
        # unnormalized (``text_projection(eos @ ln_final)``), matching ComfyUI /
        # diffusers SDXL; the Linear label_emb was trained on that raw scale.
        return context.to(self.dtype), pooled.to(self.dtype)

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        shape = (
            1,
            self.dit.in_channels,
            params.height // self._VAE_SCALE,
            params.width // self._VAE_SCALE,
        )
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def _size_embedding(self, height: int, width: int) -> torch.Tensor:
        """ComfyUI SDXL size embedding: 6 timestep_embeddings of 256, concatenated.

        Order: [height, width, crop_h, crop_w, target_height, target_width],
        with crop = (0, 0) and target = (height, width). Returns ``[1, 1536]``.
        """
        dev = torch.device(self.device)
        parts = []
        for value in (height, width, 0, 0, height, width):
            t = torch.tensor([float(value)], device=dev)
            parts.append(timestep_embedding(t, 256).to(torch.float32))
        return torch.cat(parts, dim=0).flatten().unsqueeze(0).to(dev)

    def _set_adm(self, cond: Conditioning, height: int, width: int) -> None:
        """Build the cached per-step ADM vector (pooled text + size embedding)."""
        size_embeds = self._size_embedding(height, width)
        self._y = torch.cat([cond.pooled, size_embeds], dim=-1).to(self.dtype)
        if cond.null is not None and cond.neg_pooled is not None:
            self._y_uncond = torch.cat([cond.neg_pooled, size_embeds], dim=-1).to(self.dtype)
        else:
            self._y_uncond = None

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
    ) -> torch.Tensor:
        # Scale the initial noise by the scheduler's max sigma (ComfyUI's flow
        # euler for the discrete SDXL EPS model: init = noise * sigma_max).
        sigmas = get_sigmas(params.steps, self._alphas_cumprod)
        scaled = latents * sigmas[0]

        self._set_adm(cond, params.height, params.width)
        return scaled

    def refine_latents(
        self,
        z: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
    ) -> torch.Tensor:
        """EDM refine for SDXL: add noise at the tail-schedule sigma, then euler.

        Unlike flow models (CONST blend), SDXL adds ``sigma * noise`` and must NOT
        re-scale a clean latent by sigma_max (``prepare_latent`` would blow it up
        by ~26.9). We set the ADM vector directly and run the tail steps.
        """
        refine_steps = self.REFINE_STEPS
        denoise = self.REFINE_DENOISE
        new_steps = int(refine_steps / denoise)

        refine_params = replace(params, steps=new_steps)
        full = self.schedule(refine_params)
        sub = full[-refine_steps:]
        sigma_r = self._sigma_at(sub[0].t).to(z.dtype)

        generator = torch.Generator(device=self.device).manual_seed(params.seed)
        noise = torch.randn_like(z, generator=generator)
        noised = z + sigma_r * noise

        self._set_adm(cond, refine_params.height, refine_params.width)
        x = noised
        solver = EulerSampler(self)
        x = solver.sample(
            x, sub, cond, params.guidance_scale, params.seed, desc="refining"
        )
        return self.finalize_latent(x, refine_params)

    def schedule(self, params: SamplingParams) -> list[Step]:
        dev = torch.device(self.device)
        steps = params.steps
        ts = discrete_timesteps(steps)  # noise -> clean
        sigmas = sigmas_for_timesteps(
            ts, self._alphas_cumprod
        )  # noise -> clean, trailing 0
        return [
            Step(
                t=torch.tensor(float(ts[i]), device=dev, dtype=torch.float32),
                delta=torch.tensor(sigmas[i] - sigmas[i + 1], device=dev, dtype=torch.float32),
            )
            for i in range(steps)
        ]

    def _sigma_at(self, t: torch.Tensor) -> torch.Tensor:
        """Sigma for discrete timestep index ``t`` (matches the sampling grid)."""
        abar = self._alphas_cumprod[t.to(torch.long)]
        return torch.sqrt((1.0 - abar) / abar)

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        dev = torch.device(self.device)
        t_full = t.to(dev).reshape(1)
        context = cond.cond.to(dev, dtype=self.dtype)
        # ComfyUI EPS ``calculate_input``: the UNet expects the latent scaled by
        # ``1/sqrt(sigma^2 + 1)`` (the DDPM-space latent x_t), not the raw
        # EDM-space latent. Apply it before every (un)conditional forward.
        sigma = self._sigma_at(t_full).to(latents.dtype)
        scaled = latents / torch.sqrt(sigma**2 + 1)
        with torch.no_grad():
            out = self.dit(scaled, t_full, self._y, context)
            if guidance_scale > 1.0 and self._y_uncond is not None and cond.null is not None:
                uncond = self.dit(
                    scaled, t_full, self._y_uncond, cond.null.to(dev, dtype=self.dtype)
                )
                out = uncond + guidance_scale * (out - uncond)
        # Map the UNet output to the euler velocity ``x -= delta * velocity``.
        if self.prediction_type == self.PREDICTION_V_PREDICTION:
            # ComfyUI ``V_PREDICTION``: denoised = x/(sigma^2+1) - v*sigma/sqrt(sigma^2+1),
            # so velocity = (x - denoised)/sigma = x*sigma/(sigma^2+1) + v/sqrt(sigma^2+1).
            return latents * sigma / (sigma**2 + 1) + out / torch.sqrt(sigma**2 + 1)
        # ComfyUI ``EPS``: the model output (eps) IS the velocity.
        return out

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # The denoised latent is already in the UNet's scaled space; the VAE's
        # ``decode_to_pixels`` applies the 1/scaling_factor before decoding.
        return latents

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # SDXL latents are 8x compressed; round up to a multiple of 8.
        size = round_up(width, 8), round_up(height, 8)
        # SDXL is trained near 1024x1024; strongly off-native sizes (e.g. 512)
        # produce incoherent output (observed: a red panda becomes a garden).
        if min(width, height) < 768 or max(width, height) > 1536:
            logger.warning(
                "SDXL is trained near 1024x1024; requested %sx%s is far from "
                "native and may produce garbled results. Recommended ~1024x1024.",
                width,
                height,
            )
        return size

    def _upscale_format(self) -> str:
        # The SesquiSDXL latent upscaler operates on 4-channel latents; the SDXL
        # pipeline latent is ``raw * 0.13025``, which ``make_sdxl``'s affine
        # adaptor converts to/from raw VAE space.
        return "sdxl"


__all__ = ["SdxlModel"]
