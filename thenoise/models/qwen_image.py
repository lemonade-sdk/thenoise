"""Qwen-Image adapter — dual-stream DiT + Qwen2.5-VL-7B text encoder + Qwen-Image VAE.

All variants are edit-capable: the input image is both (1) encoded by Qwen2.5-VL as
vision tokens into the text conditioning, and (2) VAE-encoded and concatenated into
the DiT token sequence as a reference latent. ``zero_cond_t`` (edit-2511, flagged by
``__index_timestep_zero__``) is detected from the checkpoint.
"""
from __future__ import annotations

import logging

import torch
from transformers import Qwen2VLProcessor

from thenoise.dit.qwen_image import models as qwen_models
from thenoise.dit.qwen_image import sampling as qwen_sampling
from thenoise.dit.qwen_image import utils as qwen_utils
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    normalize_keys,
)
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.utils.math import round_up
from thenoise.vae import load_qwen_vae

logger = logging.getLogger(__name__)


def _detect_zero_cond_t(dit_path: str) -> bool:
    from safetensors import safe_open

    with safe_open(dit_path, framework="pt") as f:
        return "__index_timestep_zero__" in f.keys()


class QwenImageModel(DiffusionModel):
    name = "qwen_image"

    DEFAULT_STEPS = 28
    DEFAULT_GUIDANCE_SCALE = 2.5
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    SAMPLER = "euler"

    # Qwen-Image VAE: 16 channels at 8x spatial compression, patchified 2x2.
    LATENT_CHANNELS = 16
    _VAE_SCALE = 8

    supports_edit = True

    # Qwen-Image uses separate ``to_q``/``to_k``/``to_v`` attention projections,
    # so LoRA factors must NOT be fused into a single ``qkv``.
    fused_attention = False

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a Qwen-Image DiT.

        Qwen-Image's distinctive blocks are the dual-stream projections (``img_in.`` /
        ``txt_in.``) and the joint time/text embedding (``time_text_embed.``). Keys
        are normalized first so repackaged checkpoints resolve identically.
        """
        keys = list(normalize_keys(f.keys()))
        has_img_in = any(k.startswith("img_in.") for k in keys)
        has_txt_in = any(k.startswith("txt_in.") for k in keys)
        has_time_text_embed = any(k.startswith("time_text_embed.") for k in keys)
        return has_img_in and has_txt_in and has_time_text_embed

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.zero_cond_t = _detect_zero_cond_t(config.dit_path)
        logger.info("Loading Qwen-Image DiT from %s (zero_cond_t=%s)", config.dit_path, self.zero_cond_t)
        self.dit = qwen_models.load_qwen_image_dit(
            config.dit_path, device=self.offload_device, zero_cond_t=self.zero_cond_t, dtype=config.dtype
        )
        self.dit.eval().requires_grad_(False)

        tokenizer_dir = qwen_utils.TOKENIZER_CONFIG_DIR
        logger.info("Loading Qwen2.5-VL text encoder from %s", config.text_encoder_path)
        self.text_encoder = qwen_utils.load_qwen2_5_vl(
            config.text_encoder_path, dtype=config.dtype, device=self.offload_device
        )
        self.text_encoder.eval().requires_grad_(False)
        self.tokenizer = qwen_utils.load_qwen2_tokenizer(tokenizer_dir)
        self.vl_processor = Qwen2VLProcessor.from_pretrained(tokenizer_dir, local_files_only=True)

        self.vae = load_qwen_vae(self.vae_path, device=self.device, disable_mmap=True)
        self.vae.eval().requires_grad_(False)

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("Qwen-Image model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        if args.image is not None:
            cond, cond_mask = qwen_utils.get_qwen_prompt_embeds_with_image(
                self.vl_processor, self.text_encoder, args.prompt, args.image
            )
        else:
            cond, cond_mask = qwen_utils.get_qwen_prompt_embeds(self.tokenizer, self.text_encoder, args.prompt)

        null = None
        null_mask = None
        if args.guidance_scale > 1.0:
            if args.image is not None:
                null, null_mask = qwen_utils.get_qwen_prompt_embeds_with_image(
                    self.vl_processor, self.text_encoder, args.negative_prompt, args.image
                )
            else:
                null, null_mask = qwen_utils.get_qwen_prompt_embeds(
                    self.tokenizer, self.text_encoder, args.negative_prompt
                )
        return Conditioning(cond=cond, cond_mask=cond_mask, null=null, null_mask=null_mask)

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        shape = (1, self.LATENT_CHANNELS, params.height // self._VAE_SCALE, params.width // self._VAE_SCALE)
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
        ref=None,
        ref_method: str = "index",
    ) -> torch.Tensor:
        """Pack the canonical latent into DiT tokens and stash conditioning once.

        The reference latent (edit) is packed and concatenated into the DiT token
        sequence; the DiT's ``img_shapes`` gains one entry per reference.
        """
        dev = torch.device(self.device)
        x = qwen_utils.pack_latents(latents.to(device=dev, dtype=self.dtype))

        self._txt = cond.cond.to(device=dev, dtype=self.dtype)
        self._txt_mask = cond.cond_mask.to(device=dev, dtype=torch.long)
        self._txt_seq_lens = [int(m.sum().item()) for m in self._txt_mask]
        self._img_shapes = [(1, params.height // self._VAE_SCALE // 2, params.width // self._VAE_SCALE // 2)]

        if ref is not None:
            ref_tokens = []
            for ref_latent in ref:
                ref_tokens.append(qwen_utils.pack_latents(ref_latent.to(device=dev, dtype=self.dtype)))
                self._img_shapes.append(
                    (1, ref_latent.shape[-2] // 2, ref_latent.shape[-1] // 2)
                )
            self._ref_tokens = torch.cat(ref_tokens, dim=1)
        else:
            self._ref_tokens = None

        if cond.null is not None:
            self._null_txt = cond.null.to(device=dev, dtype=self.dtype)
            self._null_mask = cond.null_mask.to(device=dev, dtype=torch.long)
            self._null_seq_lens = [int(m.sum().item()) for m in self._null_mask]
        else:
            self._null_txt = self._null_mask = self._null_seq_lens = None

        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        # ``mu`` (the dynamic shift) is computed from the *packed* latent token
        # count (H/16 * W/16), matching musubi-tuner's ``image_seq_len =
        # latents.shape[1]`` after ``pack_latents``. Using the raw 8x-compressed
        # grid (H/8 * W/8) inflates mu by 4x and denoises at the wrong timesteps.
        image_seq_len = (params.height // self._VAE_SCALE // 2) * (
            params.width // self._VAE_SCALE // 2
        )
        ts = qwen_sampling.get_schedule(params.steps, image_seq_len)
        return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(params.steps)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        dev = torch.device(self.device)
        t_full = torch.full((1,), float(t), dtype=latents.dtype, device=dev)
        hidden_states = latents
        if self._ref_tokens is not None:
            hidden_states = torch.cat([hidden_states, self._ref_tokens], dim=1)

        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=self.dtype):
            pos = self.dit(
                hidden_states, self._txt, t_full, self._img_shapes, self._txt_seq_lens
            )
            pos = pos[:, : latents.shape[1], :]
            if guidance_scale > 1.0 and self._null_txt is not None:
                neg = self.dit(
                    hidden_states, self._null_txt, t_full, self._img_shapes, self._null_seq_lens
                )
                neg = neg[:, : latents.shape[1], :]
                v = neg + guidance_scale * (pos - neg)
                # Re-normalize the CFG combination back to the conditional norm
                # (Qwen-Image guidance trick), preventing the extrapolated
                # prediction from blowing up / collapsing into a pattern.
                cond_norm = torch.norm(pos, dim=-1, keepdim=True)
                noise_norm = torch.norm(v, dim=-1, keepdim=True)
                v = v * (cond_norm / noise_norm)
            else:
                v = pos
        return v

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # Unpack the DiT tokens back to the canonical 4D latent.
        return qwen_utils.unpack_latents(
            latents, params.height // self._VAE_SCALE, params.width // self._VAE_SCALE
        )

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # The latent grid is patchified in 2x2 blocks on an 8x-VAE-compressed latent.
        align = self._VAE_SCALE * 2
        return round_up(width, align), round_up(height, align)

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) -> canonical reference latent."""
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(self, latents: torch.Tensor, method: str = "index", ref_index: int = 1):
        """Canonical reference latent -> packed DiT tokens (native Qwen-Image approach)."""
        if method != "index":
            raise ValueError(f"unsupported ref_latents_method {method!r}; only 'index' is supported")
        dev = torch.device(self.device)
        return qwen_utils.pack_latents(latents.to(device=dev, dtype=self.dtype)), None

    def _upscale_format(self) -> str:
        """Qwen-Image VAE -> Wan21 z-score latent format."""
        return "wan21"


__all__ = ["QwenImageModel"]
