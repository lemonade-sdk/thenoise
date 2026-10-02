"""Qwen-Image adapter — dual-stream DiT + Qwen2.5-VL-7B text encoder + Qwen-Image VAE.

All variants are edit-capable: the input image is both encoded by Qwen2.5-VL as vision
tokens into the text conditioning, and VAE-encoded into a reference latent
concatenated into the DiT token sequence. Conditioning those tokens at timestep zero
(the ``index_timestep_zero`` reference method) is a per-run choice resolved from the
``ref_method`` preference, whose automatic layer comes from the checkpoint's
``__index_timestep_zero__`` marker.

That marker is also what makes the reference K/V step-invariant, so the same run can
freeze them: ``prepare_latent`` starts the caches and ``denoise_step`` drives the
shared ``thenoise.dit.kvcache`` protocol.
"""
from __future__ import annotations

import logging

import torch

from thenoise.dit.mage_flow.keys import MAGE_LAYERS, dit_block_count, is_qwen_image_family
from thenoise.dit.qwen_image import models as qwen_models
from thenoise.dit.qwen_image import sampling as qwen_sampling
from thenoise.dit.qwen_image import utils as qwen_utils
from thenoise.dit.qwen_image.models import build_txt_positions, build_video_positions
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    normalize_keys,
)
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, SesquiLSRUpscaler
from thenoise.utils.latents import pack_latents, unpack_latents
from thenoise.utils.math import round_up
from thenoise.utils.text_encoder import (
    QWEN25_TOKENIZER_CONFIG_DIR,
    load_qwen2_5_vl_model,
    load_qwen2_5_vl_processor,
    load_qwen2_tokenizer,
)
from thenoise.vae import load_qwen_vae

logger = logging.getLogger(__name__)


class QwenImageModel(DiffusionModel):
    name = "qwen_image"

    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 28,
        "guidance_scale": 2.5,
        "sampler": "euler",
    }

    # Instruction-based editing off the Qwen-Image-Edit recipe. The KV cache is valid
    # only with ``ref_method="index_timestep_zero"``.
    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": True}

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a Qwen-Image DiT: the family block layout, deep.

        Mage-Flow shares the tensor names, so depth is the separator: 60 blocks here.
        Each detector states its own depth condition, so neither depends on catalog
        order.
        """
        keys = list(normalize_keys(f.keys()))
        return is_qwen_image_family(keys) and dit_block_count(keys) > MAGE_LAYERS

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        logger.info("Loading Qwen-Image DiT from %s", config.dit_path)
        self.dit = qwen_models.load_qwen_image_dit(
            config.dit_path, device=self.offload_device, dtype=config.dtype
        )

        tokenizer_dir = QWEN25_TOKENIZER_CONFIG_DIR
        logger.info("Loading Qwen2.5-VL text encoder from %s", config.text_encoder_path)
        self.text_encoder = load_qwen2_5_vl_model(
            config.text_encoder_path, dtype=config.dtype, device=self.offload_device
        )
        self.tokenizer = load_qwen2_tokenizer(tokenizer_dir)
        self.vl_processor = load_qwen2_5_vl_processor(self.tokenizer)

        self.vae = load_qwen_vae(self.vae_path, device=self.device)

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
        ref=None,
        ref_method: str = "index",
    ) -> torch.Tensor:
        """Pack the canonical latent into DiT tokens and stash conditioning once.

        The reference latent (edit) is packed and concatenated into the DiT token
        sequence; ``img_shapes`` gains one entry per reference and drives both the
        precomputed RoPE positions and the timestep-zero split index.
        """
        dev = torch.device(self.device)
        x = pack_latents(latents.to(device=dev, dtype=self.dtype))

        self._txt = cond.cond.to(device=dev, dtype=self.dtype)
        txt_len = int(cond.cond_mask.to(device=dev).sum().item())
        self._img_shapes = [(1, params.height // self._pixels_per_token, params.width // self._pixels_per_token)]

        if ref is not None:
            ref_tokens = []
            for ref_latent in ref:
                ref_tokens.append(pack_latents(ref_latent.to(device=dev, dtype=self.dtype)))
                self._img_shapes.append(
                    (1, ref_latent.shape[-2] // 2, ref_latent.shape[-1] // 2)
                )
            self._ref_tokens = torch.cat(ref_tokens, dim=1)
        else:
            self._ref_tokens = None

        null_len = None
        if cond.null is not None:
            self._null_txt = cond.null.to(device=dev, dtype=self.dtype)
            null_len = int(cond.null_mask.to(device=dev).sum().item())
        else:
            self._null_txt = None

        # RoPE is independent of image/timestep: build it once per run. The image
        # stream covers the concatenated base+ref tokens; target and reference
        # positions are stored apart so ``denoise_step`` can drop the references once
        # the KV cache has frozen their K/V.
        self.dit.pe_embedder.clear()
        img_pos = build_video_positions(self._img_shapes, device=dev)
        num_img_tokens = (
            self._img_shapes[0][0] * self._img_shapes[0][1] * self._img_shapes[0][2]
        )
        self.dit.pe_embedder.store("img", img_pos[:, :num_img_tokens], dtype=self.dtype)
        self.dit.pe_embedder.store("ref", img_pos[:, num_img_tokens:], dtype=self.dtype)
        max_vid_index = max(max(h // 2, w // 2) for _, h, w in self._img_shapes)
        txt_pos = build_txt_positions(max_vid_index, txt_len, device=dev)
        self.dit.pe_embedder.store("txt", txt_pos, dtype=self.dtype)
        if null_len is not None:
            null_pos = build_txt_positions(max_vid_index, null_len, device=dev)
            self.dit.pe_embedder.store("txt_uncond", null_pos, dtype=self.dtype)

        # Split point of the timestep-zero reference tokens: the base image token
        # count, or None for a single-row modulation.
        zero_cond_t = ref_method == "index_timestep_zero" and self._ref_tokens is not None
        self._timestep_zero_index = num_img_tokens if zero_cond_t else None

        self.start_kv_caches(
            params,
            has_reference=self._ref_tokens is not None,
            has_uncond=params.guidance_scale > 1.0 and self._null_txt is not None,
        )

        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        # The dynamic shift is driven by the packed token count (H/16 * W/16).
        image_seq_len = (params.height // self._pixels_per_token) * (
            params.width // self._pixels_per_token
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
        """One Qwen-Image DiT forward (+ CFG), returning the velocity."""
        dev = torch.device(self.device)
        t_full = torch.full((1,), float(t), dtype=latents.dtype, device=dev)
        pe_img = self.dit.pe_embedder["img"]
        pe_ref = self.dit.pe_embedder["ref"]

        with torch.autocast(device_type=dev.type, dtype=self.dtype):
            pos = self._dit_forward(
                latents, t_full, self._txt, self.dit.pe_embedder["txt"],
                self.kv_cache("cond"), pe_img, pe_ref,
            )
            if guidance_scale > 1.0 and self._null_txt is not None:
                neg = self._dit_forward(
                    latents, t_full, self._null_txt, self.dit.pe_embedder["txt_uncond"],
                    self.kv_cache("uncond"), pe_img, pe_ref,
                )
                v = neg + guidance_scale * (pos - neg)
                # Renormalize the CFG combination back to the conditional norm.
                cond_norm = torch.norm(pos, dim=-1, keepdim=True)
                noise_norm = torch.norm(v, dim=-1, keepdim=True)
                v = v * (cond_norm / noise_norm)
            else:
                v = pos
        return v

    def _dit_forward(
        self,
        x: torch.Tensor,
        t_full: torch.Tensor,
        ctx: torch.Tensor,
        pe_ctx: torch.Tensor,
        kv,
        pe_img: torch.Tensor,
        pe_ref: torch.Tensor,
    ) -> torch.Tensor:
        """One DiT forward with the KV cache's fill/read semantics.

        While the cache is empty the reference tokens are passed in (fill); once
        ``kv.filled`` they are dropped and each block works from its cache (read).
        """
        ref_tokens = self._ref_tokens
        if kv is not None and kv.filled:
            ref_tokens = None
        return self.dit(
            hidden_states=x,
            encoder_hidden_states=ctx,
            timestep=t_full,
            img_pe=pe_img,
            txt_pe=pe_ctx,
            ref_tokens=ref_tokens,
            ref_pe=pe_ref,
            kv=kv,
            timestep_zero_index=self._timestep_zero_index,
        )

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        self.end_kv_caches()
        return unpack_latents(
            latents, params.height // self.vae.spatial_compression, params.width // self.vae.spatial_compression
        )

    @property
    def _pixels_per_token(self) -> int:
        return self.vae.spatial_compression * self.dit.patch_size

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # Pixel dims must be a multiple of the VAE compression * DiT patch size.
        align = self._pixels_per_token
        return round_up(width, align), round_up(height, align)

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) -> canonical reference latent."""
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(self, latents: torch.Tensor, method: str = "index", ref_index: int = 1):
        """Canonical reference latent -> packed DiT tokens."""
        if method not in ("index", "index_timestep_zero"):
            raise ValueError(
                f"unsupported ref_latents_method {method!r}; expected 'index' or 'index_timestep_zero'"
            )
        dev = torch.device(self.device)
        return pack_latents(latents.to(device=dev, dtype=self.dtype)), None

    def _create_upscaler(self) -> LatentUpscaler:
        return SesquiLSRUpscaler("wan21", device=self.device, dtype=self.dtype)


__all__ = ["QwenImageModel"]
