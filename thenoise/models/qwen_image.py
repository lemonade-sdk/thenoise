"""Qwen-Image adapter — dual-stream DiT + Qwen2.5-VL-7B text encoder + Qwen-Image VAE.

All variants are edit-capable: the input image is both (1) encoded by Qwen2.5-VL as
vision tokens into the text conditioning, and (2) VAE-encoded and concatenated into
the DiT token sequence as a reference latent. Conditioning the reference tokens at
timestep zero (the ``index_timestep_zero`` reference method) is a per-run choice
resolved from the ``ref_method`` preference, whose automatic layer comes from the
checkpoint's ``__index_timestep_zero__`` marker.

That marker is also what makes the reference-latent KV cache (ComfyUI's
``FluxKVCache``) valid, so this adapter drives the shared ``thenoise.dit.kvcache``
protocol exactly like the Flux Klein one: ``prepare_latent`` starts the run's
caches and keeps the reference RoPE positions apart from the target's,
``denoise_step`` feeds the reference tokens only while the cache is still filling.
"""
from __future__ import annotations

import logging

import torch

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

    # Instruction-based editing off the Qwen-Image-Edit recipe: the input image is
    # concatenated into the token sequence as a reference latent. The KV cache
    # freezes the reference K/V across steps (ComfyUI ``FluxKVCache``), valid only
    # with ``ref_method="index_timestep_zero"`` (enforced by the pipeline).
    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": True}

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a Qwen-Image DiT.

        Qwen-Image's distinctive blocks are the dual-stream projections (``img_in.`` /
        ``txt_in.``) and the joint time/text embedding (``time_text_embed.``). Keys
        are normalized first so repackaged checkpoints resolve identically.

        The dual-stream text projections (``add_q_proj``) are what separates it from
        Qwen-Image 2.1, a single-stream model that carries the same three prefixes.
        """
        keys = list(normalize_keys(f.keys()))
        has_img_in = any(k.startswith("img_in.") for k in keys)
        has_txt_in = any(k.startswith("txt_in.") for k in keys)
        has_time_text_embed = any(k.startswith("time_text_embed.") for k in keys)
        has_txt_stream = any(k.startswith("transformer_blocks.0.attn.add_q_proj.") for k in keys)
        return has_img_in and has_txt_in and has_time_text_embed and has_txt_stream

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        logger.info("Loading Qwen-Image DiT from %s", config.dit_path)
        self.dit = qwen_models.load_qwen_image_dit(
            config.dit_path, device=self.offload_device, dtype=config.dtype
        )
        self.dit.eval().requires_grad_(False)

        tokenizer_dir = QWEN25_TOKENIZER_CONFIG_DIR
        logger.info("Loading Qwen2.5-VL text encoder from %s", config.text_encoder_path)
        self.text_encoder = load_qwen2_5_vl_model(
            config.text_encoder_path, dtype=config.dtype, device=self.offload_device
        )
        self.text_encoder.eval().requires_grad_(False)
        self.tokenizer = load_qwen2_tokenizer(tokenizer_dir)
        self.vl_processor = load_qwen2_5_vl_processor(self.tokenizer)

        self.vae = load_qwen_vae(self.vae_path, device=self.device)
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

        # Precompute the RoPE frequencies once per prompt; they are independent of
        # image/timestep and are reused across every denoise step. The image stream
        # covers the concatenated base+ref tokens; the text stream uses a single
        # index (``max_vid_index + j``) advanced across all three axes. Target and
        # reference positions are stored apart so ``denoise_step`` can drop the
        # references from the sequence once the KV cache has frozen their K/V.
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

        # Timestep-zero conditioning (``ref_method="index_timestep_zero"``) zeroes
        # the timestep on the reference tokens; the split point is the base image
        # token count. Only meaningful with references present, so an ``index`` edit
        # (or plain t2i) leaves it None -> single-row modulation.
        zero_cond_t = ref_method == "index_timestep_zero" and self._ref_tokens is not None
        self._timestep_zero_index = num_img_tokens if zero_cond_t else None

        # Reference-latent KV cache: fresh per run, one cache per conditioning branch
        # (see ``DiffusionModel.start_kv_caches``).
        self.start_kv_caches(
            params,
            has_reference=self._ref_tokens is not None,
            has_uncond=params.guidance_scale > 1.0 and self._null_txt is not None,
        )

        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        # ``mu`` (the dynamic shift) is computed from the *packed* latent token
        # count (H/16 * W/16), matching musubi-tuner's ``image_seq_len =
        # latents.shape[1]`` after ``pack_latents``. Using the raw 8x-compressed
        # grid (H/8 * W/8) inflates mu by 4x and denoises at the wrong timesteps.
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
        """One Qwen-Image DiT forward (+ CFG), returning the velocity.

        With the KV cache the reference tokens reach the DiT only while the cache is
        filling; on every later step they are dropped and each block keeps working on
        its cache buffers, whose reference suffix is left untouched
        (``QwenImageTransformer2DModel.forward`` decides fill vs read from
        ``kv.filled``).
        """
        dev = torch.device(self.device)
        t_full = torch.full((1,), float(t), dtype=latents.dtype, device=dev)
        pe_img = self.dit.pe_embedder["img"]
        pe_ref = self.dit.pe_embedder["ref"]

        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=self.dtype):
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
                # Re-normalize the CFG combination back to the conditional norm
                # (Qwen-Image guidance trick), preventing the extrapolated
                # prediction from blowing up / collapsing into a pattern.
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

        ``kv`` is ``None`` for the plain path. On the first forward of a fresh cache
        (``not kv.filled``) the reference tokens are present (fill); on every later
        step (``kv.filled``) they are dropped (read), leaving their frozen K/V in
        every block's cache buffers. The reference positions (``pe_ref``) are only
        needed while filling.
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
        # Drop the run-scoped KV cache before the VAE decode (see ``end_kv_caches``).
        self.end_kv_caches()
        # Unpack the DiT tokens back to the canonical 4D latent.
        return unpack_latents(
            latents, params.height // self.vae.spatial_compression, params.width // self.vae.spatial_compression
        )

    @property
    def _pixels_per_token(self) -> int:
        """Pixels per DiT token: the VAE's compression times the DiT's 2x2 patchify.

        The latent geometry is the VAE's (``z_dim`` / ``spatial_compression``); the
        patchify on top of it is the DiT's own, so the two stay separate concerns.
        """
        return self.vae.spatial_compression * self.dit.patch_size

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # The latent grid is patchified in 2x2 blocks on an 8x-VAE-compressed latent.
        align = self._pixels_per_token
        return round_up(width, align), round_up(height, align)

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) -> canonical reference latent."""
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(self, latents: torch.Tensor, method: str = "index", ref_index: int = 1):
        """Canonical reference latent -> packed DiT tokens (native Qwen-Image approach).

        ``index_timestep_zero`` packs identically to ``index`` (the difference is the
        timestep-zero *modulation* of the reference tokens, applied via
        ``timestep_zero_index``); anything else is rejected.
        """
        if method not in ("index", "index_timestep_zero"):
            raise ValueError(
                f"unsupported ref_latents_method {method!r}; expected 'index' or 'index_timestep_zero'"
            )
        dev = torch.device(self.device)
        return pack_latents(latents.to(device=dev, dtype=self.dtype)), None

    def _upscale_format(self) -> str:
        """Qwen-Image VAE -> Wan21 z-score latent format."""
        return "wan21"


__all__ = ["QwenImageModel"]
