"""Qwen-Image 2.1 adapter — single-stream DiT with a causal text/reference prefix.

Qwen3-VL-8B conditioner, Wan-2.2-layout 64-channel/16x RGBA VAE. The latent is the
DiT's own input: ``img_in`` takes the VAE's 64 channels directly and one token is one
latent cell, so there is no pack/unpack step and ``prepare_latent``/
``finalize_latent`` only move it.

Editing feeds the reference in twice: as vision tokens to the text encoder, and as a
VAE latent spliced into the DiT's text stream at the slot the tokenizer recorded. The
whole text + reference prefix is modulated at ``t = 0`` and is causally upstream of
the target, so its K/V are step-invariant and the KV cache is exact. That timestep-zero
conditioning is architectural, so ``index_timestep_zero`` is the only reference method.

The two halves of an edit are kept in step by construction: the reference is resized
for the text encoder with the same cover-and-crop the pipeline applies before
``encode_reference``, so the vision tokens removed and the latent tokens replacing
them describe the same pixels (one vision token per 32x32 pixels, one latent cell per
16x16 — hence the 32-pixel size alignment).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from thenoise.dit.qwen_image21 import sampling as qwen21_sampling
from thenoise.dit.qwen_image21.encoder import (
    QwenImage21TextEncoder,
    load_qwen_image21_text_encoder,
)
from thenoise.dit.qwen_image21.models import QwenImage21Sequence
from thenoise.dit.qwen_image21.utils import is_qwen_image21_key
from thenoise.dit.qwen_image21.utils import load_qwen_image21_dit
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    normalize_keys,
)
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, Qwen21TranscodeUpscaler
from thenoise.utils.image_tensor import flatten_alpha, resize_to_cover_center_crop
from thenoise.utils.lora import FUSE_GATE_UP
from thenoise.utils.math import round_up
from thenoise.vae import load_wan22_vae

logger = logging.getLogger(__name__)


@dataclass
class QwenImage21Conditioning(Conditioning):
    """``Conditioning`` plus the per-branch reference-image token slots.

    ``cond_slots[i]`` is the index in ``cond`` where the ``i``-th reference latent is
    spliced into the text stream. They are per-branch because the two prompts have
    different lengths.
    """

    cond_slots: Optional[List[int]] = None
    null_slots: Optional[List[int]] = None


class QwenImage21Model(DiffusionModel):
    name = "qwen_image21"

    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 28,
        "guidance_scale": 1.0,
        "sampler": "euler",
        # The prefix is modulated at t = 0 by construction: the only reference
        # method, and its K/V are exactly step-invariant (``KV_CACHED_SLICE``).
        "ref_method": "index_timestep_zero",
        "kv_cache": True,
    }

    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": True}

    # The step-invariant slice is the LEADING text/reference prefix (target last).
    KV_CACHED_SLICE = "prefix"

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a Qwen-Image 2.1 DiT: its per-run ``modulation``
        and zero-centred ``txt_in.text_norm``."""
        keys = set(normalize_keys(f.keys()))
        return is_qwen_image21_key(keys)

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.dit = load_qwen_image21_dit(
            config.dit_path, device=self.offload_device, dtype=config.dtype
        )

        # A fused-MLP checkpoint needs a LoRA's split SwiGLU names fused onto gate_up.
        self.lora_fusions = FUSE_GATE_UP if self.dit.params.fused_mlp else {}

        # Per-run state, filled in by ``prepare_latent``.
        self._seqs: dict[str, QwenImage21Sequence] = {}

        logger.info("Loading Qwen-Image 2.1 text encoder (Qwen3-VL-8B) from %s", config.text_encoder_path)
        self.text_encoder: QwenImage21TextEncoder = load_qwen_image21_text_encoder(
            config.text_encoder_path, dtype=config.dtype, device=self.offload_device
        )

        self.vae = load_wan22_vae(self.vae_path, device=self.device, dtype=config.dtype)
        if self.vae.z_dim != self.dit.in_channels:
            raise ValueError(
                f"the VAE's {self.vae.z_dim}ch latent does not feed this DiT's "
                f"{self.dit.in_channels}ch input"
            )

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("Qwen-Image 2.1 model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def _encoder_images(self, args: EncodePromptArgs) -> Optional[list]:
        """``args.image`` (single or list) as RGB, at the size the VAE saw.

        An alpha is composited onto white: the vision tower takes RGB only.
        """
        if args.image is None:
            return None
        images = args.image if isinstance(args.image, list) else [args.image]
        if not images:
            return None
        if args.width and args.height:
            images = [resize_to_cover_center_crop(img, args.width, args.height) for img in images]
        return [flatten_alpha(img) for img in images]

    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        """Prompt (and, when editing, the references) -> embeddings + image slots.

        One prompt at a time needs no padding, so there is no attention mask: the
        DiT splices the reference latents into the vision tokens' places.
        """
        images = self._encoder_images(args)
        cond, cond_slots = self.text_encoder(args.prompt, images)
        null = null_slots = None
        if args.guidance_scale > 1.0:
            null, null_slots = self.text_encoder(args.negative_prompt, images)
        return QwenImage21Conditioning(
            cond=cond, null=null, cond_slots=cond_slots, null_slots=null_slots
        )

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
        """Build each branch's causal sequence once for the run and keep the latent.

        The sequence holds the *projected* text + reference tokens and their RoPE
        table, so a denoise step is only the target-image tokens plus the block loop.
        Each branch gets its own, keyed ``cond`` / ``uncond`` like the KV caches.
        """
        dev = torch.device(self.device)
        x = latents.to(device=dev, dtype=self.dtype)

        refs = None
        if ref is not None:
            refs = [self.pack_reference_latent(r, ref_method, ref_index=i + 1)[0]
                    for i, r in enumerate(ref)]

        self._seqs = {
            "cond": self.dit.build_sequence(
                x, cond.cond, refs, getattr(cond, "cond_slots", None),
                name="seq", dtype=self.dtype,
            )
        }
        if cond.null is not None:
            self._seqs["uncond"] = self.dit.build_sequence(
                x, cond.null, refs, getattr(cond, "null_slots", None),
                name="seq_uncond", dtype=self.dtype,
            )

        self.start_kv_caches(
            params,
            has_reference=refs is not None,
            has_uncond=params.guidance_scale > 1.0 and "uncond" in self._seqs,
        )
        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        # One token per latent cell, so the shift's token count is the 16x grid.
        comp = self.vae.spatial_compression
        image_seq_len = (params.height // comp) * (params.width // comp)
        ts = qwen21_sampling.get_schedule(params.steps, image_seq_len)
        return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(params.steps)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One DiT forward (+ CFG) returning the velocity, in canonical latent form."""
        dev = torch.device(self.device)
        t_full = torch.full((1,), float(t), dtype=latents.dtype, device=dev)
        with torch.autocast(device_type=dev.type, dtype=self.dtype):
            pos = self.dit(latents, t_full, self._seqs["cond"], self.kv_cache("cond"))
            if guidance_scale > 1.0 and "uncond" in self._seqs:
                neg = self.dit(latents, t_full, self._seqs["uncond"], self.kv_cache("uncond"))
                v = neg + guidance_scale * (pos - neg)
            else:
                v = pos
        return v

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        self.end_kv_caches()
        self._seqs = {}
        return latents

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Align to 32 pixels: two latent cells, and one Qwen3-VL vision token.

        The edit path needs the extra factor: the reference latent replaces whole
        32x32 vision tokens.
        """
        align = 2 * self.vae.spatial_compression
        return round_up(width, align), round_up(height, align)

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) -> canonical reference latent."""
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "index_timestep_zero",
        ref_index: int = 1,
    ) -> Tuple[torch.Tensor, None]:
        """Validate the reference method and hand back the latent unchanged."""
        if method != "index_timestep_zero":
            raise ValueError(
                f"unsupported ref_latents_method {method!r}; Qwen-Image 2.1 always "
                "conditions its text/reference prefix at timestep zero"
            )
        dev = torch.device(self.device)
        return latents.to(device=dev, dtype=self.dtype), None

    # -------------------------------------------------------------- upscaling
    def _create_upscaler(self) -> LatentUpscaler:
        return Qwen21TranscodeUpscaler(self.vae, device=self.device, dtype=self.dtype)


__all__ = ["QwenImage21Conditioning", "QwenImage21Model"]
