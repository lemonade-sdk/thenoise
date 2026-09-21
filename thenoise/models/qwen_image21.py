"""Qwen-Image 2.1 adapter — single-stream DiT with a causal text/reference prefix.

Qwen-Image 2.1 shares nothing but a name with Qwen-Image 1: a different DiT (see
:mod:`thenoise.dit.qwen_image21.models`), a Wan-2.2-layout 64-channel/16x RGBA VAE,
and a Qwen3-VL-8B conditioner.

That VAE is the engine's first RGBA one, so this is the model the pipeline's channel
count exists for (see ``DiffusionModel.pixel_channels``). The boundaries that cannot
carry an alpha — the Qwen3-VL vision tokens and the pixel-domain upscaler —
composite it onto white.

The latent is the DiT's own input: ``img_in`` takes the VAE's 64 channels directly
and one token is one latent cell, so unlike Qwen-Image 1 or Flux Klein there is no
pack/unpack step and ``prepare_latent``/``finalize_latent`` only move it.

Editing feeds the reference in twice, like Qwen-Image 1: as vision tokens to the
text encoder, and as a VAE latent spliced into the DiT's text stream at the slot the
tokenizer recorded. The whole text + reference prefix is modulated at ``t = 0`` and
is causally upstream of the target, so its K/V are step-invariant and the KV cache
is exact rather than approximate. That timestep-zero conditioning is architectural —
the model has no ``index`` reference method — hence the ``index_timestep_zero``
default and the rejection of anything else.

The two halves of an edit are kept in step by construction: the reference is resized
for the text encoder with the same cover-and-crop the pipeline applies before
``encode_reference``, so the vision tokens removed and the latent tokens replacing
them describe the same pixels (one vision token per 32x32 pixels, one latent cell
per 16x16 — hence the 32-pixel size alignment).

LoRA note: the released checkpoints fuse the SwiGLU gate and up projections into one
``img_mlp.gate_up`` matrix, so a LoRA trained against the diffusers ``img_mlp.proj`` /
``img_mlp.gate_layer`` names only lands on the attention projections (the unmatched
factors are reported as unused at apply time).
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
from thenoise.utils.image_tensor import flatten_alpha, resize_to_cover_center_crop
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
        # The prefix is modulated at t = 0 by construction, so that is the only
        # reference method, and its K/V are exactly step-invariant
        # (``KV_CACHED_SLICE``), so freezing them costs no accuracy.
        "ref_method": "index_timestep_zero",
        "kv_cache": True,
    }

    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": True}

    # The step-invariant slice is the LEADING text/reference prefix (target last).
    KV_CACHED_SLICE = "prefix"

    # Separate ``to_q``/``to_k``/``to_v`` projections: LoRA factors must not be fused.
    fused_attention = False

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a Qwen-Image 2.1 DiT.

        The shared per-run ``modulation`` and the zero-centred ``txt_in.text_norm``
        exist in no other model. Keys are normalized first so repackaged checkpoints
        resolve identically.
        """
        keys = set(normalize_keys(f.keys()))
        return is_qwen_image21_key(keys)

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.dit = load_qwen_image21_dit(
            config.dit_path, device=self.offload_device, dtype=config.dtype
        )
        self.dit.eval().requires_grad_(False)

        # Per-run state, filled in by ``prepare_latent``.
        self._seqs: dict[str, QwenImage21Sequence] = {}

        # Qwen3-VL-8B with its vision tower: an edit's reference goes in here as
        # vision tokens as well as into the DiT as a latent.
        logger.info("Loading Qwen-Image 2.1 text encoder (Qwen3-VL-8B) from %s", config.text_encoder_path)
        self.text_encoder: QwenImage21TextEncoder = load_qwen_image21_text_encoder(
            config.text_encoder_path, dtype=config.dtype, device=self.offload_device
        )

        # Wan-2.2-layout VAE: 64ch latent, 16x, RGBA.
        self.vae = load_wan22_vae(self.vae_path, device=self.device, dtype=config.dtype)
        self.vae.eval().requires_grad_(False)
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

        The pipeline builds each reference latent from the same cover-and-crop, so
        the vision tokens and the latent tokens that replace them describe the same
        pixels (sizes are aligned to one vision token, see :meth:`resolve_size`).

        The vision tower only speaks RGB, so this is a boundary where an alpha has
        to go: it is composited onto white rather than dropped.
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

        There is no attention mask: one prompt at a time needs no padding, and the
        vision tokens are removed rather than masked — the DiT splices the reference
        latents into their places.
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
        Each conditioning branch needs its own (the uncond prompt is usually shorter),
        keyed ``cond`` / ``uncond`` like the run's KV caches.
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

        # Fresh KV caches, one per conditioning branch.
        self.start_kv_caches(
            params,
            has_reference=refs is not None,
            has_uncond=params.guidance_scale > 1.0 and "uncond" in self._seqs,
        )
        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        # One token per latent cell (no patchify), so the token count that drives the
        # dynamic shift is the 16x-compressed grid.
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
        """One DiT forward (+ CFG) returning the velocity, in canonical latent form.

        Plain CFG — Qwen-Image 1's norm-renormalisation is not part of this recipe.
        """
        dev = torch.device(self.device)
        t_full = torch.full((1,), float(t), dtype=latents.dtype, device=dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=self.dtype):
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

        The VAE alone would only need 16; the extra factor is the edit path, where
        the reference latent must replace whole 32x32 vision tokens.
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
        """Validate the reference method and hand back the latent unchanged.

        There is nothing to pack, and ``index`` — which would mean "condition the
        reference like the target" — does not exist here, so it is rejected rather
        than silently behaving like ``index_timestep_zero``.
        """
        if method != "index_timestep_zero":
            raise ValueError(
                f"unsupported ref_latents_method {method!r}; Qwen-Image 2.1 always "
                "conditions its text/reference prefix at timestep zero"
            )
        dev = torch.device(self.device)
        return latents.to(device=dev, dtype=self.dtype), None

    # -------------------------------------------------------------- upscaling
    def _upscale_format(self) -> str:
        raise NotImplementedError(
            "no Sesqui latent upscaler exists for the Qwen-Image 2.1 VAE yet (it is "
            "still training); name its latent format in _UPSCALER_FORMATS and return "
            "it here once the weights are committed"
        )


__all__ = ["QwenImage21Conditioning", "QwenImage21Model"]
