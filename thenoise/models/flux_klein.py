"""Flux.2 (Flux Klein) adapter — supports the 4B and 9B Klein variants.

Flux Klein is a flow-matching MMDiT operating on the Flux.2 *packed* 128-channel
latent ``[B, 128, H//16, W//16]`` (the Flux.2 VAE packs a 32ch latent 2x2 and
normalizes it via BatchNorm). That is the canonical latent here: both the DiT and
the VAE take it directly and the adapter packs/unpacks only around the denoise loop.

The Klein variant (4B / 9B) is read from the checkpoint's ``img_in`` width and
selects the matching Qwen3 text encoder (4B / 8B).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from thenoise.dit.flux2.models import Flux2Params
from thenoise.dit.flux2.sampling import get_schedule, prc_img, prc_txt, scatter_ids
from thenoise.dit.flux2.utils import (
    detect_klein_params,
    load_flux2_dit,
    load_qwen3_embedder,
)
from thenoise.dit.kvcache import KVCache
from thenoise.utils.text_encoder import find_tokenizer_dir
from thenoise.models.base import Conditioning, DiffusionModel, Step, normalize_keys
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, SesquiLSRUpscaler
from thenoise.utils.lora import FUSE_QKV
from thenoise.utils.math import round_up
from thenoise.vae import load_flux2_vae

logger = logging.getLogger(__name__)


class FluxKleinModel(DiffusionModel):
    name = "flux_klein"

    # Distilled defaults; base models: --steps 50 --guidance-scale 4.
    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 4,
        "guidance_scale": 1.0,
        "sampler": "euler",
    }

    # Reference-latent editing by "index"; the KV cache is valid only with
    # ``ref_method="index_timestep_zero"``.
    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": True}
    # Per-reference t-axis offset.
    REF_INDEX = 10

    lora_fusions = FUSE_QKV

    def _lora_key_map(self, key: str) -> str:
        """Map ComfyUI Flux.2 LoRA names to this repo's Flux.2 schema."""
        key = key.replace(".attn.to_qkv_mlp_proj", ".linear1")
        key = key.replace("single_transformer_blocks", "single_blocks")
        key = key.replace(".attn.", ".img_attn.")
        key = key.replace("transformer_blocks", "double_blocks")
        key = key.replace(".to_out.0", ".proj")
        return key

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a Flux.2 DiT: its double/single stream modulations."""
        keys = list(normalize_keys(f.keys()))
        has_img = any(k.startswith("double_stream_modulation_img.") for k in keys)
        has_txt = any(k.startswith("double_stream_modulation_txt.") for k in keys)
        has_single = any(k.startswith("single_stream_modulation.") for k in keys)
        return has_img and has_txt and has_single

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        # Klein variant (4B / 9B) from the DiT checkpoint; selects the Qwen3 size.
        self.params: Flux2Params = detect_klein_params(config.dit_path)
        self.is_8b = self.params.context_in_dim == 12288
        logger.info("Loading Flux Klein DiT (%s) from %s", self.variant_label, config.dit_path)
        self.dit = load_flux2_dit(config.dit_path, self.params, device=self.offload_device, dtype=config.dtype)

        logger.info("Loading Flux Klein text encoder (Qwen3-%s) from %s", self.text_label, config.text_encoder_path)
        self.text_encoder = load_qwen3_embedder(
            config.text_encoder_path,
            is_8b=self.is_8b,
            dtype=config.dtype,
            device=self.offload_device,
            tokenizer_dir=find_tokenizer_dir(config.text_encoder_path),
        )

        self.vae = load_flux2_vae(self.vae_path, device=self.device, dtype=self.dtype)

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

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
        args: EncodePromptArgs,
    ) -> Conditioning:
        """Encode the prompt (and the negative, under CFG).

        The edit image reaches the DiT as a reference latent.
        """
        cond = self.text_encoder(args.prompt)  # [1, 512, ctx_dim]
        null = None
        if args.guidance_scale > 1.0:
            null = self.text_encoder(args.negative_prompt)
        return Conditioning(cond=cond, null=null)

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
        """Pack the canonical latent into DiT tokens and stash conditioning, ONCE.

        ``prc_img`` converts ``[B, 128, H//16, W//16]`` -> ``[B, seq, 128]`` tokens
        plus ``[B, seq, 4]`` position ids. The text embeddings and their position ids
        are stashed too so the per-step ``denoise_step`` stays a pure DiT forward.
        Safe under the lock.

        In the edit path the references are packed the same way and concatenated into
        ``_ref_tokens``; ``ref_method`` decides whether those tokens are conditioned
        at timestep zero (``index_timestep_zero`` -> ``zero_cond_t``).
        """
        dev = torch.device(self.device)
        self._zero_cond_t = ref_method == "index_timestep_zero"
        x, x_ids = prc_img(latents.to(device=dev, dtype=self.dtype))
        self._img_ids = x_ids  # used by ``finalize_latent``

        self._txt = cond.cond.to(device=dev, dtype=self.dtype)
        _, txt_ids = prc_txt(self._txt)

        if cond.null is not None:
            self._un_txt = cond.null.to(device=dev, dtype=self.dtype)
            _, un_txt_ids = prc_txt(self._un_txt)
        else:
            self._un_txt = un_txt_ids = None

        if ref is not None:
            # Each ref gets a successive t-axis index; the target's image ids stay
            # apart from the reference ids so the KV cache can drop the reference
            # tokens from the sequence while still knowing their positions.
            ref_tokens, ref_ids = [], []
            for i, ref_latent in enumerate(ref):
                t, ids = self.pack_reference_latent(ref_latent, ref_method, ref_index=i + 1)
                ref_tokens.append(t)
                ref_ids.append(ids)
            self._ref_tokens = torch.cat(ref_tokens, dim=1)
            ref_ids = torch.cat(ref_ids, dim=1)
            self._ref_token_count = self._ref_tokens.shape[1]
        else:
            self._ref_tokens = None
            self._ref_token_count = 0
            ref_ids = torch.zeros(1, 0, 4, device=dev, dtype=torch.long)

        self.dit.pe_embedder.clear()
        self.dit.pe_embedder.store("img", x_ids, dtype=self.dtype)
        self.dit.pe_embedder.store("ref", ref_ids, dtype=self.dtype)
        self.dit.pe_embedder.store("txt", txt_ids, dtype=self.dtype)
        if un_txt_ids is not None:
            self.dit.pe_embedder.store("txt_uncond", un_txt_ids, dtype=self.dtype)

        self.start_kv_caches(
            params,
            has_reference=self._ref_tokens is not None,
            has_uncond=params.guidance_scale > 1.0 and self._un_txt is not None,
        )

        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        image_seq_len = (params.width // self.vae.spatial_compression) * (
            params.height // self.vae.spatial_compression
        )
        ts = get_schedule(params.steps, image_seq_len)
        # Step.t is the flow timestep (1 -> 0); delta = t_i - t_{i+1}.
        return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(params.steps)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One Flux.2 DiT forward (+ CFG), returning the raw output as velocity.

        The Flux.2 flow ODE integrates ``x += (t_prev - t_curr) * v``, which is the
        shared Euler update ``x -= delta * v`` when ``v`` is the model's raw output.
        """
        dev = torch.device(self.device)
        t_full = torch.full((len(latents),), float(t), dtype=latents.dtype, device=dev)
        pe_img = self.dit.pe_embedder["img"]
        pe_ref = self.dit.pe_embedder["ref"]
        kv_cond = self.kv_cache("cond")
        kv_uncond = self.kv_cache("uncond")
        with torch.autocast(device_type=dev.type, dtype=self.dtype):
            pos = self._dit_forward(latents, t_full, self._txt, self.dit.pe_embedder["txt"], kv_cond, pe_img, pe_ref)
            if guidance_scale > 1.0 and self._un_txt is not None:
                neg = self._dit_forward(latents, t_full, self._un_txt, self.dit.pe_embedder["txt_uncond"], kv_uncond, pe_img, pe_ref)
                v = neg + guidance_scale * (pos - neg)
            else:
                v = pos
        return v

    def _dit_forward(
        self,
        x: torch.Tensor,
        t_full: torch.Tensor,
        ctx: torch.Tensor,
        pe_ctx: torch.Tensor,
        kv: KVCache | None,
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
            x=x,
            pe_x=pe_img,
            timesteps=t_full,
            ctx=ctx,
            pe_ctx=pe_ctx,
            ref_tokens=ref_tokens,
            ref_pe=pe_ref,
            kv=kv,
            zero_cond_t=self._zero_cond_t,
        )

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Input pixels (``[C,H,W]`` in [-1, 1]) -> packed latent ``[1, 128, H//16, W//16]``."""
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "index",
        ref_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical reference latent -> (tokens, ids) at t-axis ``REF_INDEX * ref_index``.

        ``ref_index`` is the 1-based position, so the refs land on 10, 20, ...
        ``index_timestep_zero`` packs identically to ``index``: only the modulation
        differs, via ``zero_cond_t``.
        """
        if method not in ("index", "index_timestep_zero"):
            raise ValueError(
                f"unsupported ref_latents_method {method!r}; expected 'index' or 'index_timestep_zero'"
            )
        dev = torch.device(self.device)
        index = self.REF_INDEX * ref_index
        return prc_img(
            latents.to(device=dev, dtype=self.dtype),
            t_coord=torch.tensor([index], device=dev),
        )

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Unpack the DiT tokens back to the canonical packed latent."""
        self.end_kv_caches()
        x = torch.cat(scatter_ids(latents, self._img_ids)).squeeze(2)  # [B, 128, H//16, W//16]
        return x

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # The packed latent already is the DiT's grid: one 16x16 pixel cell per token.
        align = self.vae.spatial_compression
        return round_up(width, align), round_up(height, align)

    def _create_upscaler(self) -> LatentUpscaler:
        return SesquiLSRUpscaler("flux2", device=self.device, dtype=self.dtype)

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma (ER-SDE needs sigma_0 < 1)."""
        return 1.0 - percent


__all__ = ["FluxKleinModel"]
