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
from thenoise.utils.math import round_up
from thenoise.vae import load_flux2_vae

logger = logging.getLogger(__name__)


class FluxKleinModel(DiffusionModel):
    name = "flux_klein"

    # Distilled defaults (the common inference use): 4 NFEs, CFG off (guidance 1.0),
    # Flux.2's flow-matching Euler schedule. Base models should pass
    # --steps 50 --guidance-scale 4.
    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 4,
        "guidance_scale": 1.0,
        "sampler": "euler",
    }

    # Reference-latent editing: Flux2 Klein supports the ComfyUI "index" method
    # with ``ref_index_scale = 10`` (the t-axis offset for the reference latent).
    supports_edit = True
    REF_INDEX = 10

    # Reference-latent KV cache: the reference tokens' K/V can be frozen across
    # denoise steps (ComfyUI's ``FluxKVCache``). Valid only with
    # ``ref_method="index_timestep_zero"`` (enforced by the pipeline).
    supports_kv_cache = True

    def _lora_key_map(self, key: str) -> str:
        """Map ComfyUI Flux.2 LoRA names to this repo's Flux.2 schema.

        ComfyUI names the double/single stream blocks ``transformer_blocks`` /
        ``single_transformer_blocks``, the fused single-stream projection
        ``attn.to_qkv_mlp_proj`` and the double-stream attention ``attn`` (with
        ``to_out.0``); the repo uses ``double_blocks``/``single_blocks``,
        ``linear1`` and ``img_attn``/``proj``. Applied after the generic
        q/k/v fusion in ``thenoise.utils.lora``.
        """
        key = key.replace(".attn.to_qkv_mlp_proj", ".linear1")
        key = key.replace("single_transformer_blocks", "single_blocks")
        key = key.replace(".attn.", ".img_attn.")
        key = key.replace("transformer_blocks", "double_blocks")
        key = key.replace(".to_out.0", ".proj")
        return key

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
        # selects the matching Qwen3 text encoder (4B / 8B). The timestep-zero
        # reference conditioning is NOT a weight-level property here: it is chosen
        # per run from the resolved ``ref_method`` preference (``prepare_latent``),
        # whose automatic layer comes from the checkpoint markers the base class
        # already read (``self.checkpoint_prefs``).
        self.params: Flux2Params = detect_klein_params(config.dit_path)
        self.is_8b = self.params.context_in_dim == 12288
        logger.info("Loading Flux Klein DiT (%s) from %s", self.variant_label, config.dit_path)
        self.dit = load_flux2_dit(config.dit_path, self.params, device=self.offload_device, dtype=config.dtype)
        self.dit.eval().requires_grad_(False)

        logger.info("Loading Flux Klein text encoder (Qwen3-%s) from %s", self.text_label, config.text_encoder_path)
        self.text_encoder = load_qwen3_embedder(
            config.text_encoder_path,
            is_8b=self.is_8b,
            dtype=config.dtype,
            device=self.offload_device,
            tokenizer_dir=find_tokenizer_dir(config.text_encoder_path),
        )

        # Flux.2 VAE (encoder + decoder).
        self.vae = load_flux2_vae(self.vae_path, device=self.device, dtype=self.dtype)
        self.vae.eval().requires_grad_(False)

        # Register swappable components with the memory manager.
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
        """Encode the edit instruction (text-only for Flux2 Klein).

        ``args.image`` is accepted for the shared edit pipeline but unused here —
        the input image is fed to the DiT purely as a reference latent, never into the
        text encoder (unlike Qwen Image Edit).
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
        plus ``[B, seq, 4]`` position ids. The text embeddings and their (fixed)
        position ids are stashed so the per-step ``denoise_step`` stays a pure DiT
        forward. Safe under the lock.

        In the edit path (``ref`` given) the reference latent is packed the same
        way and stashed as ``_ref_tokens`` for ``denoise_step``. ``ref_method``
        decides whether those tokens are conditioned at timestep zero
        (``index_timestep_zero`` -> ``zero_cond_t`` in the DiT forward).
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
            # Pack each ref with a successive t-axis index (REF_INDEX, 2x, ...)
            # per ComfyUI, then concat all ref tokens+ids into one stream. The
            # target image ids are kept separate from the reference ids so the KV
            # cache can drop the refs from the sequence (read mode) while still
            # knowing their positions (fill mode).
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

        # Reference-latent KV cache (ComfyUI ``FluxKVCache``): one cache per
        # conditioning branch, created fresh per run so a later request can never
        # reuse a stale cache. The uncond branch cache is only allocated when CFG
        # is actually active (``guidance_scale > 1.0``).
        self._kv: dict[str, KVCache] | None = None
        if params.kv_cache and self._ref_tokens is not None and self.supports_kv_cache:
            self._kv = {"cond": KVCache("cond")}
            if params.guidance_scale > 1.0 and self._un_txt is not None:
                self._kv["uncond"] = KVCache("uncond")

        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        image_seq_len = (params.width // self.vae.spatial_compression) * (
            params.height // self.vae.spatial_compression
        )
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

        With the KV cache the reference tokens are present only while the cache is
        filling; on every later step they are dropped and their cached K/V is
        re-appended (``Flux2.forward`` decides fill vs read from ``kv.filled``).
        """
        dev = torch.device(self.device)
        t_full = torch.full((len(latents),), float(t), dtype=latents.dtype, device=dev)
        pe_img = self.dit.pe_embedder["img"]
        pe_ref = self.dit.pe_embedder["ref"]
        kv_cond = self._kv["cond"] if self._kv is not None else None
        kv_uncond = self._kv.get("uncond") if self._kv is not None else None
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=self.dtype):
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

        ``kv`` is ``None`` for the plain path. On the first forward of a fresh
        cache (``not kv.filled``) the reference tokens are present (fill); on every
        later step (``kv.filled``) they are dropped (read). The reference positions
        (``pe_ref``) are only needed while filling.
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
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) -> canonical reference latent.

        Returns the Flux.2 packed latent ``[1, 128, H//16, W//16]`` (normalized).
        """
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "index",
        ref_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical reference latent -> (tokens, ids), t-axis = REF_INDEX*ref_index.

        ``ref_index`` is the 1-based position (ComfyUI ``ref_index_scale``): the
        first ref uses 10, the second 20, etc. ``index_timestep_zero`` packs
        identically to ``index`` (only the *modulation* differs, handled by
        ``zero_cond_t``); anything else is rejected rather than silently ignored.
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
        # Drop the run-scoped KV cache before the VAE decode (the controller
        # offloads the DiT right after this, so freeing the cache tensors avoids
        # holding GBs of frozen K/V through the decode).
        self._kv = None
        x = torch.cat(scatter_ids(latents, self._img_ids)).squeeze(2)  # [B, 128, H//16, W//16]
        return x

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # The packed latent is H//16 x W//16, so pixel dims must be multiples of 16.
        align = self.vae.spatial_compression
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
