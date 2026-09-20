"""Krea 2 (K2) adapter."""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from einops import rearrange

from thenoise.dit.krea2 import utils as krea2_utils
from thenoise.dit.krea2.reference import pack_reference
from thenoise.dit.krea2.sampling import encode_prompts, prepare, prepare_edit, timesteps
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


class Krea2Model(DiffusionModel):
    name = "krea2"

    # Model-owned defaults (incl. advanced sampler params -- not exposed to API/CLI).
    DEFAULT_STEPS = 8
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    DEFAULT_Y1 = 0.5
    DEFAULT_Y2 = 1.15
    DEFAULT_MU = 1.15

    # Reference-image editing (Krea 2 identity-edit / pose LoRA path).
    supports_edit = True
    DEFAULT_REF_METHOD = "fit"
    DEFAULT_GROUNDING_PX = 768

    # Resolution-aware schedule interpolation endpoints (image-token counts).
    DEFAULT_MINRES = 256
    DEFAULT_MAXRES = 1280

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is the Krea2 (single-stream MMDiT) DiT.

        Krea2's distinctive blocks are the text-fusion stream (``txtfusion.``)
        and text-MLP stream (``txtmlp.``). Repackaged checkpoints (e.g. ComfyUI
        exports) prefix every key with a generic wrapper such as
        ``model.diffusion_model.``, so keys are normalized first and the match
        is done on the architecture signature, not the raw prefix.
        """
        keys = list(normalize_keys(f.keys()))
        has_txtfusion = any(k.startswith("txtfusion.") for k in keys)
        has_txtmlp = any(k.startswith("txtmlp.") for k in keys)
        return has_txtfusion and has_txtmlp

    def __init__(
        self,
        *,
        config: ModelConfig,
    ):
        super().__init__(config=config)

        logger.info("Loading Krea 2 DiT from %s", config.dit_path)
        self.dit = krea2_utils.load_krea2_dit(
            config.dit_path,
            device=self.offload_device,
            dtype=config.dtype,
        )
        self.dit.eval().requires_grad_(False)

        logger.info("Loading Krea 2 text encoder from %s", config.text_encoder_path)
        self.encoder = krea2_utils.load_krea2_text_encoder(
            config.text_encoder_path,
            dtype=config.dtype,
            device=self.offload_device,
            tokenizer_dir=krea2_utils.find_tokenizer_dir(config.text_encoder_path),
        )

        # Qwen-Image VAE
        self.vae = (
            load_qwen_vae(self.vae_path, device=self.device)
            .to(self.dtype)
            .eval()
            .requires_grad_(False)
        )

        # VAE latent geometry (shared Qwen-Image VAE): 8x spatial compression.
        self._compression = self.vae.compression

        # Register swappable components with the memory manager.
        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.encoder)
        self.memory.register("vae", self.vae)

        logger.info("Krea 2 model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(
        self,
        args: EncodePromptArgs,
    ) -> Conditioning:
        """Text-encoder only: RAW prompt embeddings (DiT fusion in ``fuse_text``).

        In the edit path (``args.image`` set) the instruction is encoded image-grounded
        (the reference image feeds the Qwen3-VL as vision tokens); text-only stays the
        fast path.
        """
        cfg = args.guidance_scale > 1.0
        images = args.image if args.image is not None else None
        if images is not None and not isinstance(images, (list, tuple)):
            images = [images]
        txt, txtmask, untxt, untxtmask = encode_prompts(
            self.encoder,
            [args.prompt],
            [args.negative_prompt],
            cfg=cfg,
            images=images,
            grounding_px=self.DEFAULT_GROUNDING_PX,
        )
        return Conditioning(
            cond=txt, cond_mask=txtmask, null=untxt, null_mask=untxtmask
        )

    def fuse_text(self, cond: Conditioning) -> Conditioning:
        """DiT text fusion: raw embeddings -> cross-attention conditioning.

        Runs ONCE here (inside the dit block, DiT resident) so it is cached and
        reused across denoise steps. Fusion is independent of image/timestep, but
        depends on the (LoRA-adjusted) DiT weights, so it runs after
        ``switch_loras``.
        """
        dev = torch.device(self.device)
        with torch.no_grad():
            txt_fused = self.dit.fuse_text(
                cond.cond.to(device=dev, dtype=self.dtype),
                cond.cond_mask.to(device=dev),
            )
            untxt_fused = None
            if cond.null is not None:
                untxt_fused = self.dit.fuse_text(
                    cond.null.to(device=dev, dtype=self.dtype),
                    cond.null_mask.to(device=dev),
                )
        return Conditioning(
            cond=txt_fused,
            cond_mask=cond.cond_mask,
            null=untxt_fused,
            null_mask=cond.null_mask,
        )

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        return torch.randn(
            1,
            self.vae.z_dim,
            params.height // self._compression,
            params.width // self._compression,
            device=dev,
            dtype=self.dtype,
            generator=generator,
        )

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
        ref: Optional[list[torch.Tensor]] = None,
        ref_method: str = "fit",
    ) -> torch.Tensor:
        """Patchify the canonical latent and build pos/mask for the DiT, ONCE.

        ``prepare`` converts the latent to ``[B, seq, C*patch^2]`` image tokens and derives
        the combined image+text position/mask tensors; those are stashed on the instance so
        each ``denoise_step`` stays a pure DiT forward. In the edit path (``ref`` given) the
        reference tokens are prepended (frame=1..N) and ``ref_len`` is stashed so
        ``denoise_step`` drops them from the DiT output.
        """
        dev = torch.device(self.device)
        patch = self.dit.config.patch

        # Fresh prompt: drop any stale frequency entries from the previous one.
        self.dit.posemb.clear()

        txt = cond.cond.to(device=dev, dtype=self.dtype)
        txtmask = cond.cond_mask.to(device=dev)
        if ref is not None:
            img, pos, mask, ref_len = prepare_edit(
                latents, ref, txt.shape[1], patch, txtmask, ref_method
            )
            self._ref_len = ref_len
            self._ref_tokens = img[:, :ref_len]  # stashed refs, prepended in ``denoise_step``
            target = img[:, ref_len:]  # the sampler integrates over the target only
        else:
            img, pos, mask = prepare(latents, txt.shape[1], patch, txtmask)
            self._ref_len = 0
            self._ref_tokens = None
            target = img
        self.dit.posemb.store("cond", pos, dtype=self.dtype)
        self._txt, self._mask = txt, mask

        if cond.null is not None:
            untxt = cond.null.to(device=dev, dtype=self.dtype)
            untxtmask = cond.null_mask.to(device=dev)
            if ref is not None:
                _, unpos, unmask, _ = prepare_edit(
                    latents, ref, untxt.shape[1], patch, untxtmask, ref_method
                )
            else:
                _, unpos, unmask = prepare(latents, untxt.shape[1], patch, untxtmask)
            self.dit.posemb.store("uncond", unpos, dtype=self.dtype)
            self._untxt, self._unmask = untxt, unmask
        else:
            self._untxt = self._unmask = None

        return target

    def schedule(self, params: SamplingParams) -> list[Step]:
        patch = self.dit.config.patch
        align = self._compression * patch
        seq_len = (params.height // align) * (params.width // align)
        x1 = (self.DEFAULT_MINRES // align) ** 2
        x2 = (self.DEFAULT_MAXRES // align) ** 2
        ts = timesteps(
            seq_len, params.steps, x1, x2,
            y1=self.DEFAULT_Y1, y2=self.DEFAULT_Y2, mu=self.DEFAULT_MU,
        )
        return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(len(ts) - 1)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        dev = torch.device(self.device)
        device_type = torch.device(dev).type
        t_full = torch.full((len(latents),), t, dtype=latents.dtype, device=dev)
        # Prepend the reference tokens so the DiT sees ``[refs | target | text]``.
        img = latents
        if self._ref_tokens is not None:
            img = torch.cat([self._ref_tokens, latents], dim=1)
        with torch.autocast(device_type=device_type, dtype=self.dtype):
            cond_out = self.dit(
                img=img, context=self._txt, t=t_full, mask=self._mask,
                freqs=self.dit.posemb["cond"], ref_len=self._ref_len,
            )
            if guidance_scale > 1.0 and self._untxt is not None:
                uncond = self.dit(
                    img=img, context=self._untxt, t=t_full, mask=self._unmask,
                    freqs=self.dit.posemb["uncond"], ref_len=self._ref_len,
                )
                v = uncond + guidance_scale * (cond_out - uncond)
            else:
                v = cond_out
        return v

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # Unpatchify back to the canonical 4D latent [B, C, H//8, W//8].
        patch = self.dit.config.patch
        h_ = params.height // (self._compression * patch)
        w_ = params.width // (self._compression * patch)
        return rearrange(
            latents,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            ph=patch,
            pw=patch,
            h=h_,
            w=w_,
        )

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # The latent grid is patchified in `patch`-sized blocks, so width/height
        # must be multiples of compression * patch. Round up otherwise.
        align = self._compression * self.dit.config.patch
        return round_up(width, align), round_up(height, align)

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma (ComfyUI ModelSamplingFlux, shift=mu=1.15).

        Used by the ER-SDE solver to nudge the first sigma just below 1.
        """
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        t = 1.0 - percent
        mu = self.DEFAULT_MU
        return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))

    def _upscale_format(self) -> str:
        """Qwen-Image VAE -> Wan21 z-score latent format."""
        return "wan21"

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) -> canonical reference latent."""
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "fit",
        ref_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical reference latent -> (tokens, ids).

        ``method`` selects the training-matched geometry: ``fit`` (identity-edit, v1.2)
        fits the ref to the target grid with a centered offset.
        """
        patch = self.dit.config.patch
        if method == "fit":
            gh, gw = latents.shape[-2] // patch, latents.shape[-1] // patch
            return pack_reference(
                latents, gh, gw, patch, ref_index, device=torch.device(self.device)
            )
        raise ValueError(
            f"unsupported ref_latents_method {method!r}; "
            "supported: 'fit'"
        )
