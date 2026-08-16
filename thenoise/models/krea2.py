"""Krea 2 (K2) adapter."""
from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
from einops import rearrange

from thenoise.dit.krea2 import utils as krea2_utils
from thenoise.dit.krea2.sampling import encode_prompts, prepare, timesteps
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    normalize_keys,
)
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
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_dir: Optional[str] = None,
        upscaler_dir: Optional[str] = None,
    ):
        super().__init__(
            dit_path=dit_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            device=device,
            dtype=dtype,
            lora_dir=lora_dir,
            upscaler_dir=upscaler_dir,
        )

        logger.info("Loading Krea 2 DiT from %s", dit_path)
        self.dit = krea2_utils.load_krea2_dit(
            dit_path,
            device=device,
            dtype=dtype,
        )
        self.dit.eval().requires_grad_(False)

        logger.info("Loading Krea 2 text encoder from %s", text_encoder_path)
        self.encoder = krea2_utils.load_krea2_text_encoder(
            text_encoder_path, dtype=dtype, device=device
        )

        # Qwen-Image VAE
        self.vae = (
            load_qwen_vae(self.vae_path, device=self.device, disable_mmap=True)
            .to(self.dtype)
            .eval()
            .requires_grad_(False)
        )

        # VAE latent geometry (shared Qwen-Image VAE): 8x spatial compression.
        self._compression = self.vae.compression

        logger.info("Krea 2 model ready on %s (%s)", device, dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        guidance_scale: float,
    ) -> Conditioning:
        cfg = guidance_scale > 1.0
        txt, txtmask, untxt, untxtmask = encode_prompts(
            self.encoder, [prompt], [negative_prompt], cfg=cfg
        )
        # Fuse the text stream ONCE here (prompt stage) so it is cached and reused
        # across denoise steps and across runs with the same prompt/LoRA config. The
        # fusion is independent of image/timestep, but depends on the (LoRA-adjusted)
        # DiT weights
        dev = torch.device(self.device)
        with torch.no_grad():
            txt_fused = self.dit.fuse_text(
                txt.to(device=dev, dtype=self.dtype),
                txtmask.to(device=dev),
            )
            untxt_fused = None
            if cfg:
                untxt_fused = self.dit.fuse_text(
                    untxt.to(device=dev, dtype=self.dtype),
                    untxtmask.to(device=dev),
                )
        return Conditioning(
            cond=txt_fused,
            cond_mask=txtmask,
            null=untxt_fused,
            null_mask=untxtmask,
        )

    def init_latents(self, height: int, width: int, seed: int) -> torch.Tensor:
        dev = torch.device(self.device)
        generator = torch.Generator(device=dev).manual_seed(seed)
        return torch.randn(
            1,
            self.vae.z_dim,
            height // self._compression,
            width // self._compression,
            device=dev,
            dtype=self.dtype,
            generator=generator,
        )

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Patchify the canonical latent and build pos/mask for the DiT, ONCE.

        ``prepare`` converts the latent to ``[B, seq, C*patch^2]`` image tokens
        and derives the combined image+text position/mask tensors. Those (plus the
        text embeddings, moved to device) are stashed on the instance so the
        per-step ``denoise_step`` stays a pure DiT forward. Safe under the lock.
        """
        dev = torch.device(self.device)
        patch = self.dit.config.patch

        txt = cond.cond.to(device=dev, dtype=self.dtype)
        txtmask = cond.cond_mask.to(device=dev)
        img, pos, mask = prepare(latents, txt.shape[1], patch, txtmask)
        self._freqs = self.dit.posemb(pos).to(self.dtype)
        self._txt, self._pos, self._mask = txt, pos, mask

        if cond.null is not None:
            untxt = cond.null.to(device=dev, dtype=self.dtype)
            untxtmask = cond.null_mask.to(device=dev)
            _, unpos, unmask = prepare(latents, untxt.shape[1], patch, untxtmask)
            self._freqs_un = self.dit.posemb(unpos).to(self.dtype)
            self._untxt, self._unpos, self._unmask = untxt, unpos, unmask
        else:
            self._untxt = self._unpos = self._unmask = None
            self._freqs_un = None

        return img

    def schedule(self, steps: int, height: int, width: int) -> list[Step]:
        patch = self.dit.config.patch
        align = self._compression * patch
        seq_len = (height // align) * (width // align)
        x1 = (self.DEFAULT_MINRES // align) ** 2
        x2 = (self.DEFAULT_MAXRES // align) ** 2
        ts = timesteps(
            seq_len, steps, x1, x2,
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
        with torch.autocast(device_type=device_type, dtype=self.dtype):
            cond_out = self.dit(
                img=latents, context=self._txt, t=t_full, pos=self._pos, mask=self._mask, freqs=self._freqs
            )
            if guidance_scale > 1.0 and self._untxt is not None:
                uncond = self.dit(
                    img=latents, context=self._untxt, t=t_full, pos=self._unpos, mask=self._unmask, freqs=self._freqs_un
                )
                v = uncond + guidance_scale * (cond_out - uncond)
            else:
                v = cond_out
        return v

    def finalize_latent(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Unpatchify back to the canonical 4D latent [B, C, H//8, W//8].
        patch = self.dit.config.patch
        h_ = height // (self._compression * patch)
        w_ = width // (self._compression * patch)
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
