"""Z-Image (S3-DiT) adapter — supports the distilled Z-Image-Turbo checkpoint.

Turbo is an 8-NFE flow model with guidance disabled. The latent is the canonical 4D
format ([B, 16, H//8, W//8]); the VAE is Flux's (decoder) and the caption encoder is
Qwen3.
"""
from __future__ import annotations

import logging

import torch

from thenoise.dit.lumina.keys import has_learned_pad_tokens, is_s3dit
from thenoise.dit.zimage import sampling as zimage_sampling
from thenoise.dit.zimage.utils import (
    load_zimage_dit,
    load_zimage_text_encoder,
)
from thenoise.utils.text_encoder import find_tokenizer_dir
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    normalize_keys,
)
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, SesquiLSRUpscaler
from thenoise.utils.lora import FUSE_QKV
from thenoise.utils.math import round_up
from thenoise.vae import load_flux_vae

logger = logging.getLogger(__name__)


class ZImageModel(DiffusionModel):
    name = "zimage"

    # Distilled Turbo defaults (guidance 1 = CFG off).
    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 8,
        "guidance_scale": 1.0,
        "sampler": "euler",
    }

    MAX_SEQUENCE_LENGTH = 512

    lora_fusions = FUSE_QKV

    def _lora_key_map(self, key: str) -> str:
        """Diffusers LoRAs name the output projection ``to_out.0``; the model names it
        ``out``."""
        return key.replace(".to_out.0", ".out")

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is the Z-Image S3-DiT: the family signature plus its
        learned padding."""
        keys = list(normalize_keys(f.keys()))
        return is_s3dit(keys) and has_learned_pad_tokens(keys)

    def __init__(
        self,
        *,
        config: ModelConfig,
    ):
        super().__init__(config=config)

        logger.info("Loading Z-Image DiT from %s", config.dit_path)
        self.dit = load_zimage_dit(config.dit_path, device=self.offload_device, dtype=config.dtype)

        logger.info("Loading Z-Image text encoder from %s", config.text_encoder_path)
        self.text_encoder, self.tokenizer = load_zimage_text_encoder(
            config.text_encoder_path,
            dtype=config.dtype,
            device=self.offload_device,
            tokenizer_dir=find_tokenizer_dir(config.text_encoder_path),
        )

        # Flux VAE (decoder-only).
        self.vae = load_flux_vae(self.vae_path, device=self.device, dtype=self.dtype)

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("Z-Image model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(
        self,
        args: EncodePromptArgs,
    ) -> Conditioning:
        cond = self._encode_prompt(args.prompt)
        null = None
        if args.guidance_scale > 1.0:
            null = self._encode_prompt(args.negative_prompt)
        return Conditioning(cond=cond, null=null)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Chat template -> tokenize -> caption embeddings ``[1, n_valid, cap_feat_dim]``.

        Takes ``hidden_states[-2]`` and keeps only the non-padded tokens.
        """
        dev = torch.device(self.device)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(dev)
        mask = inputs.attention_mask.to(dev).bool()

        out = self.text_encoder(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
        emb = out.hidden_states[-2]  # [1, seq, 2560]
        valid = emb[0][mask[0]]  # [n_valid, 2560]
        return valid.unsqueeze(0).to(self.dtype)

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        # The DiT patchifies internally, so the noise is the raw VAE latent.
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
    ) -> torch.Tensor:
        # The DiT expects an F (frame) axis: [B, C, H, W] -> [B, C, 1, H, W].
        latents = latents.unsqueeze(2)
        self.dit.prepare_rope([latents[0]], [cond.cond[0]])
        if cond.null is not None:
            self.dit.prepare_rope([latents[0]], [cond.null[0]], key="_neg", clear=False)
        return latents

    def schedule(self, params: SamplingParams) -> list[Step]:
        dev = torch.device(self.device)
        sigmas = zimage_sampling.get_sigmas(params.steps, dev)
        # Step.t carries the sigma grid (1 -> 0); the model timestep t = 1 - sigma is
        # derived in ``denoise_step``.
        return [
            Step(t=sigmas[i], delta=sigmas[i] - sigmas[i + 1])
            for i in range(params.steps)
        ]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        dev = torch.device(self.device)
        x_list = [latents[0]]  # [C, 1, H, W]
        cap = [cond.cond[0]]   # [n_valid, cap_feat_dim]
        # ``t`` is sigma (see ``schedule``); the DiT's model timestep is ``1 - sigma``.
        t_full = torch.full((1,), 1.0 - float(t), device=dev, dtype=latents.dtype)

        # The reference integrates ``-model_out``, so the velocity is the negated
        # DiT output.
        pos = self.dit(x_list, t_full, cap)[0].unsqueeze(0)  # [1, C, 1, H, W]
        v_pos = -pos
        if guidance_scale > 1.0 and cond.null is not None:
            neg = self.dit(x_list, t_full, [cond.null[0]], rope_key="_neg")[0].unsqueeze(0)
            v_uncond = -neg
            v = v_uncond + guidance_scale * (v_pos - v_uncond)
        else:
            v = v_pos
        return v

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # [B, C, 1, H, W] -> [B, C, H, W]
        return latents.squeeze(2)

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # Pixel dims must be a multiple of the VAE compression * DiT patch size.
        align = self.vae.spatial_compression * self.dit.patch_size
        return round_up(width, align), round_up(height, align)

    def _create_upscaler(self) -> LatentUpscaler:
        return SesquiLSRUpscaler("flux", device=self.device, dtype=self.dtype)

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma (ER-SDE needs sigma_0 < 1)."""
        return 1.0 - percent
