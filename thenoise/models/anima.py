"""Anima (Cosmos-Predict2 2B text2image) adapter."""
from __future__ import annotations

import logging

import torch

from thenoise.dit.anima import utils as anima_utils
from thenoise.dit.anima import sampling as anima_sampling
from thenoise.dit.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    normalize_keys,
)
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, SesquiLSRUpscaler
from thenoise.utils.math import round_up
from thenoise.utils.text_encoder import load_qwen3_text_encoder, load_t5_tokenizer
from thenoise.vae import load_qwen_vae

logger = logging.getLogger(__name__)


class AnimaModel(DiffusionModel):
    name = "anima"

    # Defaults for the turbo version
    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 8,
        "guidance_scale": 1,
    }
    DEFAULT_FLOW_SHIFT = 3.0

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is the Anima DiT: LLM adapter + adaln modulation."""
        keys = list(normalize_keys(f.keys()))
        has_llm_adapter = any("llm_adapter" in k for k in keys)
        has_adaln = any("adaln_modulation" in k for k in keys)
        return has_llm_adapter and has_adaln

    def __init__(
        self,
        *,
        config: ModelConfig,
    ):
        super().__init__(config=config)

        logger.info("Loading Anima DiT from %s", config.dit_path)
        self.dit = anima_utils.load_anima_model(
            self.offload_device,
            config.dit_path,
            dit_weight_dtype=config.dtype,
        )

        # Text encoder (Qwen3-0.6B) + tokenizers.
        logger.info("Loading Anima text encoder from %s", config.text_encoder_path)
        self.text_encoder, self.qwen3_tokenizer = load_qwen3_text_encoder(
            config.text_encoder_path, dtype=config.dtype, device=self.offload_device
        )
        self.t5_tokenizer = load_t5_tokenizer(None)

        self.tokenize_strategy = AnimaTokenizeStrategy(
            qwen3_tokenizer=self.qwen3_tokenizer,
            t5_tokenizer=self.t5_tokenizer,
            qwen3_max_length=512,
            t5_max_length=512,
        )
        self.encoding_strategy = AnimaTextEncodingStrategy()

        self.vae = load_qwen_vae(self.vae_path, device=self.device).to(self.dtype)

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("Anima model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(
        self,
        args: EncodePromptArgs,
    ) -> Conditioning:
        """Raw Qwen3/T5 embeddings; the DiT-side fusion happens in ``fuse_text``."""
        dev = torch.device(self.device)
        cond = self._encode_raw(args.prompt, dev)
        null = (
            self._encode_raw(args.negative_prompt, dev)
            if args.guidance_scale > 1.0
            else None
        )
        return Conditioning(cond=cond, null=null)

    def _encode_raw(self, prompt: str, dev: torch.device):
        """Tokenize -> Qwen3 encode -> ``[prompt_embeds, qwen3_mask, t5_ids, t5_mask]``."""
        tokens = self.tokenize_strategy.tokenize(prompt)
        return self.encoding_strategy.encode_tokens(
            self.tokenize_strategy, [self.text_encoder], tokens
        )

    def fuse_text(self, cond: Conditioning) -> Conditioning:
        """DiT-side fusion: raw embeddings -> LLM-adapter cross-attention conditioning."""
        dev = torch.device(self.device)
        fused = self._fuse_raw(cond.cond, dev)
        null = self._fuse_raw(cond.null, dev) if cond.null is not None else None
        return Conditioning(cond=fused, null=null)

    def _fuse_raw(self, embed, dev: torch.device) -> torch.Tensor:
        """DiT LLM-adapter cross-attention embedding (bf16)."""
        crossattn_emb = self.dit._preprocess_text_embeds(
            source_hidden_states=embed[0].to(dev),
            target_input_ids=embed[2].to(dev),
            target_attention_mask=embed[3].to(dev),
            source_attention_mask=embed[1].to(dev),
        )
        crossattn_emb[~embed[3].bool()] = 0
        return crossattn_emb.to(torch.bfloat16)

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
    ) -> torch.Tensor:
        # The Anima DiT expects a frame axis: [B, C, H, W] -> [B, C, 1, H, W].
        # RoPE depends only on (T, H, W), so compute it once for the whole run.
        latents = latents.unsqueeze(2)
        self.dit.pos_embedder.clear()
        self.dit.pos_embedder.store("emb", latents.shape, latents.device, dtype=self.dtype)
        return latents

    def schedule(self, params: SamplingParams) -> list[Step]:
        dev = torch.device(self.device)
        timesteps, sigmas = anima_sampling.get_timesteps_sigmas(params.steps, self.DEFAULT_FLOW_SHIFT, dev)
        timesteps = (timesteps / 1000).to(dev, dtype=self.dtype)
        sigmas = sigmas.to(dev)
        return [
            Step(t=timesteps[i], delta=sigmas[i] - sigmas[i + 1])
            for i in range(len(sigmas) - 1)
        ]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        t_expand = t.expand(latents.shape[0])
        noise_pred = self.dit(latents, t_expand, cond.cond)
        if guidance_scale > 1.0 and cond.null is not None:
            uncond = self.dit(latents, t_expand, cond.null)
            noise_pred = uncond + guidance_scale * (noise_pred - uncond)
        return noise_pred

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # [B, C, 1, H, W] -> [B, C, H, W]
        return latents.squeeze(2)

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # Pixel dims must be a multiple of the VAE compression * DiT patch size.
        align = self.vae.spatial_compression * self.dit.patch_spatial
        return round_up(width, align), round_up(height, align)

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma under the flow shift (ER-SDE needs sigma_0 < 1)."""
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        t = 1.0 - percent
        shift = self.DEFAULT_FLOW_SHIFT
        return (shift * t) / (1.0 + (shift - 1.0) * t)

    def _create_upscaler(self) -> LatentUpscaler:
        return SesquiLSRUpscaler("wan21", device=self.device, dtype=self.dtype)
