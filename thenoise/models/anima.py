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
from thenoise.utils.math import round_up
from thenoise.vae import load_qwen_vae

logger = logging.getLogger(__name__)


class AnimaModel(DiffusionModel):
    name = "anima"

    # Defaults for the turbo version
    DEFAULT_STEPS = 8
    DEFAULT_GUIDANCE_SCALE = 1
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    # Matches ComfyUI's Anima sampling_settings ``shift: 3.0`` (ModelSamplingDiscreteFlow).
    DEFAULT_FLOW_SHIFT = 3.0

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is the Anima DiT.

        Anima (Cosmos-Predict2) is uniquely identified by its LLM adapter
        (``llm_adapter.``) combined with adaln-modulated transformer blocks
        (``adaln_modulation``). We deliberately do NOT match on generic wrapper
        prefixes (``net.`` / ``model.diffusion_model.``) or generic attention
        names (``cross_attn``/``self_attn``) — those are shared across many
        model families and would falsely claim repackaged checkpoints of other
        models. Keys are normalized, then matched on Anima's own signature.
        """
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
            config.device,
            config.dit_path,
            loading_device=config.device,
            dit_weight_dtype=config.dtype,
        )
        self.dit.eval().requires_grad_(False)

        # Text encoder (Qwen3-0.6B) + tokenizers.
        logger.info("Loading Anima text encoder from %s", config.text_encoder_path)
        self.text_encoder, self.qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
            config.text_encoder_path, dtype=config.dtype, device=config.device
        )
        self.text_encoder.eval().requires_grad_(False)
        self.t5_tokenizer = anima_utils.load_t5_tokenizer(None)

        # Tokenize / encode strategies (called directly, not through the global registry).
        self.tokenize_strategy = AnimaTokenizeStrategy(
            qwen3_tokenizer=self.qwen3_tokenizer,
            t5_tokenizer=self.t5_tokenizer,
            qwen3_max_length=512,
            t5_max_length=512,
        )
        self.encoding_strategy = AnimaTextEncodingStrategy()

        # Qwen-Image VAE (single-frame decode).
        self.vae = (
            load_qwen_vae(self.vae_path, device=self.device, disable_mmap=True)
            .to(self.dtype)
            .eval()
            .requires_grad_(False)
        )

        logger.info("Anima model ready on %s (%s)", config.device, config.dtype)

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
        """Tokenize -> Qwen3 encode -> LLM-adapter cross-attention embedding (bf16)."""
        dev = torch.device(self.device)
        with torch.no_grad():
            tokens = self.tokenize_strategy.tokenize(prompt)
            # [prompt_embeds, qwen3_mask, t5_ids, t5_mask]
            embed = self.encoding_strategy.encode_tokens(self.tokenize_strategy, [self.text_encoder], tokens)
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
        num_channels = self.dit.LATENT_CHANNELS
        shape = (1, num_channels, params.height // self._VAE_SCALE, params.width // self._VAE_SCALE)
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
    ) -> torch.Tensor:
        # The Anima DiT expects a frame axis: [B, C, H, W] -> [B, C, 1, H, W].
        return latents.unsqueeze(2)

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
        with torch.no_grad():
            noise_pred = self.dit(latents, t_expand, cond.cond)
            if guidance_scale > 1.0 and cond.null is not None:
                uncond = self.dit(latents, t_expand, cond.null)
                noise_pred = uncond + guidance_scale * (noise_pred - uncond)
        return noise_pred

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # Drop the frame axis back to canonical 4D: [B, C, 1, H, W] -> [B, C, H, W].
        return latents.squeeze(2)

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # The latent grid is patchified in 2x2 blocks (patch_spatial=2) on an
        # 8x-VAE-compressed latent, so pixel dims must be multiples of
        # 8 * 2 = 16. Round up to the nearest multiple.
        align = 16
        return round_up(width, align), round_up(height, align)

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma.

        Used by the ER-SDE solver to nudge the first sigma just below 1.
        """
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        t = 1.0 - percent
        shift = self.DEFAULT_FLOW_SHIFT
        return (shift * t) / (1.0 + (shift - 1.0) * t)

    def _upscale_format(self) -> str:
        """Qwen-Image VAE -> Wan21 z-score latent format."""
        return "wan21"
