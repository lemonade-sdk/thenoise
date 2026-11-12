"""Wan 2.2 TI2V-5B generation from Comfy-Org checkpoints."""
from __future__ import annotations

import io
import dataclasses
import logging
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from thenoise.dit.wan_video.checkpoint import load_wan_native_transformer
from thenoise.models.video_base import VideoModel
from thenoise.samplers import Step, create_sampler
from thenoise.dit.wan_video.sampling import get_sigmas, wan_frame_grid
from thenoise.text_encoders.wan_video import load_wan_text_encoder
from thenoise.utils.image_tensor import cover_crop, load_image, pil_to_pixels, pixels_to_pil
from thenoise.vae.wan_video import load_wan_native_video_vae
from thenoise.video.container import encode_video
from thenoise.video.jobs import Cancellation
from thenoise.video.spec import ResolvedVideo, VideoSpec

log = logging.getLogger("uvicorn.error")


class _WanVelocity:
    """Adapt Wan's velocity callback to the shared image/video sampler contract."""

    def __init__(self, predict):
        self._predict = predict

    def begin_sampling(self, _schedule):
        pass

    def denoise_step(self, x, t, _cond, _guidance_scale, _index):
        return self._predict(x, float(t))


class WanVideo(VideoModel):
    name = "Wan2.2-TI2V-5B"
    CAPABILITIES = {**VideoModel.CAPABILITIES, "t2v": True, "i2v": True}
    DEFAULT_PREFS = {
        "negative_prompt": "",
        "width": 1280,
        "height": 704,
        "frames": 33,
        "fps": 24,
        "steps": 12,
        "guidance_scale": 5.0,
        "output_format": "webm",
        "sampler": "unipc",
    }

    @staticmethod
    def detect(f) -> bool:
        from thenoise.utils.safetensors import unwrap_key
        keys = {unwrap_key(k) for k in f.keys()}
        return ({"patch_embedding.weight", "blocks.0.self_attn.q.weight",
                 "text_embedding.0.weight"}.issubset(keys) and
                f.get_slice(next(k for k in f.keys() if unwrap_key(k) == "patch_embedding.weight")).get_shape()[1] == 48)

    def __init__(self, *, config):
        super().__init__(config=config)
        self.transformer = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None

    @classmethod
    def resolve_video(cls, spec: VideoSpec) -> ResolvedVideo:
        """Apply this model's defaults and geometric restrictions."""
        resolved = super().resolve_video(spec)
        if resolved.sampler not in ("euler", "unipc"):
            raise ValueError("Wan TI2V-5B supports only euler or unipc samplers")
        width = (resolved.width // 32) * 32
        height = (resolved.height // 32) * 32
        if width == 0 or height == 0:
            raise ValueError("Wan video requires width and height of at least 32")
        wanted = (spec.frames if spec.frames is not None else
                  spec.duration * resolved.fps if spec.duration is not None else resolved.frame_count)
        return dataclasses.replace(
            resolved, width=width, height=height,
            frame_count=wan_frame_grid(wanted, upward=spec.frames is not None),
        )

    def _load(self, token: Cancellation) -> None:
        token.check()
        if self.transformer is not None:
            return
        vae = load_wan_native_video_vae(self.vae_path, check_cancel=token.check)
        token.check()
        # UMT5's gated FFN can overflow FP16 before its final normalization.
        # BF16 preserves the FP16 checkpoint's memory footprint but widens the exponent.
        text_encoder, tokenizer = load_wan_text_encoder(
            self.text_encoder_path, dtype=torch.bfloat16, check_cancel=token.check,
        )
        token.check()
        transformer = load_wan_native_transformer(
            self.dit_path, dtype=torch.float16, check_cancel=token.check,
        )
        token.check()
        if vae.dtype != torch.float32:
            raise RuntimeError("Wan causal video VAE must remain in FP32")
        self.memory.register("vae", vae)
        self.memory.register("text_encoder", text_encoder)
        self.memory.register("dit", transformer)
        self.vae, self.text_encoder, self.tokenizer, self.transformer = (
            vae, text_encoder, tokenizer, transformer,
        )
        log.info("Wan TI2V-5B (Comfy-Org weights) loaded")

    def _prompt_embeddings(self, prompt: str, token: Cancellation) -> torch.Tensor:
        ids, mask = self.tokenizer.encode(prompt)
        token.check()
        features = self.text_encoder(ids.to(self.device), mask.to(self.device))
        return features.to(dtype=self.transformer.dtype)

    def _stats(self, dtype: torch.dtype, device: torch.device):
        config = self.vae.config
        shape = (1, config.z_dim, 1, 1, 1)
        mean = torch.as_tensor(config.latents_mean, dtype=dtype, device=device).view(shape)
        std = torch.as_tensor(config.latents_std, dtype=dtype, device=device).view(shape)
        return mean, std

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(z.dtype, z.device)
        return (z - mean) / std

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(z.dtype, z.device)
        return z * std + mean

    def _image_condition(self, spec: ResolvedVideo, token: Cancellation) -> torch.Tensor:
        image = load_image(io.BytesIO(spec.image))
        frame = cover_crop(image, spec.width, spec.height)
        pixels = pil_to_pixels(frame, channels=3)
        video = pixels[None, :, None].to(self.device)
        token.check()
        latent = self.vae.encode(video, return_dict=False)[0].mode()
        token.check()
        return self._normalize(latent)

    def _sample(self, latents: torch.Tensor, spec: ResolvedVideo,
                positive: torch.Tensor, negative: torch.Tensor | None,
                condition: torch.Tensor | None, token: Cancellation, progress) -> torch.Tensor:
        patch_t, patch_h, patch_w = self.transformer.config.patch_size
        b, _, t, h, w = latents.shape
        mask = torch.ones((b, 1, t, 1, 1), device=latents.device, dtype=torch.float32)
        if condition is not None:
            if condition.shape != latents[:, :, :1].shape:
                raise ValueError("I2V first-frame latent shape mismatch")
            mask[:, :, 0] = 0
            latents = latents.clone()
            latents[:, :, :1] = condition
        # A token's timestep is zero on the conditioned I2V frame.
        patch_mask = mask.expand(-1, -1, -1, h, w)
        patch_mask = patch_mask[:, 0, ::patch_t, ::patch_h, ::patch_w].flatten(1)

        def predict(x: torch.Tensor, sigma: float) -> torch.Tensor:
            token.check()
            model_input = x.to(self.transformer.dtype)
            if condition is not None:
                model_input = model_input.clone()
                model_input[:, :, :1] = condition.to(model_input.dtype)
            timestep = patch_mask * (sigma * 1000.0)
            cond_vel = self.transformer(
                model_input, timestep=timestep, encoder_hidden_states=positive,
                return_dict=False,
            )[0].float()
            if negative is None:
                return cond_vel
            token.check()
            uncond_vel = self.transformer(
                model_input, timestep=timestep, encoder_hidden_states=negative,
                return_dict=False,
            )[0].float()
            return uncond_vel + spec.guidance_scale * (cond_vel - uncond_vel)

        def step_done(x: torch.Tensor, i: int, total: int) -> torch.Tensor:
            if condition is not None:
                x[:, :, :1] = condition.to(x.dtype)
            progress("sampling", i, total)
            token.check()
            return x

        sigmas = get_sigmas(spec.steps)
        ts = torch.tensor(sigmas, device=latents.device, dtype=torch.float32)
        schedule = [Step(t=ts[i], delta=ts[i] - ts[i + 1])
                    for i in range(len(ts) - 1)]
        return create_sampler(spec.sampler, _WanVelocity(predict)).sample(
            latents, schedule, None, spec.guidance_scale, spec.seed,
            step_callback=step_done,
        )

    def _decode(self, latents: torch.Tensor, token: Cancellation, progress):
        chunks = latents.shape[2]
        decoded = 0

        def before(_module, _args):
            token.check()

        def after(_module, _args, _result):
            nonlocal decoded
            decoded += 1
            progress("decoding", decoded, chunks)

        progress("decoding", 0, chunks)
        with ExitStack() as hooks:
            hooks.callback(self.vae.decoder.register_forward_pre_hook(before).remove)
            hooks.callback(self.vae.decoder.register_forward_hook(after).remove)
            try:
                video = self.vae.decode(self._denormalize(latents), return_dict=False)[0]
                token.check()
                return [pixels_to_pil(video[0, :, i]) for i in range(video.shape[2])]
            finally:
                self.vae.clear_cache()

    def generate(self, spec: ResolvedVideo, token: Cancellation, progress, output: Path) -> None:
        from thenoise.inference import inference, inference_lock

        pixels = spec.width * spec.height
        with inference_lock, inference():
            progress("loading", 0, 0)
            self._load(token)
            with self.memory.stage("text_encoder", pixels=pixels) as stage:
                positive = stage.run(self._prompt_embeddings, spec.prompt, token)
                negative = (stage.run(self._prompt_embeddings, spec.negative_prompt, token)
                            if spec.guidance_scale > 1 else None)
            token.check()

            condition = None
            if spec.image is not None:
                with self.memory.stage("vae", pixels=pixels) as stage:
                    condition = stage.run(self._image_condition, spec, token)

            shape = (1, 48, (spec.frame_count - 1) // 4 + 1,
                     spec.height // 16, spec.width // 16)
            rng = torch.Generator(device="cpu").manual_seed(spec.seed)
            latents = torch.randn(shape, generator=rng, dtype=torch.float32).to(self.device)
            start = time.monotonic()
            with self.memory.stage("dit", pixels=pixels) as stage:
                latents = stage.run(self._sample, latents, spec, positive, negative,
                                    condition, token, progress)
            token.check()
            log.info("Wan sampling finished in %.1fs", time.monotonic() - start)
            del positive, negative, condition

            with self.memory.stage("vae", pixels=pixels) as stage:
                frames = stage.run(self._decode, latents, token, progress)
            del latents

        token.check()
        progress("encoding", 0, spec.frame_count)
        encode_video(frames, spec, token, output, progress)
        log.info("Wan video saved: %s", output)

    def _clear_weights(self) -> None:
        self.vae = self.text_encoder = self.tokenizer = self.transformer = None
