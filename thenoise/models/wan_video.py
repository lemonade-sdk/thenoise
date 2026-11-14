"""Wan 2.2 generation from Comfy-Org checkpoints: one adapter, one variant per
released transformer (TI2V-5B, or the two files of T2V-A14B)."""
from __future__ import annotations

import dataclasses
import io
import logging
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from thenoise.dit.wan_video.checkpoint import load_wan_native_transformer
from thenoise.dit.wan_video.model import CHANNELS, Wan22Params, detect_params
from thenoise.utils.wan_weights import wan_dit_key
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


@dataclasses.dataclass(frozen=True)
class Experts:
    """A two-expert schedule: where the sigma hands over, and how the low-noise half
    differs."""

    boundary: float
    flow_shift: float
    #: Upstream ships guidance 4.0 high and 3.0 low.
    low_cfg_ratio: float


@dataclasses.dataclass(frozen=True)
class Variant:
    """What a released Wan 2.2 transformer needs to be driven as."""

    name: str
    prefs: dict
    i2v: bool
    wan21_vae: bool
    experts: Experts | None


#: One entry per released transformer, keyed by the variant name `CHANNELS` gives the
#: shape `detect_params` reads. `DEFAULT_PREFS` covers what the variants share; these
#: cover what a variant decides.
VARIANTS: dict[str, Variant] = {variant.name: variant for variant in (
    Variant(
        name="ti2v_5b",
        prefs=dict(width=1280, height=704, frames=33, fps=24, steps=12,
                   guidance_scale=5.0, sampler="unipc"),
        i2v=True, wan21_vae=False, experts=None),
    Variant(
        name="t2v_14b",
        prefs=dict(width=832, height=480, frames=33, fps=16, steps=40,
                   guidance_scale=4.0, sampler="euler"),
        i2v=False, wan21_vae=True,
        experts=Experts(boundary=0.875, flow_shift=12.0, low_cfg_ratio=3.0 / 4.0)),
)}


def variant_for(params: Wan22Params) -> Variant:
    """The policy for the variant a checkpoint says this is."""
    for variant in VARIANTS.values():
        if CHANNELS[variant.name] == params.in_channels:
            return variant
    raise ValueError(f"no Wan 2.2 policy for a {params.in_channels}-channel transformer")


class Wan22Model(VideoModel):
    """Wan 2.2 from Comfy-Org checkpoints.

    One adapter covers TI2V-5B and the two-expert T2V-A14B: the DiT header says which
    variant this is, and ``VARIANTS`` says what follows from that — request defaults,
    whether I2V exists at all, which codec, and whether a second expert file is
    required. I2V with a CLIP image tower would be a third variant, not this file.
    """

    name = "wan22"
    CAPABILITIES = {**VideoModel.CAPABILITIES, "t2v": True}
    DEFAULT_PREFS = {"negative_prompt": "", "output_format": "webm"}

    @staticmethod
    def detect(f) -> bool:
        """Any Wan 2.2 transformer: which variant it is comes from its shapes, later."""
        from thenoise.utils.safetensors import unwrap_key
        keys = {unwrap_key(k) for k in f.keys()}
        return {"patch_embedding.weight", "blocks.0.self_attn.q.weight",
                "text_embedding.0.weight"}.issubset(keys)

    @classmethod
    def validate_paths(cls, dit_paths: dict[str, str]) -> None:
        """A two-expert model is two files, so its second one is not optional."""
        experts = variant_for(detect_params(dit_paths[""])).experts
        if experts is None:
            if len(dit_paths) > 1:
                raise ValueError(
                    "--dit-low carries the low-noise half of a two-expert transformer; "
                    f"{dit_paths['']} has one expert")
            return
        if "low" not in dit_paths:
            raise ValueError(f"{dit_paths['']} is one expert of a two-expert "
                             "transformer: give the other with --dit-low")
        low = Path(dit_paths["low"]).resolve()
        if low == Path(dit_paths[""]).resolve():
            raise ValueError("--dit and --dit-low must be different checkpoints")
        if not low.is_file():
            raise FileNotFoundError(f"low-noise DiT checkpoint not found: {low}")

    def _lora_key_map(self, key: str) -> str:
        """Map native ComfyUI Wan LoRA projection names to our DiT module paths."""
        for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha"):
            if not key.endswith(suffix):
                continue
            stem = key[:-len(suffix)]
            if stem.startswith("lora_unet_"):
                raw = stem.removeprefix("lora_unet_")
                match = re.fullmatch(r"blocks_(\d+)_(self_attn|cross_attn)_(q|k|v|o)", raw)
                ffn = re.fullmatch(r"blocks_(\d+)_ffn_(0|2)", raw)
                if match:
                    stem = f"blocks.{match[1]}.{match[2]}.{match[3]}"
                elif ffn:
                    stem = f"blocks.{ffn[1]}.ffn.{ffn[2]}"
                else:
                    return key  # unknown training schema: never guess a projection
            stem = (stem.removeprefix("transformer.")
                    .removeprefix("model.diffusion_model.")
                    .removeprefix("diffusion_model."))
            return wan_dit_key(stem + ".weight").removesuffix(".weight") + suffix
        return key

    def __init__(self, *, config):
        super().__init__(config=config)
        self.params: Wan22Params = detect_params(self.dit_path)
        self.variant = variant_for(self.params)
        # `i2v` is the one capability a variant, rather than the model, decides.
        self.CAPABILITIES = {**self.CAPABILITIES, "i2v": self.variant.i2v}
        self.transformer = None
        self.low_transformer = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None

    def load_weights(self) -> None:
        self._load(Cancellation())

    @property
    def default_prefs(self) -> dict[str, Any]:
        """The class defaults under this variant's, which is what a request resolves to."""
        return {**self.DEFAULT_PREFS, **self.variant.prefs}

    def resolve_video(self, spec: VideoSpec) -> ResolvedVideo:
        """Apply this variant's defaults and geometric restrictions."""
        resolved = super().resolve_video(spec)
        if resolved.sampler not in ("euler", "unipc"):
            raise ValueError("Wan 2.2 supports only euler or unipc samplers")
        if resolved.image is not None and not self.variant.i2v:
            raise ValueError("Wan 2.2 T2V-A14B does not support I2V (requires CLIP vision)")
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
        vae = load_wan_native_video_vae(self.vae_path, self._load_target(self.vae_path),
                                        wan21=self.variant.wan21_vae)
        token.check()
        text_encoder, tokenizer = load_wan_text_encoder(
            self.text_encoder_path, self._load_target(self.text_encoder_path))
        token.check()
        transformer = load_wan_native_transformer(self.dit_path, self._load_target(self.dit_path))
        low = None
        if self.variant.experts is not None:
            token.check()
            low_path = self.dit_paths["low"]
            low = load_wan_native_transformer(low_path, self._load_target(low_path))
        token.check()
        self.memory.register("vae", vae)
        self.memory.register("text_encoder", text_encoder)
        self.memory.register("dit", transformer)
        if low is not None:
            self.memory.register("dit_low", low)
        self.vae, self.text_encoder, self.tokenizer = vae, text_encoder, tokenizer
        self.transformer, self.low_transformer = transformer, low
        log.info("Wan 2.2 %s loaded (%s)", self.variant.name,
                 "two experts" if low is not None else "one transformer")

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
        patch_t, patch_h, patch_w = self.transformer.params.patch_size
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

    def _frames(self, latents: torch.Tensor, token: Cancellation, progress):
        """One PIL frame at a time, pulled from the decoder as the muxer wants it."""
        chunks = latents.shape[2]
        decoded = 0

        def before(_module, _args):
            token.check()

        progress("decoding", 0, chunks)
        with ExitStack() as hooks, tqdm(total=chunks, desc="decoding") as bar:
            def after(_module, _args, _result):
                nonlocal decoded
                decoded += 1
                bar.update(1)
                progress("decoding", decoded, chunks)

            hooks.callback(self.vae.decoder.register_forward_pre_hook(before).remove)
            hooks.callback(self.vae.decoder.register_forward_hook(after).remove)
            for video in self.vae.decode_frames(self._denormalize(latents)):
                for i in range(video.shape[2]):
                    token.check()
                    # uint8 is a finer step than a BF16 mantissa.
                    yield pixels_to_pil(video[0, :, i].float())

    def _render(self, latents, spec: ResolvedVideo, token: Cancellation, progress,
                output: Path) -> None:
        """Decode straight into the container: the clip never exists all at once.

        Stays inside the ``vae`` stage because the generator pulls from the VAE as the
        encoder drains it — which is also what lets one frame, rather than the clip,
        size the stage's headroom.
        """
        progress("encoding", 0, spec.frame_count)
        encode_video(self._frames(latents, token, progress), spec, token, output, progress)

    def generate(self, spec: ResolvedVideo, token: Cancellation, progress, output: Path) -> None:
        from thenoise.inference import inference, inference_lock

        # One frame, not the clip: `_render` hands frames to the encoder as they are
        # decoded, so nothing in the run grows with `frame_count`. Whether a long clip
        # still wants more headroom than one frame is a measurement, not a correctness
        # question.
        pixels = spec.width * spec.height
        with inference_lock, inference():
            if spec.lora_specs and not self.lora_dir:
                raise ValueError("video LoRAs require --lora-dir")
            if self._lora_state_unfit:
                self.unload()  # a partially merged LoRA must never be sampled again
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

            spatial_factor = getattr(self.vae.config, "scale_factor_spatial", 16)
            shape = (1, self.vae.config.z_dim, (spec.frame_count - 1) // 4 + 1,
                     spec.height // spatial_factor, spec.width // spatial_factor)
            rng = torch.Generator(device="cpu").manual_seed(spec.seed)
            latents = torch.randn(shape, generator=rng, dtype=torch.float32).to(self.device)
            latents = self._run_denoiser(latents, spec, positive, negative,
                                         condition, token, progress, pixels)
            token.check()
            del positive, negative, condition

            with self.memory.stage("vae", pixels=pixels) as stage:
                stage.run(self._render, latents, spec, token, progress, output)
            del latents

        log.info("Wan video saved: %s", output)

    def _run_denoiser(self, latents, spec, positive, negative, condition,
                      token, progress, pixels):
        experts = self.variant.experts
        if experts is None:
            with self.memory.stage("dit", pixels=pixels) as stage:
                self.switch_loras(spec.lora_specs, self.transformer)
                return stage.run(self._sample, latents, spec, positive, negative,
                                 condition, token, progress)
        if condition is not None or spec.image is not None:
            raise ValueError("Wan 2.2 T2V-A14B does not support image conditioning")
        if spec.lora_specs:
            raise ValueError("Wan 2.2 T2V-A14B expert-specific LoRAs are not supported")
        return self._run_experts(latents, spec, positive, negative, token, progress,
                                 pixels, experts)

    def _run_experts(self, latents, spec, positive, negative, token, progress, pixels,
                     experts: Experts) -> torch.Tensor:
        """Denoise each half of the sigma range on the expert trained for it."""
        sigmas = get_sigmas(spec.steps, shift=experts.flow_shift)
        start = 0
        while start < spec.steps:
            high = sigmas[start] > experts.boundary
            end = start + 1
            while end < spec.steps and (sigmas[end] > experts.boundary) == high:
                end += 1
            model = self.transformer if high else self.low_transformer
            guidance = spec.guidance_scale * (1.0 if high else experts.low_cfg_ratio)
            stage_name = "dit" if high else "dit_low"
            ts = torch.tensor(sigmas[start:end + 1], device=latents.device,
                              dtype=torch.float32)
            schedule = [Step(t=ts[i], delta=ts[i] - ts[i + 1])
                        for i in range(len(ts) - 1)]

            def predict(x: torch.Tensor, sigma: float) -> torch.Tensor:
                token.check()
                x = x.to(model.dtype)
                timestep = torch.full((x.shape[0],), sigma * 1000.0,
                                      device=x.device, dtype=torch.float32)
                cond_vel = model(x, timestep=timestep, encoder_hidden_states=positive,
                                 return_dict=False)[0].float()
                if negative is None:
                    return cond_vel
                token.check()
                uncond_vel = model(x, timestep=timestep, encoder_hidden_states=negative,
                                   return_dict=False)[0].float()
                return uncond_vel + guidance * (cond_vel - uncond_vel)

            def step_done(x, i, _total):
                progress("sampling", start + i, spec.steps)
                token.check()
                return x

            if not high:
                self.memory.offload("dit")  # release the first expert before swapping
            with self.memory.stage(stage_name, pixels=pixels) as stage:
                latents = stage.run(
                    create_sampler(spec.sampler, _WanVelocity(predict)).sample,
                    latents, schedule, None, guidance, spec.seed,
                    step_callback=step_done,
                )
            start = end
        return latents

    def _clear_weights(self) -> None:
        self.vae = self.text_encoder = self.tokenizer = None
        self.transformer = self.low_transformer = None
