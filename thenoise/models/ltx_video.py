"""LTX 2.3/2.5 audiovisual video adapters for TheNoise's shared controller.

The controller owns execution, stages, LoRAs, cancellation, progress and encoding.
This module owns only the LTX kernels and the low-resolution -> refine transition.
All LTX numerical modules are native PyTorch and use TheNoise's existing
checkpoint loader, attention, quantization and memory-management seams.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from thenoise.dit.ltx_video.checkpoint import (
    VERSION_23, VERSION_25, load_original_components,
)
from thenoise.dit.ltx_video.sampling import REFINE_GRID, stage_grids
from thenoise.models.config import EncodePromptArgs, VideoSamplingParams
from thenoise.models.video_base import Segment, VideoModel, Waveform
from thenoise.samplers import Step, sigma_steps
from thenoise.text_encoders.ltx_video import load_ltx_gemma
from thenoise.utils.image_tensor import cover_crop
from thenoise.vae.ltx_audio import LATENT_TOKENS_PER_SECOND
from thenoise.video.spec import ResolvedVideo, VideoSpec

logger = logging.getLogger(__name__)


def frame_grid(requested: float, *, explicit: bool) -> int:
    """Causal first frame followed by groups of eight; explicit counts round up."""
    q = (requested - 1) / 8
    return max(9, 8 * (math.ceil(q) if explicit else math.floor(q + 0.5)) + 1)


def _pack(x: torch.Tensor) -> torch.Tensor:
    """AV video tokens [B,C,T,H,W] -> [B,T*H*W,C]."""
    return x.permute(0, 2, 3, 4, 1).flatten(1, 3)


def _unpack(x: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    return x.reshape(x.shape[0], frames, height, width, -1).permute(0, 4, 1, 2, 3)


def _normal_video(x, vae):
    mean = vae.latents_mean.to(x.device, x.dtype).reshape(1, -1, 1, 1, 1)
    std = vae.latents_std.to(x.device, x.dtype).reshape(1, -1, 1, 1, 1)
    return (x - mean) * vae.config.scaling_factor / std


def _denormal_video(x, vae):
    mean = vae.latents_mean.to(x.device, x.dtype).reshape(1, -1, 1, 1, 1)
    std = vae.latents_std.to(x.device, x.dtype).reshape(1, -1, 1, 1, 1)
    return x * std / vae.config.scaling_factor + mean


@dataclass(frozen=True)
class AVConditioning:
    """Connector features for both streams, per the shared ``Conditioning`` convention:
    the ``null_*`` halves are ``None`` when guidance is off.
    """

    video: torch.Tensor
    audio: torch.Tensor
    mask: torch.Tensor
    null_video: torch.Tensor | None = None
    null_audio: torch.Tensor | None = None
    null_mask: torch.Tensor | None = None


class _PromptTower(nn.Module):
    """One TheNoise text-encoder residency role for Gemma and its LTX connectors."""

    def __init__(self, gemma: nn.Module, connectors: nn.Module):
        super().__init__()
        self.gemma = gemma
        self.connectors = connectors

    def forward(self, ids, mask):
        hidden = self.gemma(ids, mask)
        hidden = hidden.to(self.connectors.dtype)
        return self.connectors(hidden, mask, padding_side="left")


class LTXVideo(VideoModel):
    """A shared LTX2.x kernel adapter with explicit 2.3/2.5 checkpoint differences."""

    version: ClassVar[str]
    COMPONENT_ROLES: ClassVar[tuple[str, ...]] = ("audio_vae", "upscaler",
                                                  "text_projection")
    CAPABILITIES = {**VideoModel.CAPABILITIES, "upscale": False,
                    "video": {"t2v": True, "i2v": True, "flf2v": False}}
    DEFAULT_PREFS = {
        **VideoModel.DEFAULT_PREFS,
        "width": 768, "height": 512, "frames": 33, "fps": 24,
        "steps": 8, "guidance_scale": 1.0, "flow_shift": 1.0,
        "sampler": "euler", "output_format": "mp4",
        "ref_method": "index_timestep_zero",
    }

    @classmethod
    def detect(cls, source) -> bool:
        # Match the shared, keys-only model detection contract.
        keys = set(source.names)
        is_av = (any(k.endswith("patchify_proj.weight") for k in keys) and
                 any(k.endswith("audio_patchify_proj.weight") for k in keys))
        has_keyframe = any(k.endswith("keyframes_abs_pos_embedding") for k in keys)
        return is_av and (has_keyframe == (cls.version == VERSION_25))

    @classmethod
    def validate_paths(cls, checkpoints: dict[str, object]) -> None:
        if cls.version == VERSION_25 and "text_projection" in checkpoints:
            raise ValueError("LTX 2.5 keeps its text projections with --text-encoder")

    def __init__(self, *, config):
        super().__init__(config=config)
        self._ref_low = None
        self._ref_high = None
        self._refine = False
        self._params: VideoSamplingParams | None = None

        gemma, self.tokenizer, text_cfg = load_ltx_gemma(
            self.text_encoder_path, version=self.version, cancel=lambda: None,
        )
        components, sources = load_original_components(
            {"": self.dit_path, "vae": self.vae_path, **config.extra_paths,
             "text_encoder": self.text_encoder_path},
            version=self.version, gemma_text_config=text_cfg,
            cancel=lambda: None,
            header_cache={Path(self.dit_path).expanduser().resolve(): self.header},
        )
        # Connectors are part of the text-encoder residency role, not the DiT.
        self.dit = components["dit"].to(self._load_target(str(sources["dit"])))
        self.vae = components["vae"].to(self._load_target(str(sources["vae"])))
        self.audio_vae = components["audio_vae"].to(
            self._load_target(str(sources["audio_vae"])))
        self.vocoder = components["vocoder"]
        self.upscaler = components["upscaler"].to(
            self._load_target(str(sources["upscaler"])))
        self.text_encoder = _PromptTower(gemma, components["connectors"])
        self.text_encoder.to(self._load_target(self.text_encoder_path))
        self.vocoder.to(self._load_target(str(sources["vocoder"])))
        for role in ("vae", "audio_vae", "vocoder", "upscaler", "text_encoder"):
            self.memory.register(role, getattr(self, role))
        self.memory.register("dit", self.dit)
        logger.info("LTX %s loaded with shared MemoryManager residency", self.version)

    def _create_upscaler(self):
        raise NotImplementedError("LTX uses an internal AV refinement stage")

    def lora_components(self) -> dict[str, nn.Module]:
        # Not enabled/tested for LTX: let the shared LoRA role checker reject it.
        return {}

    def _clear_weights(self):
        self.dit = self.vae = self.text_encoder = self.tokenizer = None
        self.audio_vae = self.vocoder = self.upscaler = None
        self.clear_video_state()

    def clear_video_state(self) -> None:
        """Drop reference latents on normal completion and on every failure/cancel."""
        self._ref_low = self._ref_high = None
        self._params = None
        self._refine = False

    def resolve_size(self, width, height):
        width, height = (width // 64) * 64, (height // 64) * 64
        if width < 64 or height < 64:
            raise ValueError("LTX requires output width and height >= 64 pixels")
        return width, height

    def default_edit_size(self, width, height):
        area = self.DEFAULT_PREFS["width"] * self.DEFAULT_PREFS["height"]
        scale = math.sqrt(area / (width * height))
        return round(width * scale), round(height * scale)

    def resolve_video(self, spec: VideoSpec) -> ResolvedVideo:
        r = super().resolve_video(spec)
        # Step count, guidance and solver are the request's: the released recipe is
        # DEFAULT_PREFS, and the schedule resamples the published grid to whatever else
        # is asked for. Mode/LoRA support is validated by the shared controller.
        return dataclasses.replace(r, frame_count=frame_grid(r.frame_count,
                                                            explicit=r.explicit_frames))

    def _tokenize(self, prompts: list[str]):
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("LTX Gemma tokenizer has no BOS token")
        rows = []
        for prompt in prompts:
            encoded = self.tokenizer(
                prompt.strip(), padding=False, truncation=True,
                max_length=1024, return_tensors="pt",
            )
            ids = encoded.input_ids[0].tolist()
            if not ids or ids[0] != bos_id:
                ids = [bos_id, *ids][:1024]
            rows.append(ids)
        padded = self.tokenizer.pad(
            {"input_ids": rows}, padding="max_length", max_length=1024,
            return_tensors="pt", return_attention_mask=True,
        )
        return padded.input_ids.to(self.device), padded.attention_mask.to(self.device)

    def encode_prompt(self, args: EncodePromptArgs) -> AVConditioning:
        """Both connector streams, and the unconditional halves guidance needs.

        One encoder pass carries the conditional and unconditional prompts as a batch of
        two, which is how the reference pipeline runs them.
        """
        need_null = args.guidance_scale > 1.0 or bool(args.negative_prompt.strip())
        ids, mask = self._tokenize(
            [args.prompt, args.negative_prompt] if need_null else [args.prompt])
        v, a, m = self.text_encoder(ids, mask)
        if not need_null:
            return AVConditioning(v[0:1], a[0:1], m[0:1])
        return AVConditioning(v[0:1], a[0:1], m[0:1], v[1:2], a[1:2], m[1:2])

    def fit_references(self, images, target):
        if len(images) != 1:
            raise ValueError("LTX I2V accepts exactly one starting image")
        return [cover_crop(images[0], target[0], target[1])]

    def encode_references(self, pixels, frames=1):
        if len(pixels) != 1:
            raise ValueError("LTX I2V accepts exactly one starting image")
        # One frame, encoded twice: at full resolution for the refine pass' grid, and at
        # half resolution for the grid the first half of the clip runs on. The resample
        # stays in fp32; the codec then runs in its own dtype, as on the decode side.
        frame = pixels[0][None, :, None].to(self.device)
        small = F.interpolate(frame[:, :, 0], scale_factor=0.5,
                              mode="bilinear", align_corners=False)[:, :, None]
        full, small = frame.to(self.vae.dtype), small.to(self.vae.dtype)
        with torch.no_grad():
            low = self.vae.encode(small).latent_dist.mode()
            high = self.vae.encode(full).latent_dist.mode()
        return [_normal_video(low, self.vae), _normal_video(high, self.vae)]

    def _geometry(self, params: VideoSamplingParams, high: bool):
        return ((params.frames - 1) // 8 + 1,
                params.height // (32 if high else 64),
                params.width // (32 if high else 64))

    def _video_tokens(self, params, high):
        t, h, w = self._geometry(params, high)
        return t * h * w

    def init_latents(self, params: VideoSamplingParams):
        self._params = params
        t, h, w = self._geometry(params, False)
        audio_length = max(1, round(params.frames / params.fps * LATENT_TOKENS_PER_SECOND))
        seed = torch.Generator(device="cpu").manual_seed(params.seed)
        video = torch.randn((1, 128, t, h, w), generator=seed, dtype=torch.float32)
        audio = torch.randn((1, audio_length, 128), generator=seed, dtype=torch.float32)
        return torch.cat((_pack(video), audio), dim=1).to(self.device)

    def prepare_latent(self, latents, cond, params, ref=None, ref_method="index"):
        self._ref_low = self._ref_high = None
        if ref:
            if ref_method != "index_timestep_zero" or len(ref) != 2:
                raise ValueError("LTX I2V requires one image and index_timestep_zero")
            self._ref_low, self._ref_high = ref
            spatial_tokens = self._geometry(params, False)[1] * self._geometry(params, False)[2]
            if self._ref_low.shape != (1, 128, 1, *self._geometry(params, False)[1:]):
                raise ValueError("LTX I2V low-frame latent geometry mismatch")
            latents = latents.clone()
            latents[:, :spatial_tokens] = _pack(self._ref_low)
        return latents

    def schedule(self, params: VideoSamplingParams) -> list[Step]:
        """The main stage's ladder, which is what a request's ``steps`` counts.

        The refine pass belongs to the release's recipe rather than to the request, so it
        is a second segment and not extra entries here.
        """
        main, _ = stage_grids(params.steps)
        return sigma_steps(main, self.device, torch.float32)

    def segments(self, params, schedule, guidance_scale):
        _, refine = stage_grids(params.steps)
        return [
            Segment("dit", self.dit, tuple(schedule), guidance_scale),
            Segment("dit", self.dit,
                    tuple(sigma_steps(refine, self.device, torch.float32)),
                    guidance_scale),
        ]

    def begin_segment(self, segment: Segment) -> None:
        super().begin_segment(segment)
        # The main half always starts at pure noise; the refine half starts at the
        # re-noise level. Counting segments instead would break when ``stages.run``
        # replays its prepare callable after a memory squeeze.
        self._refine = float(segment.steps[0].t) < 1.0

    def transition_segment(self, x, previous: Segment, next_segment: Segment,
                           params: VideoSamplingParams, token):
        """At the stage boundary upsample the video latents and re-noise AV jointly."""
        token.check()
        t, h, w = self._geometry(params, False)
        video = _unpack(x[:, :t*h*w], t, h, w)
        audio = x[:, t*h*w:]
        with self.memory.stage("upscaler", pixels=params.width * params.height) as st:
            raw = _denormal_video(video, self.vae).to(self.upscaler.dtype)
            upscale = st.run(self.upscaler, raw)
        upscale = _normal_video(upscale, self.vae).float()
        t1, h1, w1 = self._geometry(params, True)
        if upscale.shape != (1, 128, t1, h1, w1):
            raise ValueError(f"LTX upscaler returned {tuple(upscale.shape)}, expected {(1,128,t1,h1,w1)}")
        sigma = REFINE_GRID[0]
        gen = torch.Generator(device="cpu").manual_seed(params.seed + 5)
        vnoise = torch.randn(upscale.shape, generator=gen, dtype=torch.float32).to(self.device)
        anoise = torch.randn(audio.shape, generator=gen, dtype=torch.float32).to(self.device)
        v = (1 - sigma) * upscale + sigma * vnoise
        a = (1 - sigma) * audio + sigma * anoise
        if self._ref_high is not None:
            v[:, :, :1] = self._ref_high
        token.check()
        return torch.cat((_pack(v), a), dim=1)

    def denoise_step(self, latents, t, cond: AVConditioning, guidance_scale: float, i):
        """One step of the joined audio+video latent.

        Guidance is the shared two-forward CFG, one scale over both streams. The released
        distilled checkpoints run with it off; upstream's separate audio scale,
        ``guidance_rescale``, STG and modality-isolation guidance are not implemented here
        (see ``docs/models/ltx.md``).
        """
        vel = self._velocity(latents, t, cond.video, cond.audio, cond.mask)
        if cond.null_video is None:
            return vel
        uncond = self._velocity(latents, t,
                                cond.null_video, cond.null_audio, cond.null_mask)
        return uncond + guidance_scale * (vel - uncond)

    def _velocity(self, latents, t, video_cond, audio_cond, cond_mask) -> torch.Tensor:
        params = self._params
        if params is None:
            raise RuntimeError("LTX denoise_step requires initialized clip parameters")
        lt, h, w = self._geometry(params, self._refine)
        nv = lt * h * w
        video = latents[:, :nv]
        audio = latents[:, nv:]
        if video.shape[1] != nv or audio.shape[-1] != 128:
            raise ValueError("LTX video/audio latent geometry mismatch")
        scaled_t = t.reshape(1).to(device=self.device, dtype=torch.float32) * float(
            self.dit.config.timestep_scale_multiplier
        )
        vtime = scaled_t
        mask = None
        ref = self._ref_high if self._refine else self._ref_low
        if ref is not None:
            mask = torch.zeros((1, nv), device=self.device, dtype=torch.float32)
            mask[:, :h*w] = 1.0
            video = video.clone()
            video[:, :h*w] = _pack(ref).to(video.dtype)
            vtime = scaled_t[:, None] * (1 - mask)
        dtype = self.dit.dtype
        pv, pa = self.dit(
            hidden_states=video.to(dtype),
            audio_hidden_states=audio.to(dtype),
            encoder_hidden_states=video_cond,
            audio_encoder_hidden_states=audio_cond,
            encoder_attention_mask=cond_mask,
            audio_encoder_attention_mask=cond_mask,
            timestep=vtime,
            audio_timestep=scaled_t,
            sigma=scaled_t,
            audio_sigma=scaled_t,
            num_frames=lt, height=h, width=w, fps=params.fps,
            audio_num_frames=audio.shape[1],
            video_keyframes_mask=None,
            use_cross_timestep=True, return_dict=False,
        )
        if mask is not None:
            pv = pv.float() * (1 - mask)[:, :, None]
        return torch.cat((pv.float(), pa.float()), dim=1)

    def finalize_latent(self, latents, params):
        self._ref_low = self._ref_high = None
        return latents

    def audio_components(self) -> tuple[str, ...]:
        return ("audio_vae", "vocoder")

    def decode_audio(self, latents):
        params = self._params
        if params is None:
            raise RuntimeError("LTX audio decoding has no clip parameters")
        nv = self._video_tokens(params, True)
        a = latents[:, nv:]
        mean = self.audio_vae.latents_mean.to(a.device, a.dtype)
        std = self.audio_vae.latents_std.to(a.device, a.dtype)
        a = (a * std + mean).unflatten(2, (8, 16)).transpose(1, 2)
        mel = self.audio_vae.decode(a.to(self.audio_vae.dtype), return_dict=False)[0]
        sound = self.vocoder(mel.to(self.vocoder.dtype))
        return Waveform(sound, self.vocoder.output_sampling_rate)

    def decode_frames(self, latents) -> Iterator[torch.Tensor]:
        params = self._params
        if params is None:
            raise RuntimeError("LTX video decoding has no clip parameters")
        t, h, w = self._geometry(params, True)
        video = _unpack(latents[:, :t*h*w], t, h, w)
        video = _denormal_video(video, self.vae).to(self.vae.dtype)
        pixels = self.vae.decode(video, None, return_dict=False)[0]
        for i in range(pixels.shape[2]):
            # VideoPipelineController converts these [-1,1] pixels once, not twice.
            yield pixels[0, :, i].float().clamp(-1, 1)


class LTX23Video(LTXVideo):
    name = "LTX-2.3-Distilled"
    version = VERSION_23
    #: Its codecs, vocoder and projections ship inside the transformer file -- in BF16
    #: and in the quantised releases that pack the same way.
    REQUIRED_COMPONENTS = ("text_encoder",)


class LTX25Video(LTXVideo):
    name = "LTX-2.5-Distilled"
    version = VERSION_25
    #: No all-in-one 2.5 release: its codecs and upscaler ship apart.
    REQUIRED_COMPONENTS = ("vae", "text_encoder")
