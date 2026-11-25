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
from typing import ClassVar, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from thenoise.dit.ltx_video.checkpoint import (
    VERSION_23, VERSION_25, load_original_components,
)
from thenoise.dit.ltx_video.keys import SIGNATURE_23, SIGNATURE_25
from thenoise.dit.ltx_video.sampling import REFINE_GRID, stage_grids
from thenoise.models.config import EncodePromptArgs, VideoSamplingParams
from thenoise.models.video_base import Segment, VideoModel, Waveform
from thenoise.samplers import Step, sigma_steps
from thenoise.text_encoders.ltx_video import PROMPT_CONTEXT, load_ltx_gemma
from thenoise.utils.image_tensor import cover_crop
from thenoise.vae.ltx_audio import LATENT_TOKENS_PER_SECOND
from thenoise.vae.ltx_video import TEMPORAL_COMPRESSION
from thenoise.video.spec import ResolvedVideo, VideoSpec, frame_grid

logger = logging.getLogger(__name__)

#: The spatial upscaler's gain, which is why a request's size is what the second stage
#: renders and the first stage renders half of it.
UPSCALER_FACTOR = 2


@dataclasses.dataclass(frozen=True)
class Frame:
    """One condition frame at each of the two stages' grids.

    Both are written together because a refine pass handed the frame its own resolution
    has never seen is a silent wrong answer.
    """

    low: torch.Tensor | None = None
    high: torch.Tensor | None = None


@dataclasses.dataclass(frozen=True)
class Clip:
    """The frames a request pins the clip to: its first, and optionally its last.

    One object rather than loose attributes because both ends are written and dropped
    together, and a clip left holding the previous clip's frame is a silent wrong answer.
    """

    start: Frame = Frame()
    end: Frame = Frame()


def _pack(x: torch.Tensor) -> torch.Tensor:
    """AV video tokens [B,C,T,H,W] -> [B,T*H*W,C]."""
    return x.permute(0, 2, 3, 4, 1).flatten(1, 3)


def _unpack(x: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    return x.reshape(x.shape[0], frames, height, width, -1).permute(0, 4, 1, 2, 3)


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

    def forward(self, ids: torch.Tensor):
        """Gemma at the prompt's own length, then the connectors' fixed context.

        That width is a property of their learned registers, not of the tokenizer, so it
        is filled here: one copy of the features, instead of 48 layers of attention over
        a thousand positions of which two dozen are the prompt.
        """
        hidden = self.gemma(ids).to(self.connectors.dtype)
        pad = PROMPT_CONTEXT - hidden.shape[1]
        if pad < 0:
            raise ValueError(f"Gemma produced {hidden.shape[1]} tokens for a "
                             f"{PROMPT_CONTEXT}-wide connector context")
        hidden = F.pad(hidden, (0, 0, pad, 0))
        positions = torch.arange(PROMPT_CONTEXT, device=ids.device)
        mask = (positions >= pad).to(torch.long).expand(hidden.shape[0], -1)
        return self.connectors(hidden, mask)


class LTXVideo(VideoModel):
    """A shared LTX2.x kernel adapter with explicit 2.3/2.5 checkpoint differences."""

    version: ClassVar[str]
    COMPONENT_ROLES: ClassVar[tuple[str, ...]] = ("audio_vae", "upscaler",
                                                  "text_projection")
    CAPABILITIES = {**VideoModel.CAPABILITIES, "upscale": False,
                    "video": {"t2v": True, "i2v": True, "flf2v": True,
                                  "ref2v": False}}
    DEFAULT_PREFS = {
        **VideoModel.DEFAULT_PREFS,
        "width": 768, "height": 512, "frames": 33, "fps": 24,
        "steps": 8, "guidance_scale": 1.0, "flow_shift": 1.0,
        "sampler": "euler", "ref_method": "index_timestep_zero",
    }

    @classmethod
    def validate_paths(cls, checkpoints: dict[str, object]) -> None:
        for header in checkpoints.values():
            stated = str(header.metadata.get("model_version", "")
                         or header.metadata_json("config").get("model_version", ""))
            if stated and not stated.startswith(cls.version):
                raise ValueError(f"{header.path} states model_version {stated}, "
                                 f"which {cls.name} is not")
        if cls.version == VERSION_25 and "text_projection" in checkpoints:
            raise ValueError("LTX 2.5 keeps its text projections with --text-encoder")

    def __init__(self, *, config):
        super().__init__(config=config)
        self._clip = Clip()

        gemma, self.tokenizer, text_cfg = load_ltx_gemma(
            self.text_encoder_path, version=self.version,
            device=self._load_target(self.text_encoder_path),
        )
        components, _ = load_original_components(
            {"": self.dit_path, "vae": self.vae_path, **config.extra_paths,
             "text_encoder": self.text_encoder_path},
            version=self.version, gemma_text_config=text_cfg,
            headers=config.all_headers(),
            device_for=self.memory.device_for,
        )
        self.dit = components["dit"]
        self.vae = components["vae"]
        self.audio_vae = components["audio_vae"]
        self.vocoder = components["vocoder"]
        self.upscaler = components["upscaler"]
        self.text_encoder = _PromptTower(gemma, components["connectors"])
        for role in ("vae", "audio_vae", "vocoder", "upscaler", "text_encoder"):
            self.memory.register(role, getattr(self, role))
        self.memory.register("dit", self.dit)
        logger.info("LTX %s loaded with shared MemoryManager residency", self.version)

    def _create_upscaler(self):
        raise NotImplementedError("LTX uses an internal AV refinement stage")

    def _clear_weights(self):
        self.dit = self.vae = self.text_encoder = self.tokenizer = None
        self.audio_vae = self.vocoder = self.upscaler = None
        self.clear_video_state()

    def clear_video_state(self) -> None:
        """Drop reference latents on normal completion and on every failure/cancel."""
        self._clip = Clip()

    def resolve_size(self, width, height):
        # The grid the first stage renders on, which is the codec's compression twice
        # over because a low stage is half the delivered size.
        grid = self.vae.spatial_compression_ratio * UPSCALER_FACTOR
        width, height = (width // grid) * grid, (height // grid) * grid
        if width < grid or height < grid:
            raise ValueError(f"LTX requires output width and height >= {grid} pixels")
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
        return dataclasses.replace(r, frame_count=frame_grid(
            r.frame_count, compression=TEMPORAL_COMPRESSION, upward=r.explicit_frames))

    def encode_prompt(self, args: EncodePromptArgs) -> AVConditioning:
        """Both connector streams, and the unconditional halves guidance needs.

        Each prompt is encoded on its own. The released pipeline batches them, which
        means padding every prompt to the context width; alone, a prompt is 48 layers of
        attention over its own length and needs no mask at all. Two rows of two dozen
        tokens are not worth a wider kernel.
        """
        need_null = args.guidance_scale > 1.0 or bool(args.negative_prompt.strip())
        prompts = [args.prompt, args.negative_prompt] if need_null else [args.prompt]
        rows = [self.text_encoder(self.tokenizer.encode(prompt).to(self.device))
                for prompt in prompts]
        if not need_null:
            return AVConditioning(*rows[0])
        return AVConditioning(*rows[0], *rows[1])

    def fit_references(self, images, target):
        if len(images) > 2:
            raise ValueError("LTX pins a clip on its first frame and, at most, its last")
        return [cover_crop(image, target[0], target[1]) for image in images]

    def encode_references(self, pixels, frames=1):
        """Each endpoint twice, low resolution first: ``[start_low, start_high,
        end_low, end_high]``, which :meth:`prepare_latent` reads in pairs.

        Twice because the two stages sample on different grids: half the delivered picture
        before the spatial upscaler, all of it after. The resample is fp32 and the codec
        casts to its own dtype at its own boundary, as on decode.
        """
        encoded = []
        for image in pixels:
            frame = image[None, :, None].to(self.device)
            small = F.interpolate(frame[:, :, 0], scale_factor=0.5,
                                  mode="bilinear", align_corners=False)[:, :, None]
            with torch.no_grad():
                encoded += [self.vae.encode_pixels_to_latents(small),
                            self.vae.encode_pixels_to_latents(frame)]
        return encoded

    def _geometry(self, params: VideoSamplingParams, high: bool):
        """Latent (frames, height, width).

        The codec says how far it folds each way; the stage before the spatial upscaler
        divides by its gain as well, since it renders half the delivered picture.
        """
        frames = (params.frames - 1) // self.vae.temporal_compression_ratio + 1
        compress = self.vae.spatial_compression_ratio
        if not high:
            compress *= UPSCALER_FACTOR
        return frames, params.height // compress, params.width // compress

    def _video_tokens(self, params, high):
        t, h, w = self._geometry(params, high)
        return t * h * w

    def init_latents(self, params: VideoSamplingParams):
        t, h, w = self._geometry(params, False)
        audio_length = max(1, round(params.frames / params.fps * LATENT_TOKENS_PER_SECOND))
        dev = self.device
        seed = torch.Generator(device=dev).manual_seed(params.seed)
        video = torch.randn((1, self.vae.latent_channels, t, h, w),
                            generator=seed, device=dev, dtype=torch.float32)
        audio = torch.randn((1, audio_length, 128),
                            generator=seed, device=dev, dtype=torch.float32)
        return torch.cat((_pack(video), audio), dim=1)

    def prepare_latent(self, latents, cond, params, ref=None, ref_method="index"):
        """Overwrite the clip's endpoints once, and keep both resolutions for the boundary.

        A pinned frame is never denoised rather than re-injected every step: the per-token
        timestep :meth:`_velocity` takes from the same mask makes its velocity come back
        zero, so the solver leaves it where it was written.
        """
        clip = Clip()
        if ref:
            if ref_method != "index_timestep_zero":
                raise ValueError("LTX pins a condition frame at timestep zero; it has no"
                                 f" '{ref_method}' path")
            if len(ref) not in (2, 4):
                raise ValueError("LTX takes each condition frame as a low- and a "
                                 "high-resolution latent, in that order")
            clip = Clip(start=Frame(*ref[:2]),
                        end=Frame(*ref[2:]) if len(ref) > 2 else Frame())
            t, h, w = self._geometry(params, False)
            latents = latents.clone()
            for index, frame in ((0, clip.start), (t - 1, clip.end)):
                if frame.low is None:
                    continue
                if frame.low.shape != (1, self.vae.latent_channels, 1, h, w):
                    raise ValueError("LTX condition-frame latent geometry mismatch")
                latents[:, index * h * w:(index + 1) * h * w] = _pack(frame.low)
        self._clip = clip
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
            Segment("dit", self.dit, tuple(schedule), guidance_scale, params),
            Segment("dit", self.dit,
                    tuple(sigma_steps(refine, self.device, torch.float32)),
                    guidance_scale, params),
        ]

    def transition_segment(self, x, previous: Segment, next_segment: Segment,
                           params: VideoSamplingParams, token):
        """At the stage boundary upsample the video latents and re-noise AV jointly."""
        token.check()
        t, h, w = self._geometry(params, False)
        video = _unpack(x[:, :t*h*w], t, h, w)
        audio = x[:, t*h*w:]
        with self.memory.stage("upscaler", pixels=params.width * params.height) as st:
            raw = self.vae.from_canonical(video).to(self.upscaler.dtype)
            upscale = st.run(self.upscaler, raw)
        upscale = self.vae.to_canonical(upscale).float()
        t1, h1, w1 = self._geometry(params, True)
        want = (1, self.vae.latent_channels, t1, h1, w1)
        if upscale.shape != want:
            raise ValueError(f"LTX upscaler returned {tuple(upscale.shape)}, expected {want}")
        sigma = REFINE_GRID[0]
        dev = self.device
        gen = torch.Generator(device=dev).manual_seed(params.seed + 5)
        vnoise = torch.randn(upscale.shape, generator=gen, device=dev, dtype=torch.float32)
        anoise = torch.randn(audio.shape, generator=gen, device=dev, dtype=torch.float32)
        v = (1 - sigma) * upscale + sigma * vnoise
        a = (1 - sigma) * audio + sigma * anoise
        if self._clip.start.high is not None:
            v[:, :, :1] = self._clip.start.high
        if self._clip.end.high is not None:
            v[:, :, -1:] = self._clip.end.high
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
        segment = self._segment
        if segment is None:
            raise RuntimeError("LTX denoise_step is outside a segment")
        params = segment.params
        # Which half of the schedule a step sits in decides where the joined stream
        # splits, and the half is written on its first sigma: pure noise, or the
        # re-noise level the refine stage is entered at.
        refine = float(segment.steps[0].t) < 1.0
        lt, h, w = self._geometry(params, refine)
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
        pinned = [(index, frame.high if refine else frame.low)
                  for index, frame in ((0, self._clip.start), (lt - 1, self._clip.end))]
        pinned = [(index, ref) for index, ref in pinned if ref is not None]
        if pinned:
            mask = torch.zeros((1, nv), device=self.device, dtype=torch.float32)
            video = video.clone()
            for index, ref in pinned:
                mask[:, index*h*w:(index + 1)*h*w] = 1.0
                video[:, index*h*w:(index + 1)*h*w] = _pack(ref).to(video.dtype)
            vtime = scaled_t[:, None] * (1 - mask)
        # 2.5's learned marker names the latent frame carrying a single pixel frame -- the
        # causal codec's first -- which is what ComfyUI adds it to on every run. A pinned
        # ending sits inside a whole temporal window, so it is not one of those.
        keyframes = None
        if self.dit.config.use_keyframes_abs_pos_embedding:
            keyframes = torch.zeros((1, nv), device=self.device, dtype=torch.float32)
            keyframes[:, :h*w] = 1.0
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
            video_keyframes_mask=keyframes,
            use_cross_timestep=True,
        )
        if mask is not None:
            pv = pv.float() * (1 - mask)[:, :, None]
        return torch.cat((pv.float(), pa.float()), dim=1)

    def finalize_latent(self, latents, params):
        self._clip = Clip()
        return latents

    def audio_components(self) -> tuple[str, ...]:
        return ("audio_vae", "vocoder")

    def decode_audio(self, latents, params: VideoSamplingParams):
        nv = self._video_tokens(params, True)
        a = latents[:, nv:]
        mel = self.audio_vae.decode_to_mel(a)
        sound = self.vocoder(mel.to(self.vocoder.dtype))
        return Waveform(sound, self.vocoder.output_sampling_rate)

    def decoded_pixels(self, spec: ResolvedVideo) -> int:
        # The released decoder is not causal along time: every output frame depends on
        # every latent frame, so all of them are live before the first one is yielded.
        return spec.width * spec.height * spec.frame_count

    def decode_frames(self, latents,
                      params: VideoSamplingParams) -> Iterator[torch.Tensor]:
        t, h, w = self._geometry(params, True)
        video = _unpack(latents[:, :t*h*w], t, h, w)
        pixels = self.vae.decode_to_pixels(video)
        for i in range(pixels.shape[2]):
            # VideoPipelineController converts these [-1,1] pixels once, not twice.
            yield pixels[0, :, i].float().clamp(-1, 1)


class LTX23Video(LTXVideo):
    name = "ltx23"
    version = VERSION_23
    SIGNATURE = SIGNATURE_23
    #: Its codecs, vocoder and projections ship inside the transformer file -- in BF16
    #: and in the quantised releases that pack the same way.
    REQUIRED_COMPONENTS = ("text_encoder",)


class LTX25Video(LTXVideo):
    name = "ltx25"
    version = VERSION_25
    SIGNATURE = SIGNATURE_25
    #: No all-in-one 2.5 release: its codecs and upscaler ship apart.
    REQUIRED_COMPONENTS = ("vae", "text_encoder")
