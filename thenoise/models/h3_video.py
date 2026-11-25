"""H3 video adapter — one packed sequence carrying a clip and its sound.

Two streams, one solver. The picture runs the shifted-flow ladder its release was sampled
on (shift 12.0) and the sound runs its own (3.0), so a step of the video schedule is a
different amount of progress for each of them: the solver holds the audio as a *carry* —
:mod:`thenoise.dit.h3.sampling` states the affine map — and the two conversions live here,
at the edge where the stream enters and leaves the loop. Nothing about the solver knows
there are two modalities, which is what lets previews, stages and memory sizing stay as
they are for a silent model.

A keyframe is not written into the clip. It joins the sequence as extra conditioned rows
— the same spatial grid, one latent frame, read at the conditioned timestep — so a clip can
start and end on a picture without either frame being denoised or re-injected mid-run.

The two codecs keep their own clocks and the frame rate reconciles them: the picture's
length is whatever the video codec decodes back from its latent frames, and the sound's is
forty latent frames to the second, so a request's frame count and fps decide both shapes.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Tuple

import torch

from thenoise.dit.h3 import (
    H3_AV_SIGNATURE,
    H3Params,
    Layout,
    detect_params,
    load_h3_av_dit,
)
from thenoise.dit.h3 import sampling as h3_sampling
from thenoise.dit.h3.encoder import VIDEO_PICTURE_LABEL, VISION_SHORT_EDGE, load_h3_text_encoder
from thenoise.models.base import Conditioning
from thenoise.models.config import (
    EncodePromptArgs,
    ModelConfig,
    SamplingParams,
    VideoSamplingParams,
)
from thenoise.models.video_base import VideoModel, Waveform
from thenoise.samplers import Step
from thenoise.utils.image_tensor import cover_crop, resize_to_short_edge
from thenoise.utils.safetensors import Header
from thenoise.vae import load_h3_audio_vae, load_h3_video_vae
from thenoise.vae.h3 import TemporalPlan, snap_frames
from thenoise.video.spec import ResolvedVideo, VideoSpec

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image

logger = logging.getLogger(__name__)

#: Pixels a latent cell covers once the DiT has patched it: 16 by the codec, 2 again by
#: the model. A canvas off this grid would be padded on the way in and cropped on the way
#: out, which is a resample nobody asked for.
CANVAS_MULTIPLE = 32

#: The area the release was trained at (1344×768 at 24 fps). Bigger works — the rope
#: normalizes by area, so the lattice does not break — but it is off distribution.
TRAINING_AREA = 768 * 1344


@dataclass
class H3VideoConditioning(Conditioning):
    """``Conditioning`` plus the per-branch modality tags the DiT modulates by.

    Per-branch because the two prompts carry the keyframes at different offsets whenever
    the negative prompt is not empty.
    """

    tags: Optional[torch.Tensor] = None
    null_tags: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _Streams:
    """The run's two latents, and the one flat tensor the solver steps between them.

    One solver means one tensor, and the streams cannot share an axis: a video latent is a
    spatio-temporal lattice and an audio latent is a stereo pair of feature tracks. So the
    pack is flat and the split is a count, held here rather than repeated at each end.
    """

    video: Tuple[int, int, int, int]
    audio: Tuple[int, int, int]

    @property
    def video_rows(self) -> int:
        return math.prod(self.video)

    @property
    def audio_rows(self) -> int:
        return math.prod(self.audio)

    def split(self, flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.video_rows
        if flat.shape[1] != n + self.audio_rows:
            raise ValueError(
                f"the clip carries {self.video_rows} video and {self.audio_rows} audio "
                f"values, the latent has {flat.shape[1]}"
            )
        return (flat[:, :n].reshape((1, *self.video)),
                flat[:, n:].reshape((1, *self.audio)))

    def join(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return torch.cat((video.reshape(1, -1), audio.reshape(1, -1)), dim=1)


class H3VideoModel(VideoModel):
    """MiniMax H3 ``fl2va``: text-to-video with audio, optionally keyed on two frames."""

    name = "h3_video"

    REQUIRED_COMPONENTS = ("vae", "text_encoder")
    COMPONENT_ROLES = ("audio_vae",)

    DEFAULT_PREFS = {
        **VideoModel.DEFAULT_PREFS,
        "width": 1344, "height": 768, "frames": 124, "fps": 24,
        "steps": 20, "guidance_scale": 1.0, "flow_shift": 12.0,
        "sampler": "euler", "ref_method": "index_timestep_zero",
    }

    CAPABILITIES = {**VideoModel.CAPABILITIES, "upscale": False,
                    "video": {"t2v": True, "i2v": True, "flf2v": True}}

    #: A keyframe enters at a fixed timestep and is never denoised, which is what
    #: ``index_timestep_zero`` means here; that it arrives as extra rows rather than as an
    #: overwritten latent frame is the model's business, and ``prepare_latent`` is where it
    #: says so.
    REF_METHODS = ("index_timestep_zero",)

    SIGNATURE = H3_AV_SIGNATURE

    @classmethod
    def validate_paths(cls, checkpoints: dict[str, Header]) -> None:
        super().validate_paths(checkpoints)
        if "audio_vae" not in checkpoints:
            raise ValueError(
                f"{cls.name} decodes a sound as well as a picture, so it needs "
                "--audio-vae"
            )

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.params: H3Params = detect_params(self.header, self.SIGNATURE)
        self.dit = load_h3_av_dit(
            self.header, self.params,
            device=self._load_target(self.dit_path), dtype=config.dtype,
        )

        logger.info("Loading H3 video VAE from %s", self.vae_path)
        self.vae = load_h3_video_vae(self.vae_path, device=self._load_target(self.vae_path))

        audio_path = config.extra_paths["audio_vae"]
        logger.info("Loading H3 audio VAE from %s", audio_path)
        self.audio_vae = load_h3_audio_vae(audio_path,
                                           device=self._load_target(audio_path))
        if self.params.audio_dim != self.audio_vae.z_dim:
            raise ValueError(
                f"{audio_path} decodes a {self.audio_vae.z_dim}-channel audio latent "
                f"while this DiT emits {self.params.audio_dim}; the two codecs are not a "
                "pair"
            )

        logger.info("Loading H3 text encoder from %s", config.text_encoder_path)
        self.text_encoder = load_h3_text_encoder(
            config.text_encoder_path,
            dtype=config.dtype,
            device=self._load_target(config.text_encoder_path),
            picture_label=VIDEO_PICTURE_LABEL,
        )

        self.memory.register("dit", self.dit)
        self.memory.register("vae", self.vae)
        self.memory.register("audio_vae", self.audio_vae)
        self.memory.register("text_encoder", self.text_encoder)

        self._reset_run()
        logger.info("H3 video model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ request
    def resolve_size(self, width: int, height: int) -> Tuple[int, int]:
        """Rounded down onto the grid; the trained area is a warning, not a wall."""
        size = (width // CANVAS_MULTIPLE * CANVAS_MULTIPLE,
                height // CANVAS_MULTIPLE * CANVAS_MULTIPLE)
        if 0 in size:
            raise ValueError(f"H3 video requires width and height of at least {CANVAS_MULTIPLE}")
        if size[0] * size[1] > TRAINING_AREA:
            logger.warning(
                "%dx%d is more area than H3 video was trained on (%d); the rope "
                "normalizes by area, so this runs — it is just off distribution",
                size[0], size[1], TRAINING_AREA)
        return size

    def default_edit_size(self, width: int, height: int) -> Tuple[int, int]:
        """An unnamed keyed clip is sampled at its first frame's aspect, at the default area."""
        area = self.DEFAULT_PREFS["width"] * self.DEFAULT_PREFS["height"]
        scale = math.sqrt(area / (width * height))
        return round(width * scale), round(height * scale)

    def resolve_video(self, spec: VideoSpec) -> ResolvedVideo:
        resolved = super().resolve_video(spec)
        # Steps, guidance and solver stay the request's: DEFAULT_PREFS is the released
        # recipe and the schedule is rebuilt from whatever steps are asked for.
        return dataclasses.replace(
            resolved, frame_count=snap_frames(resolved.frame_count,
                                              upward=resolved.explicit_frames))

    def fit_references(self, images: Sequence["Image.Image"],
                       target: Tuple[int, int]) -> List["Image.Image"]:
        """Cover-crop, both frames alike.

        The release's node stretches its first frame to the canvas and crops its last,
        which differs from cropping both only when the request names a size whose aspect
        the picture disagrees with — and a keyframe that is stretched is a keyframe the
        clip will not start on.
        """
        if len(images) > 2:
            raise ValueError("H3 video keys a clip on its first frame and, at most, its last")
        return [cover_crop(image, target[0], target[1]) for image in images]

    def encode_references(self, pixels, frames=1) -> List[torch.Tensor]:
        """One latent per keyframe, in the order the prompt labels them.

        A keyframe is a single frame, which is what the codec's encoder answers for one
        frame: its posterior mean, the same answer every time. It is the *layout* that
        decides where in the clip's timeline the frame sits, not this call.
        """
        with torch.no_grad():
            return [self.vae.encode_pixels_to_latents(image[None, :, None])
                    for image in pixels]

    # ------------------------------------------------------------ conditioning
    def encode_prompt(self, args: EncodePromptArgs) -> H3VideoConditioning:
        """Prompt and keyframes through the language model, as raw conditioning.

        The negative branch sees the same pictures: a keyframe is what the clip is asked to
        start and end on, not something to steer away from.
        """
        images = [resize_to_short_edge(img, VISION_SHORT_EDGE, align=CANVAS_MULTIPLE)
                  for img in _as_list(args.image)]
        cond, cond_mask, tags = self.text_encoder(args.prompt, images)
        null = null_mask = null_tags = None
        if args.guidance_scale > 1.0:
            null, null_mask, null_tags = self.text_encoder(args.negative_prompt, images)
        return H3VideoConditioning(
            cond=cond, cond_mask=cond_mask, null=null, null_mask=null_mask,
            tags=tags, null_tags=null_tags,
        )

    def fuse_text(self, cond: Conditioning) -> H3VideoConditioning:
        """Project the prompt into the trunk once per run: it does not depend on the step."""
        dev = torch.device(self.device)
        return H3VideoConditioning(
            cond=self.dit.refine_text(cond.cond.to(dev, self.dtype)),
            cond_mask=cond.cond_mask,
            null=(None if cond.null is None
                  else self.dit.refine_text(cond.null.to(dev, self.dtype))),
            null_mask=cond.null_mask,
            tags=getattr(cond, "tags", None),
            null_tags=getattr(cond, "null_tags", None),
        )

    # ------------------------------------------------------------ the run
    def _streams(self, params: VideoSamplingParams) -> _Streams:
        """The two latent shapes a request of this length and size implies."""
        video_t = TemporalPlan.of_frames(params.frames).latent_t
        h = params.height // self.vae.spatial_compression
        w = params.width // self.vae.spatial_compression
        # The sound is as long as the picture is, counted in the audio codec's frames.
        audio_t = max(1, round(params.frames / params.fps * self.audio_vae.latents_per_second))
        return _Streams(video=(self.vae.z_dim, video_t, h, w),
                        audio=(self.audio_vae.z_dim, 2, audio_t))

    def init_latents(self, params: VideoSamplingParams) -> torch.Tensor:
        """Noise for both streams, as one flat row the solver steps as a unit.

        Neither stream needs pre-scaling to travel together: the carry is 1 at pure noise,
        which is all either of them is here.
        """
        dev = torch.device(self.device)
        streams = self._streams(params)
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        video = torch.randn(streams.video, generator=generator, device=dev, dtype=torch.float32)
        audio = torch.randn(streams.audio, generator=generator, device=dev, dtype=torch.float32)
        return streams.join(video, audio)

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: VideoSamplingParams,
        ref: Optional[List[torch.Tensor]] = None,
        ref_method: str = "index",
    ) -> torch.Tensor:
        """Build the packing this run steps through; the latent itself is untouched.

        The keyframes become rows of the same lattice at the timeline's ends, so they shift
        the target's coordinates ahead of them without ever being denoised themselves.
        """
        keyframes = _as_list(ref)
        if keyframes and ref_method != "index_timestep_zero":
            raise ValueError(
                "H3 video reads a keyframe at the conditioned timestep and never "
                f"denoises it; '{ref_method}' describes overwriting a frame of the clip"
            )
        dev = torch.device(self.device)
        streams = self._streams(params)
        # A keyframe sits on the clip's own timeline: the first frame opens it, the last
        # closes it, and a lone one is the opening frame.
        anchors = [0, params.frames - 1][:len(keyframes)]
        self._streams_ = streams
        self._shift = float(params.flow_shift)
        self._seed = params.seed
        self._keyframes = [k.to(dev, self.dtype) for k in keyframes]
        self._layout = Layout.av(cond.cond.shape[1], *streams.video[1:],
                                 audio_t=streams.audio[2], keyframes=anchors)
        self._null_layout = (
            None if cond.null is None
            else Layout.av(cond.null.shape[1], *streams.video[1:],
                           audio_t=streams.audio[2], keyframes=anchors)
        )
        self.dit.pe_embedder.clear()
        self.dit.prepare_rope(self._layout, self.dtype)
        if self._null_layout is not None:
            self.dit.prepare_rope(self._null_layout, self.dtype, name="uncond")
        return latents

    def begin_sampling(self, schedule: List[Step]) -> None:
        """This run's sigmas, which a PDD head bank averages over."""
        sigmas = [float(step.t) for step in schedule]
        if schedule:
            sigmas.append(float(schedule[-1].t) - float(schedule[-1].delta))
        self._sample_sigmas = torch.tensor(sigmas, dtype=torch.float32)

    def schedule(self, params: VideoSamplingParams) -> list[Step]:
        """The video stream's shifted ladder; the audio rides it, re-shifted per step."""
        ladder = h3_sampling.get_sigmas(params.steps, shift=params.flow_shift)
        return [Step(t=ladder[i], delta=ladder[i] - ladder[i + 1])
                for i in range(params.steps)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One packed forward over both streams, answering one velocity for the pack.

        ``t`` is the video sigma; the audio's is where the audio schedule puts it at that
        same progress. Both streams are read at their own sigmas and returned in the
        carried variable, which is the one the solver steps.
        """
        sigma = float(t)
        velocity = self._velocity(latents, sigma, cond.cond, self._layout,
                                  getattr(cond, "tags", None))
        if guidance_scale > 1.0 and cond.null is not None:
            neg = self._velocity(latents, sigma, cond.null, self._null_layout,
                                 getattr(cond, "null_tags", None), rope="uncond")
            return neg + guidance_scale * (velocity - neg)
        return velocity

    def _velocity(self, latents, sigma, context, layout, tags, rope="cond"):
        streams = self._streams_
        if streams is None:
            raise RuntimeError("H3 video denoise_step outside a prepared run")
        video, carried = streams.split(latents)
        own = h3_sampling.carry(carried, sigma, shift=self._shift)
        device = torch.device(self.device)
        video_velocity, audio_velocity = self.dit(
            video.to(device, self.dtype), own.to(device, self.dtype),
            sigma, h3_sampling.audio_sigma(sigma, shift=self._shift),
            context, layout, self._keyframes, tags=tags, seed=self._seed,
            sample_sigmas=self._sample_sigmas, rope=rope,
        )
        # The audio's velocity comes back in its own variable; the pack carries a scaled
        # version of it, and the carry's own slope belongs to the answer.
        return streams.join(video_velocity.float(),
                            h3_sampling.audio_velocity(audio_velocity, own, sigma=sigma,
                                                       shift=self._shift).float())

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        self.dit.pe_embedder.clear()
        return latents

    # ------------------------------------------------------------ codec
    def decoded_pixels(self, spec: ResolvedVideo) -> int:
        """One clip walk's worth of pixels, which is not one frame's.

        The decoder re-reads each clip's overlap and composites every frame of the clip it
        is walking before it hands one back, so the canvas is live times the frames of that
        clip however finely it is tiled in space — a stage sized by a single frame of the
        clip survives only by being retried.
        """
        plan = TemporalPlan.of(1)
        return spec.width * spec.height * (plan.chunk_tokens
                                           + plan.overlap_tokens) * plan.ratio_t

    def _create_upscaler(self):
        raise NotImplementedError(
            "no upscaler has been released for this codec; the canvas a request names is "
            "the canvas the clip is sampled and decoded on"
        )

    def audio_components(self) -> tuple:
        return ("audio_vae",)

    def decode_audio(self, latents: torch.Tensor,
                     params: VideoSamplingParams) -> Optional[Waveform]:
        """The pack's audio slice, un-carried, then through the vocoder.

        The carry ends at ``S``, the ratio of the two shifts, so the finished slice is
        ``S`` times the latent the codec was trained on — whatever the path there was.
        """
        _, carried = self._streams(params).split(latents)
        own = h3_sampling.carry(carried, 0.0, shift=params.flow_shift)
        pcm = self.audio_vae.decode_to_pcm(own.to(self.device))
        return Waveform(pcm[0].float(), self.audio_vae.sample_rate)

    def decode_frames(self, latents: torch.Tensor,
                      params: VideoSamplingParams) -> Iterator[torch.Tensor]:
        """The pack's video slice, decoded a clip at a time and handed out a frame at a time.

        The codec yields whole runs of frames and the container wants one at a time, and it
        has already clamped to the range the pipeline works in.
        """
        video, _ = self._streams(params).split(latents)
        for run in self.vae.decode_to_pixel_frames(video.to(self.device)):
            for index in range(run.shape[2]):
                yield run[0, :, index].float()

    # ------------------------------------------------------------ teardown
    def _reset_run(self) -> None:
        self._streams_ = None
        self._shift = float(self.DEFAULT_PREFS["flow_shift"])
        self._seed = 0
        self._keyframes: List[torch.Tensor] = []
        self._layout: Optional[Layout] = None
        self._null_layout: Optional[Layout] = None
        self._sample_sigmas = None

    def clear_video_state(self) -> None:
        self._reset_run()

    def _clear_weights(self) -> None:
        self.dit = self.vae = self.audio_vae = self.text_encoder = None


def _as_list(value) -> list:
    """``None`` / one item / a sequence of them -> a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


__all__ = ["H3VideoConditioning", "H3VideoModel"]
