"""H3 video adapter — one packed sequence carrying a clip and its sound.

Two streams, one solver, so the audio travels in a *carry* whose affine map lives in
:mod:`thenoise.dit.h3.sampling` and the two conversions in and out of it live here. The
codecs keep their own clocks: the picture's length is what the video codec decodes back from
its latent frames, the sound's forty latent frames to the second.
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
    Keyframe,
    Layout,
    pixel_frames,
    Reference,
    detect_params,
    load_h3_av_dit,
)
from thenoise.dit.h3 import sampling as h3_sampling
from thenoise.dit.h3.layout import FRAME_RESCALE
from thenoise.dit.h3.encoder import (
    GuideMedia,
    VISION_FRAMES_PER_SECOND,
    VISION_SHORT_EDGE,
    VIDEO_CLIP_LABEL,
    VIDEO_PICTURE_LABEL,
    VIDEO_SOUND_LABEL,
    load_h3_text_encoder,
)
from thenoise.models.base import Conditioning
from thenoise.models.config import (
    EncodePromptArgs,
    ModelConfig,
    SamplingParams,
    VideoSamplingParams,
)
from thenoise.models.video_base import Condition, VideoModel, Waveform
from thenoise.samplers import Step
from thenoise.utils.sol_attention import routing_unavailable
from thenoise.utils.image_tensor import (
    ReferenceSizing, cover_crop, resize_to_short_edge)
from thenoise.utils.safetensors import Header
from thenoise.vae import load_h3_audio_vae, load_h3_video_vae
from thenoise.vae.h3 import CLIP_LENGTH, TemporalPlan, snap_frames
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
    """``Conditioning`` plus the modality tags the DiT modulates by, per branch: an
    non-empty negative prompt puts the keyframes at a different offset in each.
    """

    tags: Optional[torch.Tensor] = None
    null_tags: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _Streams:
    """The run's two latents, and the one flat tensor the solver steps between them.

    The streams cannot share an axis, so the pack is flat and the split is a count.
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
    """MiniMax H3: video with audio, keyed on frames or conditioned on references."""

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
                    "video": {"t2v": True, "i2v": True, "flf2v": True,
                                  "ref2v": True}}

    #: What the canvas means to a reference: an area to fit into, never a frame to crop
    #: to. ``fit_guides`` restates the cap as the run's own canvas; this is the claim for
    #: anywhere the rule is read on its own.
    REFERENCE_SIZING = ReferenceSizing(
        fit="area", cap=DEFAULT_PREFS["width"] * DEFAULT_PREFS["height"],
        align=CANVAS_MULTIPLE, down_only=True)
    # Routing is measured good on a clip and bad on a still.
    ATTENTION_METHODS = (*VideoModel.ATTENTION_METHODS, "optimized")

    def attention_routing_blocked(self):
        return routing_unavailable(self.device, self.dtype, self.params.head_dim)

    #: A keyframe enters at a fixed timestep and is never denoised, which is what
    #: ``index_timestep_zero`` means here.
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
            picture_label=VIDEO_PICTURE_LABEL, clip_label=VIDEO_CLIP_LABEL,
            sound_label=VIDEO_SOUND_LABEL,
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

    def fit_guides(self, images, target):
        """Resized, never cropped, and no ceiling on the count.

        A reference is not a frame of the clip, so nothing has to line up with the
        canvas: it keeps the shape it arrived in and packs on its own grid, with the
        canvas only deciding how much area that grid may cover.
        """
        sizing = dataclasses.replace(self.REFERENCE_SIZING, cap=target[0] * target[1])
        return [sizing.apply(image) for image in images]

    def encode_sound(self, pcm, sample_rate=None) -> torch.Tensor:
        """The codec's posterior mean for a reference's sound, in its own latent shape.

        Mono becomes the two lanes the stream carries, and any rate is moved here: a
        latent frame is a fixed number of samples at the codec's rate, so the sound has to
        arrive on that grid to be a condition at all.
        """
        with torch.no_grad():
            return self.audio_vae.encode_to_latents(pcm, sample_rate)

    def encode_references(self, pixels, frames=1) -> List[torch.Tensor]:
        """One latent per condition, in the order the prompt labels them.

        A condition carries its own frames and the codec is causal, so a reference clip
        goes through in one piece and comes back one latent for all of them, cropped to the
        ladder the decoder walks. One frame in is the codec's posterior mean, the same
        answer every time.
        """
        with torch.no_grad():
            return [self.vae.encode_pixels_to_latents(one[None, :, None] if one.ndim == 3
                                                      else one[None, :, :guide_clip(one.shape[1])])
                    for one in pixels]

    # ------------------------------------------------------------ conditioning
    def encode_prompt(self, args: EncodePromptArgs) -> H3VideoConditioning:
        """Prompt and conditions through the language model, as raw conditioning.

        The negative branch sees the same conditions: a keyframe is not something to steer
        away from.
        """
        media = presented(args.image, args.guides)
        cond, cond_mask, tags = self.text_encoder(args.prompt, media)
        null = null_mask = null_tags = None
        if args.guidance_scale > 1.0:
            null, null_mask, null_tags = self.text_encoder(args.negative_prompt, media)
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
    def _guide_sound(self, latent: torch.Tensor, frame: Optional[int],
                     frames: int, target_t: int) -> torch.Tensor:
        """A condition's sound, cut to the room the clip's audio timeline has for it.

        An anchored soundtrack sits inside the clip's own stretch of the time axis, so it
        ends where the clip does; one whose anchor leaves no whole audio frame is a sound
        this run cannot place. A condition with no frame keeps all of itself, because it
        moves the target ahead of it rather than into it.
        """
        if frame is None:
            return latent
        room = math.floor(target_t - FRAME_RESCALE * frame)
        if room < 1:
            raise ValueError(
                f"a sound anchored at frame {frame} has no room left in the {target_t} "
                f"audio frames of a {frames}-frame clip")
        return latent[..., :min(latent.shape[-1], room)]

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
        """Noise for both streams as one flat row: the carry is 1 at pure noise, so neither
        stream needs pre-scaling to travel together.
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
        conditions: Sequence[Condition] = (),
    ) -> torch.Tensor:
        """Build the packing this run steps through; the latent itself is untouched.

        ``conditions`` have been through the codec already, each with the frame it is
        anchored at or with none, and each carrying a picture, a sound, or the two of them
        as one condition.
        """
        keyframes = _as_list(ref)
        if (keyframes or conditions) and ref_method != "index_timestep_zero":
            raise ValueError(
                "H3 video reads a condition at the conditioned timestep and never "
                f"denoises it; '{ref_method}' describes overwriting a frame of the clip"
            )
        dev = torch.device(self.device)
        streams = self._streams(params)
        # A keyframe is a frame of the clip itself: the first opens it, the last closes it,
        # and a lone one is the opening frame.
        pinned = [Keyframe(index) for index in [0, params.frames - 1][:len(keyframes)]]
        # Each stream is read off its own cursor, in its own segment order: the trunk's
        # rows go pinned conditions first and the ones ahead of the timeline after them.
        ahead: List[Reference] = []
        video_rows = [*keyframes]
        sound_rows: List[torch.Tensor] = []
        ahead_video: List[torch.Tensor] = []
        ahead_sound: List[torch.Tensor] = []
        for one in conditions:
            if one.video is None and one.audio is None:
                raise ValueError("a condition with nothing in it conditions nothing")
            frame = (None if one.frame is None
                     else one.frame if one.frame >= 0 else params.frames + one.frame)
            sound = (self._guide_sound(one.audio, frame, params.frames, streams.audio[2])
                     if one.audio is not None else None)
            audio_t = 0 if sound is None else sound.shape[3]
            latent_t, latent_h, latent_w = (one.video.shape[2:] if one.video is not None
                                            else (0, 0, 0))
            if one.frame is None:
                ahead.append(Reference(kind=("audio" if one.video is None
                                             else "image" if latent_t == 1 else "video"),
                                       latent_t=latent_t, latent_h=latent_h,
                                       latent_w=latent_w, audio_t=audio_t))
                if one.video is not None:
                    ahead_video.append(one.video)
                if sound is not None:
                    ahead_sound.append(sound)
                continue
            # The condition's own length in pixel frames, by the decode plan the clip
            # follows: an anchor near the end must leave room for all of them.
            length = pixel_frames(latent_t)
            if not 0 <= frame < params.frames or frame + length > params.frames:
                raise ValueError(
                    f"a {length}-frame condition anchored at frame {one.frame} does not fit "
                    f"a {params.frames}-frame clip")
            pinned.append(Keyframe(frame=frame, latent_t=latent_t, audio_t=audio_t))
            if one.video is not None:
                video_rows.append(one.video)
            if sound is not None:
                sound_rows.append(sound)
        self._streams_ = streams
        self._shift = float(params.flow_shift)
        self._seed = params.seed
        self._keyframes = [one.to(dev, self.dtype) for one in [*video_rows, *ahead_video]]
        self._condition_audio = [one.to(dev, self.dtype)
                                 for one in [*sound_rows, *ahead_sound]]
        self._layout = Layout.av(cond.cond.shape[1], *streams.video[1:],
                                 audio_t=streams.audio[2], keyframes=pinned, refs=ahead)
        self._null_layout = (
            None if cond.null is None
            else Layout.av(cond.null.shape[1], *streams.video[1:],
                           audio_t=streams.audio[2], keyframes=pinned, refs=ahead)
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
        """One packed forward over both streams. ``t`` is the video sigma; the audio's is
        where its own schedule puts the same progress. The answer comes back in the carried
        variable, which is the one the solver steps.
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
            context, layout, self._keyframes, audio_conditions=self._condition_audio,
            tags=tags, seed=self._seed,
            sample_sigmas=self._sample_sigmas, rope=rope, attention=self._attention,
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

        The decoder re-reads each clip's overlap and composites the whole clip before it
        hands a frame back, so the canvas is live times that clip's frames however finely
        it is tiled in space.
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

        The carry ends at ``S``, the ratio of the two shifts, so what reaches the vocoder is
        ``S`` times the latent it was trained on.
        """
        _, carried = self._streams(params).split(latents)
        own = h3_sampling.carry(carried, 0.0, shift=params.flow_shift)
        pcm = self.audio_vae.decode_to_pcm(own.to(self.device))
        return Waveform(pcm[0].float(), self.audio_vae.sample_rate)

    def decode_frames(self, latents: torch.Tensor,
                      params: VideoSamplingParams) -> Iterator[torch.Tensor]:
        """The pack's video slice, decoded a clip at a time and yielded a frame at a time,
        already clamped to the range the pipeline works in.
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
        self._condition_audio: List[torch.Tensor] = []
        self._layout: Optional[Layout] = None
        self._null_layout: Optional[Layout] = None
        self._sample_sigmas = None

    def clear_video_state(self) -> None:
        self._reset_run()

    def _clear_weights(self) -> None:
        self.dit = self.vae = self.audio_vae = self.text_encoder = None


def presented(images, guides) -> List[GuideMedia]:
    """What the conditioner is shown, in the order the release presents it.

    The pictures first -- a clip's own frames, then a picture reference -- then each
    reference clip with its soundtrack's label ahead of it, then a standalone sound on its
    own. Ordinals count per kind across the whole run, so a first frame and a picture
    reference are ``<Picture 1>`` and ``<Picture 2>``, the way a prompt would name them.
    """
    items = [GuideMedia("picture", (_vision(image),)) for image in _as_list(images)]
    guides = list(guides)
    items += [GuideMedia("picture", (_vision(media.frames[0]),))
              for media in guides if len(media.frames) == 1]
    for media in guides:
        if len(media.frames) <= 1:
            continue
        if media.pcm is not None:
            items.append(GuideMedia("sound"))
        items.append(clip_vision(media))
    items += [GuideMedia("sound") for media in guides
              if not media.frames and media.pcm is not None]
    return items


def clip_vision(media) -> GuideMedia:
    """A reference clip at two frames a second, each tagged with the frame it came from.

    The index and the source rate are what the seconds marker the processor writes over
    each block is computed from, so they travel with the frames rather than being implied
    by how many there are. A file that states no rate is read at the sampling rate, which
    is the assumption the sampling itself was made under.
    """
    rate = media.fps or VISION_FRAMES_PER_SECOND
    shown = max(1, int(len(media.frames) * VISION_FRAMES_PER_SECOND / rate))
    indices = tuple(round(index * rate / VISION_FRAMES_PER_SECOND)
                    for index in range(shown))
    return GuideMedia("clip",
                      tuple(_vision(media.frames[min(index, len(media.frames) - 1)])
                            for index in indices),
                      indices, rate, len(media.frames))


def _vision(image):
    """The copy the vision tower is worth showing it: 32 pixels per token is coarser than
    the DiT's own, so a bigger picture buys detail the conditioning cannot carry."""
    return resize_to_short_edge(image, VISION_SHORT_EDGE, align=CANVAS_MULTIPLE)


def guide_clip(frames: int) -> int:
    """How many leading frames of a reference become one clip latent; 1 for a picture.

    The codec runs the shortest clip it trained on and whole clips more, so a guide is
    cropped down onto that ladder and never padded up to it: a clip that repeats its own
    last frame to reach a length is saying something the file did not. Fewer frames than
    the shortest clip is a picture, which is what a single frame is.
    """
    if frames < 2:
        return 1
    length = snap_frames(frames, upward=False)
    if length > frames:
        length -= CLIP_LENGTH
    return max(length, 1)


def _as_list(value) -> list:
    """``None`` / one item / a sequence of them -> a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


__all__ = ["H3VideoConditioning", "H3VideoModel"]
