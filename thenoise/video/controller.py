"""The video pipeline controller — what ``PipelineController`` is to an image model.

Owns the inference-mode boundary, what a request may ask of the loaded model, the
single-entry cache over the two encode stages, which component is resident for which
span and the headroom each stage asks for, progress and cancellation, LoRA switching,
the streamed decode and the container. The adapter keeps the kernels — encode, seed,
schedule, one denoise step, the codec's frames — and knows nothing about jobs or how
long a stage may take, which is also what lets ``Runtime.run`` rebuild an adapter that
reports itself unfit.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable, Optional

from thenoise.models.video_base import Condition

import torch
from PIL import Image
from tqdm import tqdm

from thenoise.inference import inference
from thenoise.jobs import Cancellation, stepping
from thenoise.models.config import EncodePromptArgs, VideoSamplingParams
from thenoise.samplers import check_sampler, create_sampler
from thenoise.stages import Stages
from thenoise.utils.image_tensor import load_image, pil_to_pixels, pixels_to_pil
from thenoise.utils.media import ReferenceMedia, read_reference
from thenoise.utils.lora_roles import check_lora_roles
from thenoise.utils.pipeline_cache import PipelineCache, content_digest
from thenoise.video.container import encode_video
from thenoise.video.spec import ResolvedVideo, VideoSpec

logger = logging.getLogger(__name__)


class VideoPipelineController:
    """Drives one video adapter through a resolved request."""

    def __init__(self, model):
        self.model = model
        self._cache = PipelineCache()

    # ------------------------------------------------------------ listing
    def list_loras(self) -> list[str]:
        """Available LoRA names."""
        return self.model.list_loras()

    def resolve(self, spec: VideoSpec) -> ResolvedVideo:
        """Refuse what this model cannot serve, then apply its defaults and geometry."""
        model = self.model
        mode = spec.mode
        if not model.capability("video")[mode]:
            raise ValueError(f"model '{model.name}' does not support {mode}")
        resolved = model.resolve_video(spec)
        # After the merge: a preference can name a solver or a role too.
        check_sampler(resolved.sampler)
        check_lora_roles(resolved.lora_specs, model.lora_components())
        return resolved

    # ---------------------------------------------------------- pipeline cache
    def _cache_key_prompt(self, spec: ResolvedVideo,
                          ref_key: Optional[tuple]) -> tuple:
        """Key for the RAW text conditioning; the fused context is per segment.

        ``ref_key`` keeps a conditioning that saw the reference frames apart from one
        that did not.
        """
        return ("prompt", spec.prompt, spec.negative_prompt, spec.guidance_scale,
                ref_key)

    def _cache_key_reference(self, images: list[Image.Image],
                             spec: ResolvedVideo) -> tuple:
        """Key for the encoded reference latents.

        The clip's length is in it because a codec that concatenates its reference
        encodes it into a clip-long latent.
        """
        return ("reference", self.model.REFERENCE_SIZING, (spec.width, spec.height),
                spec.frame_count, content_digest(images))

    @staticmethod
    def _reference_images(spec: ResolvedVideo) -> list[Image.Image]:
        """The clip's own frames, [] for text-to-video."""
        return [load_image(io.BytesIO(data))
                for data in (spec.image, spec.end_image) if data is not None]

    @staticmethod
    def _guides(spec: ResolvedVideo) -> list[ReferenceMedia]:
        """The reference conditions, each read from whatever kind of file it arrived as.

        Reading stops where the clip ends: nothing longer than the run has anywhere to go,
        and a file is whoever sent it, at whatever length they recorded.
        """
        return [read_reference(guide.media, name=f"reference {index}",
                               max_seconds=spec.frame_count / spec.fps)
                for index, guide in enumerate(spec.references, 1)]

    # ------------------------------------------------------------ render
    def render(self, spec: ResolvedVideo, token: Cancellation,
               progress: Callable[[str, int, int], None], output: Path) -> None:
        try:
            self._render(spec, token, progress, output)
        finally:
            self.model.clear_video_state()

    def _render(self, spec: ResolvedVideo, token: Cancellation,
                progress: Callable[[str, int, int], None], output: Path) -> None:
        """Encode -> denoise -> decode into ``output``, checking ``token`` throughout.

        The clip latent is not cached: nothing downstream of sampling exists for video.
        """
        model = self.model
        # A frame sizes the stages that hold a frame. Decoding is the model's to size:
        # frames reach the container as they are handed back, but a codec whose decoder
        # runs over the whole clip first holds the whole clip while it does.
        pixels = spec.width * spec.height
        decoding = model.decoded_pixels(spec)
        with inference():
            if spec.lora_specs and not model.lora_dir:
                raise ValueError("video LoRAs require --lora-dir")
            params = VideoSamplingParams(
                height=spec.height, width=spec.width, steps=spec.steps, seed=spec.seed,
                guidance_scale=spec.guidance_scale, sampler=spec.sampler,
                frames=spec.frame_count, flow_shift=spec.flow_shift,
                fps=spec.fps, attention=spec.attention,
            )
            images = self._reference_images(spec)
            guides = self._guides(spec)
            # What the text encoder is shown: a reference clip is one picture to it, for
            # as long as the presentation of a clip and its sound is unwritten.
            posters = [media.frames[0] for media in guides if media.frames]
            ref_key = None
            if images or guides:
                ref_key = (self._cache_key_reference([*images, *posters], spec),
                           tuple(guide.frame for guide in spec.references))
            prompt_key = self._cache_key_prompt(spec, ref_key)
            stages = Stages(model, self._cache)

            # Stage 0: the conditions the run steps alongside — keyframes, then references.
            ref: tuple[list, list] = ([], [])
            if images or guides:
                ref = stages.run(self._encode_references, images, guides, spec,
                                 resident=("vae",), pixels=pixels,
                                 cache="reference", key=ref_key)
                token.check()

            # Stage 1: prompt conditioning — text encoder only, and checked after
            # stage 0 because its store clears this one.
            cond_raw = stages.run(
                model.encode_prompt,
                EncodePromptArgs(prompt=spec.prompt, negative_prompt=spec.negative_prompt,
                                 guidance_scale=spec.guidance_scale,
                                 image=[*images, *posters] or None),
                resident=("text_encoder",),
                pixels=max(pixels, sum(image.width * image.height
                                       for image in [*images, *posters])),
                cache="prompt", key=prompt_key)
            token.check()

            latents = model.init_latents(params)
            x = None
            cond = None
            done = 0

            segments = model.segments(
                params, model.schedule(params), spec.guidance_scale
            )
            total_steps = sum(len(segment.steps) for segment in segments)
            previous = None

            for segment in segments:
                # LTX requires an upscaling/re-noising transition between
                # its two sampling phases. Other models use the default no-op.
                if previous is not None:
                    x = model.transition_segment(
                        x, previous, segment, params, token
                    )
                    token.check()

                def prepare(segment=segment):
                    nonlocal x, cond
                    model.switch_loras(
                        spec.lora_specs.get(segment.stage),
                        segment.module,
                        segment.stage,
                    )
                    model.begin_segment(segment)
                    cond = model.fuse_text(cond_raw)

                    # Preparation must happen while the stage is resident.
                    if x is None:
                        # Asked for only by a model that takes reference conditions, so
                        # the ones without them keep their own signature.
                        pinned = {"conditions": ref[1]} if ref[1] else {}
                        x = model.prepare_latent(
                            latents, cond, params, ref=ref[0] or None,
                            ref_method=model.pref("ref_method"), **pinned,
                        )

                x = stages.run(
                    lambda: self._denoise(
                        x, segment, cond, params, token,
                        progress, done, total_steps,
                    ),
                    resident=(segment.stage,),
                    pixels=pixels,
                    prepare=prepare,
                )

                done += len(segment.steps)
                previous = segment

            x = model.finalize_latent(x, params)
            del cond_raw, latents, ref

            # Audio decoding is optional and gets its own residency stage.
            audio = None
            roles = model.audio_components()

            if roles:
                audio = stages.run(
                    model.decode_audio, x,
                    params,
                    resident=roles,
                    pixels=pixels,
                )
                token.check()

            stages.run(
                self._mux, x, spec, params, token, progress, output, audio,
                resident=("vae",),
                pixels=decoding,
            )
            del x, audio

        logger.info("Video saved: %s", output)

    def _encode_references(self, images: list[Image.Image],
                           guides: list[ReferenceMedia],
                           spec: ResolvedVideo) -> tuple[list[torch.Tensor],
                                                         list[Condition]]:
        """The clip's frames and its reference conditions, fitted and put through the codec.

        Kept apart on the way out because the packing records which pinned row is which,
        and a reference is fitted by a different rule than a keyframe on purpose. Each
        keeps the frame it was anchored at, or none at all.

        Each reference is fitted on its own: two of them need not share a shape, since
        neither is cropped and each keeps a grid of its own in the trunk.
        """
        model = self.model

        def encoded(pictures, fit):
            return model.encode_references(
                [pil_to_pixels(f, model.pixel_channels)
                 for f in fit(pictures, (spec.width, spec.height))],
                frames=spec.frame_count)

        conditions = []
        for media, guide in zip(guides, spec.references):
            if len(media.frames) > 1:
                raise ValueError(
                    f"a reference clip of {len(media.frames)} frames needs the codec to "
                    "put a clip's frames into one latent, which it does not do yet")
            if media.pcm is not None:
                raise ValueError("a reference's sound needs condition rows in the "
                                 "trunk's audio stream, which it does not pack yet")
            conditions.append(Condition(video=encoded(media.frames, model.fit_guides)[0],
                                        frame=guide.frame))
        return (encoded(images, model.fit_references) if images else [], conditions)

    def _denoise(self, x, segment, cond, params, token: Cancellation, progress,
                 done: int, total: int):
        """One segment's solver run, from a fresh solver.

        ``stage.run`` retries an out-of-memory attempt, so the solver must not carry
        history from the attempt that failed.
        """
        return create_sampler(params.sampler, self.model).sample(
            x, list(segment.steps), cond, segment.guidance_scale, params.seed,
            step_callback=stepping(token, progress, "sampling", done, total))

    def _mux(self, latents, spec: ResolvedVideo, params: VideoSamplingParams,
             token: Cancellation, progress, output: Path, audio=None) -> None:
        """Decode into the container: the clip never exists all at once.

        Stays inside the ``vae`` stage because the generator pulls from the codec as the
        encoder drains it — which is also what lets one frame, rather than the clip, size
        the stage's headroom. The call itself is made inside, not beside the latents,
        because ``stages.run`` replays its callable on a retry and a generator started
        once would come back half-drained.
        """
        total = spec.frame_count

        def frames():
            progress("decoding", 0, total)
            with tqdm(total=total, desc="decoding") as bar:
                for i, frame in enumerate(self.model.decode_frames(latents, params)):
                    token.check()
                    progress("decoding", i + 1, total)
                    bar.update(1)
                    # uint8 is a finer step than a BF16 mantissa.
                    yield pixels_to_pil(frame)

        progress("encoding", 0, total)
        if audio is None:
            encode_video(frames(), spec, token, output, progress)
        else:
            encode_video(frames(), spec, token, output, progress,
                         audio=audio.samples, sample_rate=audio.sample_rate)


__all__ = ["VideoPipelineController"]
