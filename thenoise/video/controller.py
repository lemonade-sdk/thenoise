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

import torch
from PIL import Image
from tqdm import tqdm

from thenoise.inference import inference
from thenoise.jobs import Cancellation, stepping
from thenoise.models.config import EncodePromptArgs, VideoSamplingParams
from thenoise.samplers import check_sampler, create_sampler
from thenoise.stages import Stages
from thenoise.utils.image_tensor import load_image, pil_to_pixels, pixels_to_pil
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
        """The clip's reference frames, [] for text-to-video."""
        return [load_image(io.BytesIO(data))
                for data in (spec.image, spec.end_image) if data is not None]

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
        # One frame, not the clip: frames reach the container as they are decoded, so
        # nothing in the run grows with `frame_count`. Whether a long clip still wants
        # more headroom than one frame is a measurement, not a correctness question.
        pixels = spec.width * spec.height
        with inference():
            if spec.lora_specs and not model.lora_dir:
                raise ValueError("video LoRAs require --lora-dir")
            params = VideoSamplingParams(
                height=spec.height, width=spec.width, steps=spec.steps, seed=spec.seed,
                guidance_scale=spec.guidance_scale, sampler=spec.sampler,
                frames=spec.frame_count, flow_shift=spec.flow_shift,
                fps=spec.fps,
            )
            images = self._reference_images(spec)
            ref_key = (self._cache_key_reference(images, spec) if images else None)
            prompt_key = self._cache_key_prompt(spec, ref_key)
            stages = Stages(model, self._cache)

            # Stage 0: reference latent(s).
            ref = None
            if images:
                ref = stages.run(self._encode_references, images, spec,
                                 resident=("vae",), pixels=pixels,
                                 cache="reference", key=ref_key)
                token.check()

            # Stage 1: prompt conditioning — text encoder only, and checked after
            # stage 0 because its store clears this one.
            cond_raw = stages.run(
                model.encode_prompt,
                EncodePromptArgs(prompt=spec.prompt, negative_prompt=spec.negative_prompt,
                                 guidance_scale=spec.guidance_scale, image=images or None),
                resident=("text_encoder",),
                pixels=max(pixels, sum(image.width * image.height
                                       for image in images)),
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
                        x = model.prepare_latent(
                            latents, cond, params, ref=ref,
                            ref_method=model.pref("ref_method"),
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

            # Audio decoding is optional and belongs to its own
            # MemoryManager stage.
            audio = None
            roles = model.audio_components()

            if roles:
                audio = stages.run(
                    model.decode_audio, x,
                    resident=roles,
                    pixels=pixels,
                )
                token.check()

            stages.run(
                self._mux, x, spec, token, progress, output, audio,
                resident=("vae",),
                pixels=pixels,
            )
            del x, audio

        logger.info("Video saved: %s", output)

    def _encode_references(self, images: list[Image.Image],
                           spec: ResolvedVideo) -> list[torch.Tensor]:
        """The reference frames fitted to the clip's canvas, then through the codec."""
        model = self.model
        fitted = model.fit_references(images, (spec.width, spec.height))
        return model.encode_references(
            [pil_to_pixels(f, model.pixel_channels) for f in fitted],
            frames=spec.frame_count)

    def _denoise(self, x, segment, cond, params, token: Cancellation, progress,
                 done: int, total: int):
        """One segment's solver run, from a fresh solver.

        ``stage.run`` retries an out-of-memory attempt, so the solver must not carry
        history from the attempt that failed.
        """
        return create_sampler(params.sampler, self.model).sample(
            x, list(segment.steps), cond, segment.guidance_scale, params.seed,
            step_callback=stepping(token, progress, "sampling", done, total))

    def _mux(self, latents, spec: ResolvedVideo, token: Cancellation,
             progress, output: Path, audio=None) -> None:
        """Decode into the container: the clip never exists all at once.

        Stays inside the ``vae`` stage because the generator pulls from the codec as the
        encoder drains it — which is also what lets one frame, rather than the clip, size
        the stage's headroom.
        """
        total = spec.frame_count

        def frames():
            progress("decoding", 0, total)
            with tqdm(total=total, desc="decoding") as bar:
                for i, frame in enumerate(self.model.decode_frames(latents)):
                    token.check()
                    progress("decoding", i + 1, total)
                    bar.update(1)
                    # uint8 is a finer step than a BF16 mantissa.
                    yield pixels_to_pil(frame)

        progress("encoding", 0, total)
        if audio is None:
            # WAN and other video-only adapters keep their existing call.
            encode_video(frames(), spec, token, output, progress)
        else:
            sound, sample_rate = audio
            encode_video(frames(), spec, token, output, progress,
                         audio=sound, sample_rate=sample_rate)


__all__ = ["VideoPipelineController"]
