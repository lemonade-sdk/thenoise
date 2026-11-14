"""The video pipeline controller — what ``PipelineController`` is to an image model.

Owns the inference-mode boundary, which component is resident for which span and the
headroom each stage asks for, progress and cancellation, LoRA switching, the streamed
decode and the container. The adapter keeps the kernels — encode, seed, schedule, one
denoise step, the codec's frames — and knows nothing about jobs or how long a stage may
take, which is also what lets ``Runtime.run`` rebuild an adapter that reports itself unfit.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable, Optional

import torch
from tqdm import tqdm

from thenoise.inference import inference
from thenoise.models.config import EncodePromptArgs, VideoSamplingParams
from thenoise.samplers import create_sampler
from thenoise.utils.image_tensor import load_image, pil_to_pixels, pixels_to_pil
from thenoise.video.container import encode_video
from thenoise.video.jobs import Cancellation
from thenoise.video.spec import ResolvedVideo, VideoSpec

log = logging.getLogger("uvicorn.error")


class VideoPipelineController:
    """Drives one video adapter through a resolved request."""

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------ listing
    def list_loras(self) -> list[str]:
        """Available LoRA names."""
        return self.model.list_loras()

    def resolve(self, spec: VideoSpec) -> ResolvedVideo:
        """A request with this model's defaults and geometry applied."""
        return self.model.resolve_video(spec)

    # ------------------------------------------------------------ render
    def render(self, spec: ResolvedVideo, token: Cancellation,
               progress: Callable[[str, int, int], None], output: Path) -> None:
        """Encode -> denoise -> decode into ``output``, checking ``token`` throughout."""
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
                frames=spec.frame_count,
            )
            with model.memory.stage("text_encoder", pixels=pixels) as stage:
                cond = stage.run(model.encode_prompt, EncodePromptArgs(
                    prompt=spec.prompt, negative_prompt=spec.negative_prompt,
                    guidance_scale=spec.guidance_scale))
            token.check()

            ref: Optional[list[torch.Tensor]] = None
            if spec.image is not None:
                image = load_image(io.BytesIO(spec.image))
                with model.memory.stage("vae", pixels=pixels) as stage:
                    ref = stage.run(
                        lambda: [model.encode_reference(pil_to_pixels(f, model.pixel_channels))
                                 for f in model.fit_references([image],
                                                               (spec.width, spec.height))])
                token.check()

            latents = model.init_latents(params)
            x = None
            done = 0
            for segment in model.segments(params, model.schedule(params),
                                          spec.guidance_scale):
                with model.memory.stage(segment.stage, pixels=pixels) as stage:
                    model.switch_loras(spec.lora_specs.get(segment.stage),
                                     segment.module, segment.stage)
                    model.begin_segment(segment)
                    if x is None:  # once, and inside a stage: preparing may need weights
                        x = model.prepare_latent(latents, cond, params, ref=ref,
                                                 ref_method=model.pref("ref_method"))
                    x = stage.run(self._denoise, x, segment, cond, params,
                                  token, progress, done, spec.steps)
                done += len(segment.steps)
            x = model.finalize_latent(x, params)
            del cond, latents, ref

            with model.memory.stage("vae", pixels=pixels) as stage:
                stage.run(self._mux, x, spec, token, progress, output)
            del x

        log.info("Video saved: %s", output)

    def _denoise(self, x, segment, cond, params, token: Cancellation, progress,
                 done: int, total: int):
        """One segment's solver run, from a fresh solver.

        ``stage.run`` retries an out-of-memory attempt, so the solver must not carry
        history from the attempt that failed.
        """
        return create_sampler(params.sampler, self.model).sample(
            x, list(segment.steps), cond, segment.guidance_scale, params.seed,
            step_callback=_stepping(token, progress, done, total))

    def _mux(self, latents, spec: ResolvedVideo, token: Cancellation,
             progress, output: Path) -> None:
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
        encode_video(frames(), spec, token, output, progress)


def _stepping(token: Cancellation, progress, done: int, total: int):
    """The sampler's per-step hook: one progress tick, one cancellation check.

    ``done`` is the steps already taken by earlier segments, so the job sees one clock
    however many spans the schedule was cut into.
    """
    def step(x, i, _total):
        progress("sampling", done + i, total)
        token.check()
        return x

    return step


__all__ = ["VideoPipelineController"]
