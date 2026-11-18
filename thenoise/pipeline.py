"""The generation pipeline controller.

Owns the orchestration of generate/edit — encode -> denoise -> decode ->
postprocess -> PIL — delegating each stage to the model's kernels, the model's
latent upscaler and the pixel-domain upscaler. Also owns the inference lock, the
inference-mode boundary around a whole request, the single-entry stage cache, the
upscale plan and cache-key construction.

Each stage is cached (single-entry, on-device tensors). Keys are built from
*resolved* parameters and each embeds its upstream key, so a change cascades
downstream automatically. On the edit path a ``reference`` stage sits upstream of
``prompt``, keyed by image content hash, so re-editing the same image with a
different prompt never re-encodes it.

``resolve`` says what a request means for the loaded model and ``produce`` runs it.
``generate`` and ``edit`` are the two together; the job API uses them apart, because
what a request means is cheap to learn and running it is not.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image

from thenoise.inference import inference, inference_lock
from thenoise.jobs import UNCANCELLABLE, Cancellation, Progress, idle, stepping
from thenoise.models.base import LATENT_REF_METHODS, Conditioning
from thenoise.models.config import EncodePromptArgs, GenerateRequest, SamplingParams
from thenoise.models.pixel_base import PixelModel
from thenoise.samplers import Step, check_sampler, create_sampler, sigma_steps
from thenoise.samplers.euler import EulerSampler
from thenoise.stages import Stages
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.pipeline_cache import PipelineCache, content_digest
from thenoise.utils.image_tensor import (
    center_crop,
    load_image,
    pil_to_pixels,
    pixels_to_pil,
    resize_to_target,
)
from thenoise.postprocess import DECODE, PostprocessOptions, apply_stage
from thenoise.utils.lora_roles import (
    check_lora_roles, flat_lora_specs, lora_specs_by_role,
)
from thenoise.utils.png import build_pnginfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedImage:
    """Resolved pipeline values shared by ``generate`` and ``edit``.

    Produced by ``_resolve_pipeline`` so cache keys and the finalize tail use
    the same concrete (post-default) values in both paths. Because none of it
    needs a weight on the device, a queued request is resolved — and so refused
    or echoed — before it is accepted.
    """

    width: int
    height: int
    steps: int
    # Normalized custom sigma grid (terminal 0 included), or None for the model's
    # own schedule.
    sigmas: Optional[Tuple[float, ...]]
    guidance_scale: float
    factor: float
    upscale_type: str
    target_width: int
    target_height: int
    refined: bool
    pixel_scale: int
    effective_sampler: str
    seed: int
    pixel_upscaler: Optional[str]
    kv_cache: bool
    #: How a reference reaches the run (``index``, ``index_timestep_zero``, ``vision``),
    #: or None when it brought no picture. Meaningless in the other modes.
    ref_method: Optional[str]
    #: What the model does with the request's picture: ``t2i`` or ``i2i``.
    mode: str
    #: LoRA specs by role, so a component the model does not have is refused here.
    lora_specs: Mapping[str, list[str]]

    @property
    def is_edit(self) -> bool:
        """The request carries a picture, whichever way this model was able to use it."""
        return self.mode != "t2i"

    @property
    def has_reference_latent(self) -> bool:
        """The run encodes the picture as a latent. ``vision`` conditions with a picture
        the conditioner read and no more, so it has no stage in the VAE."""
        return self.ref_method in LATENT_REF_METHODS

    def public(self) -> dict:
        """The request as this model will actually run it.

        The prompt and the reference images are deliberately absent: the first is
        the caller's own text and the second do not belong in a status document.
        """
        return {
            "mode": self.mode, "width": self.width, "height": self.height,
            "steps": self.steps, "sigmas": list(self.sigmas) if self.sigmas else None,
            "guidance_scale": self.guidance_scale, "sampler": self.effective_sampler,
            "seed": self.seed, "upscale_factor": self.factor,
            "upscale_type": self.upscale_type, "pixel_upscaler": self.pixel_upscaler,
            "kv_cache": self.kv_cache,
            "ref_method": self.ref_method,
            "lora_specs": dict(self.lora_specs),
        }


def _image_mode(model, *, is_edit: bool) -> str:
    """The image-medium mode a request runs in: ``t2i``, or ``i2i`` when it brought a
    picture the model can take.

    What the model does *with* that picture is a separate question, answered by the
    resolved reference method rather than by the mode. A model with no way with a
    picture at all is refused here, which is the one place that question is answered.
    """
    if not is_edit:
        return "t2i"
    if not model.capability("image")["i2i"]:
        raise ValueError(f"model '{model.name}' does not support image editing")
    return "i2i"


def _normalize_sigmas(
    values: Optional[Sequence[float]],
) -> Optional[Tuple[float, ...]]:
    """Validate a sigma grid, returning the full grid (terminal 0 included).

    ``None`` and ``[]`` both mean "no grid": the model's own schedule runs. Values
    decrease toward 0.0 and the trailing 0.0 is implied, so ``len(grid) - 1`` is the
    step count. Strictly decreasing keeps the ER-SDE solver's ``logit(sigma)`` and
    ``1/(1-sigma)`` finite.
    """
    if values is None:
        return None
    try:
        grid = [float(s) for s in values]
    except (TypeError, ValueError):
        raise ValueError("sigmas must be a list of numbers") from None
    if not grid:
        return None
    for s in grid:
        if not math.isfinite(s) or not 0.0 <= s <= 1.0:
            raise ValueError(f"sigmas must be finite values in [0.0, 1.0], got {s!r}")
    if any(a <= b for a, b in zip(grid, grid[1:])):
        raise ValueError(f"sigmas must be strictly decreasing, got {grid}")
    if grid[-1] != 0.0:
        grid.append(0.0)
    if len(grid) < 2:
        raise ValueError("sigmas must contain at least one value above 0.0")
    return tuple(grid)


def _refine_schedule(
    sigma: float,
    steps: int,
    device: str,
    dtype: torch.dtype,
) -> list[Step]:
    """The refine's sub-schedule: ``steps`` Euler steps integrating ``sigma`` to 0.

    Uniform in sigma: the deltas sum to the starting sigma, the loop lands on x0,
    and every step stays conditioned above 0. A uniform grid also keeps a given
    sigma the same strength on every model.
    """
    return sigma_steps([sigma * (1 - i / steps) for i in range(steps + 1)], device, dtype)


class PipelineController:
    """Owns the generate pipeline, driving a model's kernels + the pixel upscaler."""

    def __init__(
        self,
        model: PixelModel,
        pixel_upscalers: PixelUpscalerManager,
    ):
        self.model = model
        self._pixel_upscalers = pixel_upscalers
        self._lock = inference_lock
        self._cache = PipelineCache()

    # ------------------------------------------------------------ listing
    def list_loras(self):
        """List available LoRA names."""
        return self.model.list_loras()

    def list_pixel_upscalers(self):
        """List available pixel-upscaler names."""
        return self._pixel_upscalers.list()

    # ---------------------------------------------------------- pipeline cache
    def _cache_key_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        lora_specs: Mapping[str, list[str]] | None = None,
        ref_key: Optional[Tuple] = None,
        ref_method: Optional[str] = None,
    ) -> Tuple:
        """Cache key for prompt conditioning.

        Includes ``lora_specs`` because the fused text is computed from the
        LoRA-adjusted DiT weights, keyed by role so a LoRA on one component is not
        confused with the same file on another. The edit path appends ``ref_key`` so
        encoders whose conditioning depends on the input image invalidate correctly,
        and ``ref_method`` because ``vision`` conditions on that image through the text
        encoder itself: the same picture read instead of encoded is different
        conditioning, not the same conditioning twice.
        """
        base = (
            prompt,
            negative_prompt,
            guidance_scale,
            tuple(sorted((role, spec)
                         for role, specs in (lora_specs or {}).items()
                         for spec in specs)) or None,
        )
        if ref_key is None:
            return ("prompt",) + base
        return ("prompt_edit",) + base + (ref_key, ref_method)

    def _cache_key_reference(self, images: list[Image.Image], target: Tuple[int, int]) -> Tuple:
        """Cache key for the encoded reference latent(s) (edit path).

        Hashes each image's normalized pixel bytes (RGBA when it carries
        transparency) in order, plus the sizing policy that decided the size each one
        was encoded at and the canvas size that policy was given: a model may fit a
        reference to the result (``PixelModel.fit_references``).
        """
        return ("reference", self.model.REFERENCE_SIZING, target,
                content_digest(load_image(img) for img in images))

    def _cache_key_sampling(
        self,
        prompt_key: Tuple,
        width: int,
        height: int,
        steps: int,
        seed: int,
        sampler: str,
        ref_method: Optional[str] = None,
        kv_cache: bool = False,
        sigmas: Optional[Tuple[float, ...]] = None,
    ) -> Tuple:
        """Cache key for the sampling (denoise stage).

        Embeds the prompt key so any prompt/guidance/LoRA change cascades, and the
        custom ``sigmas`` grid so two grids of the same length never share cached
        latents. The KV-cache flag is part of both T2I and edit keys so toggling
        it cannot reuse an incompatible sampling result.
        """
        base = (prompt_key, width, height, steps, seed, sampler, sigmas)
        if ref_method is None:
            return ("sampling",) + base + (kv_cache,)
        return ("sampling_edit",) + base + (ref_method, kv_cache)

    def _cache_key_decode(
        self,
        sampling_key: Tuple,
        refined: bool,
    ) -> Tuple:
        """Cache key for the VAE decode stage.

        Embeds the sampling key so any upstream change cascades. A refined run
        produces different latents at 2x, so the refine constants join the key.
        """
        if not refined:
            return ("decode", sampling_key)
        return (
            "decode_refined",
            sampling_key,
            self.model.UPSCALE_SCALE,
            self.model.REFINE_STEPS,
            self.model.REFINE_DENOISE,
        )

    # ------------------------------------------------------------ pipeline
    def generate(self, request: GenerateRequest) -> Image.Image:
        """Text-to-image pipeline. Returns a single PIL image."""
        request, r = self.resolve(request, is_edit=False)
        return self.produce(request, r)

    def edit(self, request: GenerateRequest) -> Image.Image:
        """Reference-latent instruction editing. Returns a single PIL image.

        Mirrors ``generate`` with a cached **reference** stage (the VAE-encoded
        input image, keyed by content) and image-aware prompt conditioning
        (``encode_prompt(..., image=...)``) so encoders can consume the image as
        vision tokens in addition to the reference latent.
        """
        request, r = self.resolve(request, is_edit=True)
        return self.produce(request, r)

    def resolve(self, request: GenerateRequest, *, is_edit: bool
                ) -> tuple[GenerateRequest, ResolvedImage]:
        """Refuse what this model cannot serve, then fill in every default.

        None of it needs a weight on the device, which is what lets a queued request be
        checked before it is accepted rather than after it has waited behind another.
        An edit that left its size unset takes it from the first reference, so the
        request that comes back is not always the one that arrived.

        Which of a model's ways with a picture a request gets is the caller's to ask
        for and the model's to allow: see :func:`_image_mode` and
        :attr:`DiffusionModel.REF_METHODS`.
        """
        model = self.model
        if not is_edit and not model.capability("image")["t2i"]:
            raise ValueError(f"model '{model.name}' does not support text-to-image")
        mode = _image_mode(model, is_edit=is_edit)
        if mode == "t2i":
            return request, self._resolve_pipeline(request, is_edit=False)
        images = self._edit_images(request)
        if not images:
            raise ValueError("edit requires an input image")

        # Without explicit width/height the first reference decides the output size.
        width, height = model.size_from_reference(request.width, request.height,
                                                 images[0])
        local = replace(request, width=width, height=height)
        return local, self._resolve_pipeline(local, is_edit=True)

    def produce(self, request: GenerateRequest, r: ResolvedImage, *,
                token: Cancellation = UNCANCELLABLE,
                progress: Progress = idle) -> Image.Image:
        """An already-resolved request -> the finished image.

        The half ``generate`` and ``edit`` hand to a job queue, which has resolved the
        request already and runs this later. ``token`` and ``progress`` are how a job
        under way reports itself; a synchronous caller has nobody to report to.
        """
        images = self._edit_images(request) if r.is_edit else []
        ref_key = (self._cache_key_reference(images, (r.width, r.height))
                   if images else None)
        with inference():
            pixels = self._run(request, r, ref_key=ref_key,
                               token=token, progress=progress)
            progress("postprocessing")
            return self._finalize(pixels, request, r,
                                  source=images[0] if images else None)

    def _edit_images(self, request: GenerateRequest) -> list[Image.Image]:
        """Normalize ``request.image`` (single OR list) to a list ([] if none)."""
        image = request.image
        if image is None:
            return []
        if isinstance(image, list):
            return list(image)
        return [image]

    def _encode_references(self, target: Tuple[int, int],
                           request: GenerateRequest) -> list[torch.Tensor]:
        """This edit's references, fitted to the result and through the codec."""
        model = self.model
        fitted = model.fit_references(self._edit_images(request), target)
        return model.encode_references(
            [pil_to_pixels(image, model.pixel_channels) for image in fitted])

    def _run(
        self,
        request: GenerateRequest,
        r: ResolvedImage,
        *,
        ref_key: Optional[Tuple] = None,
        token: Cancellation = UNCANCELLABLE,
        progress: Progress = idle,
    ) -> torch.Tensor:
        """Locked, cache-checked stage pipeline -> decoded pixels ``[C,H,W]``.

        Runs reference -> prompt -> sampling -> decode. The reference-latent stage runs
        only when the resolved method wants one; a picture the conditioner alone read
        still keys the prompt cache, because that conditioning saw it.

        Each stage boundary is a progress tick and a cancellation point, so a job can
        still be stopped between stages when its solver never got a step in.
        """
        model = self.model
        is_edit = r.is_edit
        loras = r.lora_specs
        stages = Stages(model, self._cache)

        prompt_key = self._cache_key_prompt(
            request.prompt, request.negative_prompt, r.guidance_scale,
            loras, ref_key=ref_key, ref_method=r.ref_method,
        )
        sampling_key = self._cache_key_sampling(
            prompt_key, r.width, r.height, r.steps, r.seed, r.effective_sampler,
            ref_method=r.ref_method, kv_cache=r.kv_cache, sigmas=r.sigmas,
        )
        decode_key = self._cache_key_decode(sampling_key, r.refined)

        with self._lock:
            progress("conditioning")
            token.check()

            # Stage 0: reference (image) latent — deterministic per image (edit).
            ref_latents: Optional[list[torch.Tensor]] = None
            if r.has_reference_latent:
                ref_latents = stages.run(self._encode_references,
                                         (r.width, r.height), request,
                                         resident=("vae",), pixels=r.width * r.height,
                                         cache="reference", key=ref_key)

            # Stage 1: raw prompt conditioning; the DiT fuses and stores it in stage 2.
            if self._cache.hit("prompt", prompt_key):
                cond = self._cache.get("prompt")
            else:
                seen = self._edit_images(request)
                cond_raw = stages.run(
                    model.encode_prompt,
                    EncodePromptArgs(prompt=request.prompt,
                                     negative_prompt=request.negative_prompt,
                                     guidance_scale=r.guidance_scale,
                                     image=request.image if is_edit else None),
                    resident=("text_encoder",),
                    pixels=sum(image.width * image.height for image in seen))

            params = SamplingParams(
                height=r.height, width=r.width, steps=r.steps, seed=r.seed,
                guidance_scale=r.guidance_scale, sampler=r.effective_sampler,
                kv_cache=r.kv_cache,
            )
            token.check()

            # Stage 2: sampling — the dit block. The DiT is resident here, so
            # LoRA switching (which may requantize) and text fusion run in it too.
            progress("sampling", 0, r.steps)

            def fuse():
                nonlocal cond
                for role, module in model.lora_components().items():
                    model.switch_loras(loras.get(role), module, role)
                if not self._cache.hit("prompt", prompt_key):
                    cond = model.fuse_text(cond_raw)
                    self._cache.store("prompt", prompt_key, cond)

            latents = stages.run(
                lambda: self._denoise(cond, params, ref_latents,
                                      r.ref_method or "index",
                                      sigmas=r.sigmas, token=token, progress=progress),
                resident=("dit",), pixels=r.width * r.height,
                cache="sampling", key=sampling_key, prepare=fuse)

            # Stage 3/4: upscale + decode (interleaved so cache hits skip upscale).
            if self._cache.hit("decode", decode_key):
                pixels = self._cache.get("decode")
            else:
                if r.refined:
                    # the upscaler builds its weights here and may round-trip the VAE;
                    # nothing moves once a stage has started, so both are held
                    progress("upscaling")
                    latents = stages.run(self._upscale_and_refine, latents, cond, params,
                                         resident=("dit", "vae"),
                                         pixels=r.target_width * r.target_height)
                    token.check()
                progress("decoding")
                pixels = stages.run(model.decode, latents,  # fp32 GPU [C,H,W]
                                    resident=("vae",),
                                    pixels=r.target_width * r.target_height,
                                    cache="decode", key=decode_key)

            return pixels

    def _denoise(
        self,
        cond: Conditioning,
        params: SamplingParams,
        ref_latents: Optional[list[torch.Tensor]] = None,
        ref_method: str = "index",
        sigmas: Optional[Tuple[float, ...]] = None,
        token: Cancellation = UNCANCELLABLE,
        progress: Progress = idle,
    ) -> torch.Tensor:
        """Shared denoising pipeline over the model's ``schedule``.

        Builds the latent and schedule, then delegates the loop to the selected
        solver. A ``sigmas`` grid replaces the model's own schedule verbatim (no
        shift applied).
        """
        model = self.model
        solver = create_sampler(params.sampler, model)

        latents = model.init_latents(params)
        if ref_latents is not None:
            x = model.prepare_latent(
                latents, cond, params, ref=ref_latents, ref_method=ref_method
            )
        else:
            x = model.prepare_latent(latents, cond, params)
        if sigmas is not None:
            schedule = sigma_steps(sigmas, model.device, model.dtype)
        else:
            schedule = model.schedule(params)
        x = solver.sample(x, schedule, cond, params.guidance_scale, params.seed,
                          step_callback=stepping(token, progress))
        return model.finalize_latent(x, params)

    # ------------------------------------------------------------- resolution
    def _resolve_pipeline(
        self, request: GenerateRequest, *, is_edit: bool
    ) -> ResolvedImage:
        """Resolve request defaults into concrete pipeline values.

        Shared by ``generate`` and ``edit`` so both use identical resolution
        (cache keys must use the actual resolved values). Everything the loaded model
        refuses for reasons it can see without running — a capability it lacks, a solver
        it has none of, a LoRA role it has no component for — is refused here too, so a
        queued request fails at submit.
        """
        model = self.model
        mode = _image_mode(model, is_edit=is_edit)
        reference = mode != "t2i"
        if request.ref_method is not None and not reference:
            raise ValueError("ref_method describes a reference; this request has none")
        # Precedence: explicit request > checkpoint marker > model default.
        width = model.pref("width", request.width)
        height = model.pref("height", request.height)
        steps = model.pref("steps", request.steps)
        guidance_scale = model.pref("guidance_scale", request.guidance_scale)
        effective_sampler = model.pref("sampler", request.sampler)
        ref_method = (model.pref("ref_method", request.ref_method) if reference
                      else None)
        # A model's ways with a picture are its own, so an unusable one is refused here
        # rather than reaching a stage that would quietly ignore it.
        if reference and ref_method not in model.REF_METHODS:
            raise ValueError(
                f"model '{model.name}' does not support ref_method {ref_method!r}; "
                f"it has {list(model.REF_METHODS)}")
        # After the merge: a preference can name the solver too.
        check_sampler(effective_sampler)

        sigmas = _normalize_sigmas(model.pref("sigmas", request.sigmas))
        if sigmas is not None:
            sigma_steps = len(sigmas) - 1
            if request.steps is None or request.steps == sigma_steps:
                steps = sigma_steps
            elif request.sigmas is not None:
                logger.warning(
                    "sigmas=%s overrides steps=%s: running %d steps",
                    sigmas, request.steps, sigma_steps,
                )
                steps = sigma_steps
            else:
                logger.info(
                    "steps=%s drops the inherited %d-step sigma grid: running the "
                    "model schedule",
                    request.steps, sigma_steps,
                )
                sigmas = None

        if request.kv_cache is None:
            kv_cache = bool(
                ref_method == "index_timestep_zero"
                and model.capability("kv_cache")
            )
        else:
            if request.kv_cache and not model.capability("kv_cache"):
                raise ValueError(
                    f"model '{model.name}' does not support the "
                    "reference-latent KV cache"
                )
            kv_cache = request.kv_cache

        # Roles are components of this model, so a request naming one it does not have
        # is refused here rather than silently dropping the LoRAs at the switch.
        loras = lora_specs_by_role(request.lora_specs)
        check_lora_roles(loras, model.lora_components())

        pixel_upscaler = request.pixel_upscaler
        if pixel_upscaler and self._pixel_upscalers.upscaler_dir:
            pixel_upscaler = self._pixel_upscalers.validate(pixel_upscaler)
        elif pixel_upscaler:
            # No --upscaler-dir configured: fall back to a latent-only upscale.
            pixel_upscaler = None
        upscale_factor = request.upscale_factor
        if request.upscale and upscale_factor == 1.0:
            upscale_factor = float(model.UPSCALE_SCALE)
        factor, upscale_type = self._resolve_upscale(
            upscale_factor, request.upscale_type, pixel_upscaler
        )
        width, height = model.resolve_size(width, height)
        target_width = width
        target_height = height
        if factor != 1.0:
            target_width = round(width * factor)
            target_height = round(height * factor)
        refined = upscale_type == "refined" and factor > 1.0
        pixel_scale = self._pixel_upscaler_scale_for(
            factor, upscale_type, pixel_upscaler
        )

        # Reference-method restrictions apply only to a run with a reference latent;
        # a cache that freezes something else takes whatever method the run resolved.
        if kv_cache and ref_method in LATENT_REF_METHODS and request.ref_method is None:
            ref_method = "index_timestep_zero"
        if kv_cache and ref_method and ref_method != "index_timestep_zero":
            raise ValueError(
                "kv_cache requires ref_method='index_timestep_zero': the cached "
                "reference K/V are only step-invariant when the reference tokens are "
                "conditioned at timestep zero"
            )

        # seed=-1 means "random"
        seed = request.seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)

        return ResolvedImage(
            width=width, height=height, steps=steps, sigmas=sigmas,
            guidance_scale=guidance_scale,
            factor=factor, upscale_type=upscale_type, target_width=target_width,
            target_height=target_height, refined=refined, pixel_scale=pixel_scale,
            effective_sampler=effective_sampler, seed=seed, pixel_upscaler=pixel_upscaler,
            kv_cache=kv_cache, ref_method=ref_method, mode=mode, lora_specs=loras,
        )

    def _finalize(
        self,
        pixels: torch.Tensor,
        request: GenerateRequest,
        r: ResolvedImage,
        *,
        source: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Decoded pixels -> final PIL image (shared tail of generate/edit).

        Decode-stage filters -> pixel upscaler -> resize -> final filters -> PIL ->
        crop -> PNG metadata.
        """
        model = self.model
        postprocess = request.postprocess_options()

        # The decode-stage filters run at the native decoded resolution, before any
        # pixel-domain upscaling.
        pixels = apply_stage(pixels, postprocess, DECODE)

        # Pixel-domain upscaler + GPU resize to target size.
        pixels = self._pixel_upscalers.apply(
            r.pixel_upscaler, pixels, r.pixel_scale
        )
        pixels = resize_to_target(pixels, r.target_width, r.target_height)

        # Stage 5: postprocess
        pixels = self.postprocess(pixels, postprocess)
        image = pixels_to_pil(pixels)

        if (image.width, image.height) != (r.target_width, r.target_height):
            image = center_crop(image, r.target_width, r.target_height)

        # Attach PNG metadata
        pnginfo = build_pnginfo(
            model=model.name,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=r.width,
            height=r.height,
            steps=r.steps,
            guidance_scale=r.guidance_scale,
            seed=r.seed,
            upscale=request.upscale,
            upscale_factor=r.factor,
            upscale_type=r.upscale_type,
            sampler=r.effective_sampler,
            postprocess=postprocess,
            lora_specs=flat_lora_specs(r.lora_specs) or None,
            pixel_upscaler=r.pixel_upscaler,
            sigmas=list(r.sigmas) if r.sigmas else None,
            ref_method=r.ref_method,
            kv_cache=r.kv_cache,
            source=source,
        )
        image._pnginfo = pnginfo
        return image

    # ------------------------------------------------------------- upscale plan
    def _resolve_upscale(
        self,
        factor: float,
        upscale_type: str,
        pixel_upscaler: Optional[str] = None,
    ) -> tuple[float, str]:
        """Validate and return the effective (factor, type).

        ``upscale_factor`` must be in (0.0, 8.0]. A pixel upscaler (selected by
        ``pixel_upscaler`` from ``upscaler_dir``) is required for ``no-refiner``
        and for ``refined`` factors above the latent 2x.
        """
        if upscale_type not in ("refined", "no-refiner"):
            raise ValueError(
                f"upscale_type must be 'refined' or 'no-refiner', "
                f"got {upscale_type!r}"
            )
        if not 0.0 < factor <= 8.0:
            raise ValueError("upscale_factor must be in (0.0, 8.0]")
        scale = self._pixel_upscalers.scale(pixel_upscaler)
        model_scale = self.model.UPSCALE_SCALE
        if scale == 0:
            # No pixel upscaler: only ``refined`` factors up to the latent 2x work.
            if factor > model_scale:
                raise ValueError(
                    f"upscale_factor > {model_scale} requires a pixel "
                    "upscaler; pass --pixel-upscaler PATH (or run "
                    "scripts/download.py --model esrgan)"
                )
            if upscale_type == "no-refiner" and factor > 1.0:
                raise ValueError(
                    "upscale_type='no-refiner' requires a pixel upscaler; "
                    "pass --pixel-upscaler PATH (or run "
                    "scripts/download.py --model esrgan)"
                )
        else:
            # The max factor is the detected model scale, times the latent 2x for
            # ``refined``.
            max_refined = model_scale * scale
            if upscale_type == "refined" and factor > max_refined:
                raise ValueError(
                    f"upscale_type='refined' with a {scale}x pixel upscaler is "
                    f"limited to factor {max_refined}"
                )
            if upscale_type == "no-refiner" and factor > scale:
                raise ValueError(
                    f"upscale_type='no-refiner' with a {scale}x pixel upscaler "
                    f"is limited to factor {scale}"
                )
        return factor, upscale_type

    def _pixel_upscaler_scale_for(
        self,
        factor: float,
        upscale_type: str,
        pixel_upscaler: Optional[str] = None,
    ) -> int:
        """Pixel-upscaler scale to apply for (factor, type, name), or 0 to skip.

        ``refined`` already gets a 2x from the latent path, so the pixel upscaler
        is only needed above that; ``no-refiner`` needs it for any upscale.
        """
        if not pixel_upscaler:
            return 0
        scale = self._pixel_upscalers.scale(pixel_upscaler)
        if scale == 0 or factor <= 1.0:
            return 0
        if upscale_type == "no-refiner":
            return scale
        return scale if factor > self.model.UPSCALE_SCALE else 0

    # ------------------------------------------------------- latent upscale
    def _upscale_and_refine(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
    ) -> torch.Tensor:
        """Hand the DiT's latent to the model's latent upscaler, then refine it.

        The upscaler takes the canonical latent at the DiT's resolution and returns
        the canonical latent at ``UPSCALE_SCALE`` times it; what is left here is the
        short low-strength refine whose output feeds ``decode``.
        """
        model = self.model
        z_up = model.get_upscaler()(latents)
        scale = model.UPSCALE_SCALE

        # Only the size is forwarded: the refine brings its own sigma and steps.
        up_params = replace(
            params,
            height=scale * params.height,
            width=scale * params.width,
        )
        return self._refine(z_up, cond, up_params)

    def _refine(
        self,
        z: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
    ) -> torch.Tensor:
        """A short low-strength refine denoise on an already-clean latent."""
        model = self.model
        steps = model.REFINE_STEPS
        sigma = model.REFINE_DENOISE
        if steps <= 0 or sigma <= 0.0:
            return z

        # Noise scaling: x = sigma*noise + (1-sigma)*z.
        generator = torch.Generator(device=model.device).manual_seed(params.seed)
        noise = torch.randn_like(z, generator=generator)
        noised = sigma * noise + (1.0 - sigma) * z

        # No ``ref`` is forwarded, so the refine runs unconditioned on the edit
        # reference.
        refine_params = replace(params, steps=steps)
        x = model.prepare_latent(noised, cond, refine_params)
        solver = EulerSampler(model)
        x = solver.sample(
            x,
            _refine_schedule(sigma, steps, model.device, model.dtype),
            cond,
            params.guidance_scale,
            params.seed,
            desc="refining",
        )
        return model.finalize_latent(x, refine_params)

    # ---------------------------------------------------------- postprocess
    def postprocess(
        self,
        pixels: torch.Tensor,
        options: Optional[PostprocessOptions] = None,
    ) -> torch.Tensor:
        """Tensor post-processing hook. Runs on the fp32 GPU pixels.

        The final stage of the declared chain, in declaration order; nothing asked
        for leaves the tensor itself untouched.
        """
        return apply_stage(
            pixels, options if options is not None else PostprocessOptions()
        )


__all__ = ["PipelineController", "ResolvedImage"]
