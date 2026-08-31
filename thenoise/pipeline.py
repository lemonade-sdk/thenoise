"""The generation pipeline controller.

Owns the orchestration of generate/edit — encode -> denoise -> decode ->
postprocess -> PIL — delegating each stage to the model's kernels
(``thenoise.models.base.DiffusionModel``) and the pixel-domain upscaler
(``thenoise.upscale.pixel.PixelUpscalerManager``). Also owns the inference lock,
the single-entry stage cache (with cascade invalidation), upscale planning, and
cache-key construction.

Pipeline caching
----------------
Each stage is cached (single-entry, on-device tensors); keys are computed from
*resolved* parameters and nested so a change cascades downstream automatically.

  Stage          | Cache key depends on                    | Cached value
  ---------------+-----------------------------------------------+-----------
  Reference      | image (content hash; edit only)          | reference latent
  Prompt         | prompt, negative_prompt, guidance_scale, lora_specs | Conditioning
  Sampling       | prompt_key + size, steps, seed, sampler, lora_specs | latents
  Upscale+refine | (driven by decode cache hit below)      | —
  VAE decode     | sampling_key + refined constants        | pixels (fp32)
  Notch filter   | (not cached; runs right after decode)   | —
  Pixel upscale  | (not cached)                            | —
  Postprocess    | (not cached — cheap)                    | —

The edit path adds the ``Reference`` stage upstream of ``Prompt``: the
VAE-encoded input image is keyed by a content hash so re-editing the same image
with a different prompt never re-encodes it.

Upscaling
---------
Two modes driven by ``upscale_factor`` (f in (0.0, 8.0]) and ``upscale_type``:
``refined`` (default) runs the latent Sesqui upscaler ``UPSCALE_SCALE``x plus a
low-strength refine, adding a pixel-domain upscaler above factor 2; ``no-refiner``
uses only the pixel-domain upscaler (no latent 2x), limited to its detected scale.
Pixel upscalers are selected by name from ``upscaler_dir`` (CLI ``--upscaler-dir``);
without one, only ``refined`` factors <= 2 are available.
"""
from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
from PIL import Image

from thenoise.locks import inference_lock
from thenoise.models.base import DiffusionModel, Conditioning
from thenoise.models.config import EncodePromptArgs, GenerateRequest, SamplingParams
from thenoise.samplers import create_sampler
from thenoise.samplers.euler import EulerSampler
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.pipeline_cache import PipelineCache
from thenoise.utils.image_tensor import (
    center_crop,
    pil_to_pixels,
    pixels_to_pil,
    resize_to_target,
    resize_to_cover_center_crop,
)
from thenoise.postprocess.film_grain import film_grain
from thenoise.postprocess.nyquist import nyquist_notch
from thenoise.postprocess.rcas import rcas
from thenoise.utils.png import build_pnginfo

logger = logging.getLogger(__name__)


# Largest side used for the edit output size when neither width nor height is
# given; the first reference image's aspect ratio is preserved.
_EDIT_DEFAULT_SIZE = 1024


@dataclass(frozen=True)
class _ResolvedRequest:
    """Resolved pipeline values shared by ``generate`` and ``edit``.

    Produced by ``_resolve_pipeline`` so cache keys and the finalize tail use
    the same concrete (post-default) values in both paths.
    """

    width: int
    height: int
    steps: int
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


class PipelineController:
    """Owns the generate pipeline, driving a model's kernels + the pixel upscaler."""

    def __init__(
        self,
        model: DiffusionModel,
        pixel_upscalers: PixelUpscalerManager,
    ):
        self.model = model
        self._pixel_upscalers = pixel_upscalers
        self._lock = inference_lock
        self._cache = PipelineCache()

    # ------------------------------------------------------------ listing
    def list_loras(self):
        """List available LoRA names (delegates to the model)."""
        return self.model.list_loras()

    def list_pixel_upscalers(self):
        """List available pixel-upscaler names (delegates to the manager)."""
        return self._pixel_upscalers.list()

    # ---------------------------------------------------------- pipeline cache
    def _cache_key_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        lora_specs: Optional[list[str]] = None,
        ref_key: Optional[Tuple] = None,
    ) -> Tuple:
        """Cache key for prompt conditioning.

        Includes ``lora_specs`` so a change of LoRA invalidates any cached text
        fusion (the fused text is computed from the LoRA-adjusted DiT weights).

        The edit path appends ``ref_key`` (the image hash) so multimodal encoders
        (e.g. Qwen Image Edit) whose conditioning depends on the input image
        invalidate correctly.
        """
        base = (
            prompt,
            negative_prompt,
            guidance_scale,
            tuple(sorted(lora_specs)) if lora_specs else None,
        )
        if ref_key is None:
            return ("prompt",) + base
        return ("prompt_edit",) + base + (ref_key,)

    def _cache_key_reference(
        self,
        images: list[Image.Image],
        width: int,
        height: int,
    ) -> Tuple:
        """Cache key for the encoded reference latent(s) (edit path).

        Hashes each image's RGB bytes (in order) plus the target size, since
        refs are resize/center-cropped to the working resolution.
        """
        digests = tuple(
            hashlib.md5(img.convert("RGB").tobytes()).hexdigest() for img in images
        )
        return ("reference", width, height, digests)

    def _cache_key_sampling(
        self,
        prompt_key: Tuple,
        width: int,
        height: int,
        steps: int,
        seed: int,
        sampler: str,
        ref_method: Optional[str] = None,
    ) -> Tuple:
        """Cache key for the sampling (denoise stage).

        Embeds the prompt key so any prompt/guidance/LoRA change cascades. The
        edit path also embeds ``ref_method`` (it changes the reference packing, hence the denoise output).
        """
        base = (prompt_key, width, height, steps, seed, sampler)
        if ref_method is None:
            return ("sampling",) + base
        return ("sampling_edit",) + base + (ref_method,)

    def _cache_key_decode(
        self,
        sampling_key: Tuple,
        refined: bool,
    ) -> Tuple:
        """Cache key for the VAE decode stage.

        Embeds the sampling key so any upstream change cascades.
        When ``refined`` is True the latent upscale-and-refine pipeline produces
        different latents (at 2x), so the refined constants are added to the key.
        The pixel-domain ESRGAN step is deliberately NOT cached (it is fast), so
        it does not participate in the key.
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
        r = self._resolve_pipeline(request)
        return self._finalize(self._run(request, r), request, r)

    def edit(self, request: GenerateRequest) -> Image.Image:
        """Reference-latent instruction editing. Returns a single PIL image.

        Mirrors ``generate`` with two additions:

        * a cached **reference** stage — the VAE-encoded input image, keyed by
          content so re-editing the same image with a different prompt never
          re-encodes it;
        * image-aware prompt conditioning (``encode_prompt(..., image=...)``) so
          multimodal encoders (Qwen Image Edit) can consume the image as vision
          tokens in addition to the reference latent.

        The prompt/sampling keys embed the reference key, so any image change
        cascades downstream automatically.
        """
        model = self.model
        if not model.supports_edit:
            raise ValueError(f"model '{model.name}' does not support image editing")
        images = self._edit_images(request)
        if not images:
            raise ValueError("edit requires an input image")

        # Derive size from the FIRST image's aspect ratio unless width/height given.
        # Without explicit width/height the first reference is resized to
        # ``_EDIT_DEFAULT_SIZE`` (1024) on its largest side, aspect preserved.
        # A local ``replace`` keeps the caller's request unmutated.
        local = request
        if request.width is None and request.height is None:
            iw, ih = images[0].size
            target = _EDIT_DEFAULT_SIZE
            if iw >= ih:
                w = target
                h = round(ih * target / iw)
            else:
                h = target
                w = round(iw * target / ih)
            local = replace(request, width=w, height=h)

        r = self._resolve_pipeline(local)
        ref_key = self._cache_key_reference(images, r.width, r.height)
        return self._finalize(
            self._run(local, r, ref_key=ref_key, ref_method="index"),
            local, r,
        )

    def _edit_images(self, request: GenerateRequest) -> list[Image.Image]:
        """Normalize ``request.image`` (single OR list) to a list ([] if none)."""
        image = request.image
        if image is None:
            return []
        if isinstance(image, list):
            return list(image)
        return [image]

    def _run(
        self,
        request: GenerateRequest,
        r: _ResolvedRequest,
        *,
        ref_key: Optional[Tuple] = None,
        ref_method: Optional[str] = None,
    ) -> torch.Tensor:
        """Locked, cache-checked stage pipeline -> decoded pixels ``[C,H,W]``.

        Runs reference -> prompt -> sampling -> decode. ``ref_key``/``ref_method``
        are only set in the edit path; their presence selects the reference stage
        and image-aware prompt conditioning. Cache keys are computed from
        *resolved* parameters (``r``) so defaults are accounted for, and each
        stage embeds the upstream key so a change cascades downstream.
        """
        model = self.model
        is_edit = ref_key is not None

        prompt_key = self._cache_key_prompt(
            request.prompt, request.negative_prompt, r.guidance_scale,
            request.lora_specs, ref_key=ref_key,
        )
        sampling_key = self._cache_key_sampling(
            prompt_key, r.width, r.height, r.steps, r.seed, r.effective_sampler,
            ref_method=ref_method,
        )
        decode_key = self._cache_key_decode(sampling_key, r.refined)

        with self._lock:
            model.switch_loras(request.lora_specs, model.dit)

            # Stage 0: reference (image) latent — deterministic per image (edit).
            ref_latents: Optional[list[torch.Tensor]] = None
            if is_edit:
                if self._cache.reference_hit(ref_key):
                    ref_latents = self._cache.reference_get()
                else:
                    ref_latents = []
                    for img in self._edit_images(request):
                        # ComfyUI-style: scale each ref to cover the working size
                        # (center-crop if the aspect ratio differs).
                        cover = resize_to_cover_center_crop(img, r.width, r.height)
                        pixels = pil_to_pixels(cover)  # [C,H,W] fp32 [-1,1]
                        ref_latents.append(model.encode_reference(pixels))  # [1,C,H,W]
                    self._cache.reference_store(ref_key, ref_latents)

            # Stage 1: prompt conditioning (image-aware for multimodal encoders).
            if self._cache.prompt_hit(prompt_key):
                cond = self._cache.prompt_get()
            else:
                cond = model.encode_prompt(
                    EncodePromptArgs(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        guidance_scale=r.guidance_scale,
                        image=request.image if is_edit else None,
                    )
                )
                self._cache.prompt_store(prompt_key, cond)

            params = SamplingParams(
                height=r.height, width=r.width, steps=r.steps, seed=r.seed,
                guidance_scale=r.guidance_scale, sampler=r.effective_sampler,
            )

            # Stage 2: sampling (denoise) — ref-conditioned only in the edit path.
            if self._cache.sampling_hit(sampling_key):
                latents = self._cache.sampling_get()
            else:
                with torch.no_grad():
                    latents = self._denoise(
                        cond, params, ref_latents, ref_method or "index"
                    )
                self._cache.sampling_store(sampling_key, latents)

            # Stage 3/4: upscale + decode (interleaved so cache hits skip upscale).
            if self._cache.decode_hit(decode_key):
                pixels = self._cache.decode_get()
            else:
                if r.refined:
                    latents = self._upscale_and_refine(latents, cond, params)
                pixels = model.decode(latents)  # fp32 GPU tensor [C,H,W]
                self._cache.decode_store(decode_key, pixels)

            return pixels

    def _denoise(
        self,
        cond: Conditioning,
        params: SamplingParams,
        ref_latents: Optional[list[torch.Tensor]] = None,
        ref_method: str = "index",
    ) -> torch.Tensor:
        """Shared denoising pipeline over the model's ``schedule``.

        Builds the latent and schedule, then delegates the loop to the selected
        solver sampler (``euler`` or ``er_sde``). Each sampler calls
        ``denoise_step`` once per schedule step and runs integration in fp32.
        ``ref_latents``/``ref_method`` are only passed in the edit path.
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
        schedule = model.schedule(params)
        x = solver.sample(x, schedule, cond, params.guidance_scale, params.seed)
        return model.finalize_latent(x, params)

    # ------------------------------------------------------------- resolution
    def _resolve_pipeline(self, request: GenerateRequest) -> _ResolvedRequest:
        """Resolve request defaults into concrete pipeline values.

        Shared by ``generate`` and ``edit`` so both use identical resolution
        (cache keys must use the actual resolved values).
        """
        model = self.model
        width = request.width or model.DEFAULT_WIDTH
        height = request.height or model.DEFAULT_HEIGHT
        steps = request.steps or model.DEFAULT_STEPS
        guidance_scale = (
            model.DEFAULT_GUIDANCE_SCALE
            if request.guidance_scale is None
            else request.guidance_scale
        )

        pixel_upscaler = request.pixel_upscaler
        if pixel_upscaler and self._pixel_upscalers.upscaler_dir:
            pixel_upscaler = self._pixel_upscalers.validate(pixel_upscaler)
        elif pixel_upscaler:
            # No --upscaler-dir configured: ignore the requested pixel upscaler
            # and fall back to a refined (latent-only) upscale rather than failing.
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
        effective_sampler = request.sampler or model.SAMPLER

        # seed=-1 is treated as "random" (same as None)
        seed = request.seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)

        return _ResolvedRequest(
            width=width, height=height, steps=steps, guidance_scale=guidance_scale,
            factor=factor, upscale_type=upscale_type, target_width=target_width,
            target_height=target_height, refined=refined, pixel_scale=pixel_scale,
            effective_sampler=effective_sampler, seed=seed, pixel_upscaler=pixel_upscaler,
        )

    def _finalize(
        self,
        pixels: torch.Tensor,
        request: GenerateRequest,
        r: _ResolvedRequest,
    ) -> Image.Image:
        """Decoded pixels -> final PIL image (shared tail of generate/edit).

        Notch filter -> pixel upscaler -> resize -> postprocess -> PIL -> crop ->
        PNG metadata. Kept in one place so the two paths stay in lockstep.
        """
        model = self.model

        # Qwen notch filter must run immediately after VAE decode (before any
        # pixel-domain upscaling), so the 2px grid pattern is removed at its
        # native resolution.
        if request.qwen_vae_enhance:
            pixels = nyquist_notch(pixels)

        # Pixel-domain upscaler (fast, not cached) + GPU resize to target size.
        pixels = self._pixel_upscalers.apply(
            r.pixel_upscaler, pixels, r.pixel_scale
        )
        pixels = resize_to_target(pixels, r.target_width, r.target_height)

        # Stage 5: postprocess (cheap — not cached)
        pixels = self.postprocess(
            pixels,
            film_grain_strength=request.film_grain,
            sharpening=request.sharpening,
        )
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
            qwen_vae_enhance=request.qwen_vae_enhance,
            film_grain=request.film_grain,
            sharpening=request.sharpening,
            lora_specs=request.lora_specs,
            pixel_upscaler=r.pixel_upscaler,
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

        ``upscale_factor`` must be in (0.0, 8.0]. ``no-refiner`` mode has no
        latent 2x multiplier so it is limited to the pixel-upscaler scale. A pixel
        upscaler (selected by ``pixel_upscaler`` from ``upscaler_dir``) is
        required for ``no-refiner`` and for ``refined`` factors above the latent
        2x.
        """
        if upscale_type not in ("refined", "no-refiner"):
            raise ValueError(
                f"upscale_type must be 'refined' or 'no-refiner', "
                f"got {upscale_type!r}"
            )
        # Models without a latent upscaler (e.g. SDXL) cannot do ``refined``;
        # degrade it to pixel-only ``no-refiner`` so ``--upscale`` still works.
        if upscale_type == "refined" and not getattr(
            self.model, "supports_latent_upscale", lambda: True
        )():
            logger.warning(
                "%s does not support latent (refined) upscale; falling back to "
                "pixel-only upscale (requires a pixel upscaler).",
                self.model.name,
            )
            upscale_type = "no-refiner"
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
                    "scripts/download_esrgan.py)"
                )
            if upscale_type == "no-refiner" and factor > 1.0:
                raise ValueError(
                    "upscale_type='no-refiner' requires a pixel upscaler; "
                    "pass --pixel-upscaler PATH (or run "
                    "scripts/download_esrgan.py)"
                )
        else:
            # Max factor depends on the detected model scale: refined gets the
            # latent 2x multiplier, no-refiner does not.
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

        ``refined`` gets a 2x from the latent path, so the pixel upscaler is only
        needed when the factor exceeds that 2x. ``no-refiner`` has no latent
        multiplier and always needs it for any upscale. Uses the detected scale
        of the requested ``pixel_upscaler``.
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
        """Upscale the canonical latent ``UPSCALE_SCALE``x in latent space, then
        run a short low-strength refine denoise at the new size.

        Sesqui operates on raw VAE latents; the adaptor converts the canonical
        latent to/from that raw space. The refined result is the canonical latent
        at the upscaled spatial size, ready for ``decode``.
        """
        model = self.model
        upscaler, adaptor = model.load_latent_upscaler()
        scale = model.UPSCALE_SCALE
        z = latents.to(device=model.device, dtype=model.dtype)

        with torch.no_grad():
            # Adaptor math in fp32; the model runs in bf16.
            raw = adaptor.to_vae_latent(z).to(model.dtype)
            h, w = z.shape[-2:]
            target = adaptor.vae_target_size((scale * h, scale * w))
            raw_up = upscaler(raw, target)
            z_up = adaptor.from_vae_latent(raw_up.float()).to(model.dtype)

        # One short low-strength refine denoise at the upscaled size. The refine
        # runs on an independent schedule (see ``refine_latents``), so the
        # original ``steps`` count is deliberately NOT forwarded.
        up_params = replace(
            params,
            height=scale * params.height,
            width=scale * params.width,
        )
        return model.refine_latents(z_up, cond, up_params)

    # ---------------------------------------------------------- postprocess
    def postprocess(
        self,
        pixels: torch.Tensor,
        *,
        film_grain_strength: float = 0.0,
        sharpening: float = 0.0,
    ) -> torch.Tensor:
        """Tensor post-processing hook. Runs on the fp32 GPU pixels."""
        if sharpening > 0.0:
            pixels = rcas(pixels, strength=sharpening)
        if film_grain_strength > 0.0:
            pixels = film_grain(pixels, strength=film_grain_strength / 10.0)
        return pixels


__all__ = ["PipelineController"]
