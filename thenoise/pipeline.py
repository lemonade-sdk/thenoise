"""The generation pipeline controller.

This class owns the *orchestration* of the generate pipeline — encode -> denoise
-> decode -> postprocess -> PIL — and delegates each stage to the model's kernels
(``thenoise.models.base.DiffusionModel``) and to the pixel-domain upscaler
(``thenoise.upscale.pixel.PixelUpscalerManager``). The model base class itself
concerns only actual generation, finishing at the VAE decode; everything after
decode (postprocess, resize, crop, PNG metadata) and every planning/decision step
lives here.

The controller owns:
  * the inference lock (serializes LoRA switching and on-device upscaler loads),
  * the single-entry stage cache (with cascade invalidation),
  * upscale planning and execution,
  * cache-key construction.

Pipeline caching
----------------
Each stage of the generate pipeline is cached (single-entry, on-device tensors).
Cache keys are computed from *resolved* parameters (after defaults are applied).
Keys are nested: the sampling key embeds the prompt key, and the decode key
embeds the sampling key. This gives automatic cascade invalidation — a change
at any stage invalidates that stage and all downstream stages.

  Stage          | Cache key depends on                    | Cached value
  ---------------+-----------------------------------------------+-----------
  Prompt         | prompt, negative_prompt, guidance_scale, lora_specs | Conditioning
  Sampling       | prompt_key + size, steps, seed, sampler, lora_specs | latents
  Upscale+refine | (driven by decode cache hit below)      | —
  VAE decode     | sampling_key + refined constants        | pixels (fp32)
  Notch filter   | (not cached; runs right after decode)   | —
  Pixel upscale  | (not cached)                            | —
  Postprocess    | (not cached — cheap)                    | —

Upscaling has two modes, driven by ``upscale_factor`` (f in (0.0, 8.0]) and
``upscale_type``:

  * ``refined`` (default): the latent (Sesqui) upscaler runs ``UPSCALE_SCALE``x
    in latent space followed by a low-strength refine denoise. For f in (1, 2]
    that 2x is the whole upscale; for f in (2, ...] a pixel-domain upscaler
    step (scale auto-detected from the model, 2/4) is added on the decoded
    image and the result is downscaled to f.
  * ``no-refiner``: only the pixel-domain upscaler step runs on the decoded
    image (no latent 2x multiplier), so f is limited to the detected scale.

Pixel-domain upscalers are selected by name from ``upscaler_dir`` (CLI
``--upscaler-dir``); the request's ``pixel_upscaler`` picks which model in that
directory is used. Today the only pixel-space upscaler is Real-ESRGAN, but the
nomenclature is kept generic so future pixel upscalers pass through the same
options. Only the last-used pixel upscaler is kept loaded (switched on change).
Without a pixel upscaler only ``refined`` factors <= 2 are available. Max factor
ranges follow the detected model scale: ``refined`` up to ``UPSCALE_SCALE *
pixel_scale``, ``no-refiner`` up to ``pixel_scale``.
"""
from __future__ import annotations

import random
from dataclasses import replace
from typing import Optional, Tuple

import torch
from PIL import Image

from thenoise.locks import inference_lock
from thenoise.models.base import DiffusionModel, Conditioning
from thenoise.models.config import GenerateRequest, SamplingParams
from thenoise.samplers import create_sampler
from thenoise.samplers.euler import EulerSampler
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.pipeline_cache import PipelineCache
from thenoise.utils.image_tensor import center_crop, pixels_to_pil, resize_to_target
from thenoise.postprocess.film_grain import film_grain
from thenoise.postprocess.nyquist import nyquist_notch
from thenoise.postprocess.rcas import rcas
from thenoise.utils.png import build_pnginfo


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
    ) -> Tuple:
        """Cache key for prompt conditioning.

        Includes ``lora_specs`` so a change of LoRA invalidates any cached text
        fusion (the fused text is computed from the LoRA-adjusted DiT weights).
        """
        return (
            "prompt",
            prompt,
            negative_prompt,
            guidance_scale,
            tuple(sorted(lora_specs)) if lora_specs else None,
        )

    def _cache_key_sampling(
        self,
        prompt_key: Tuple,
        width: int,
        height: int,
        steps: int,
        seed: int,
        sampler: str,
    ) -> Tuple:
        """Cache key for the sampling (denoise stage).

        Embeds the prompt key so any prompt/guidance/LoRA change cascades.
        """
        return (
            "sampling",
            prompt_key,
            width,
            height,
            steps,
            seed,
            sampler,
        )

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
        """Encode -> denoise -> decode -> postprocess. Returns a single PIL image.

        Each pipeline stage is cached (single-entry). Cache keys are computed from
        *resolved* parameters so that defaults are accounted for. A change at any
        stage invalidates that stage and all downstream stages automatically via
        the nested key structure.
        """
        model = self.model

        # --- resolve defaults (cache keys must use actual resolved values) ---
        width = request.width or model.DEFAULT_WIDTH
        height = request.height or model.DEFAULT_HEIGHT
        steps = request.steps or model.DEFAULT_STEPS
        guidance_scale = (
            model.DEFAULT_GUIDANCE_SCALE
            if request.guidance_scale is None
            else request.guidance_scale
        )

        # Resolve upscale parameters: ``upscale`` is a legacy alias for a 2x
        # refined upscale; an explicit factor/type overrides it. A requested
        # pixel upscaler is validated (exists in upscaler_dir) before planning.
        pixel_upscaler = request.pixel_upscaler
        if pixel_upscaler:
            pixel_upscaler = self._pixel_upscalers.validate(pixel_upscaler)
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

        # --- compute cache keys (pure data, no model access) ---
        prompt_key = self._cache_key_prompt(
            request.prompt, request.negative_prompt, guidance_scale,
            request.lora_specs,
        )
        sampling_key = self._cache_key_sampling(
            prompt_key, width, height, steps, seed, effective_sampler
        )
        decode_key = self._cache_key_decode(sampling_key, refined)

        # --- locked section: cache checks + model access ---
        with self._lock:
            model.switch_loras(request.lora_specs, model.dit)

            # Stage 1: prompt conditioning
            if self._cache.prompt_hit(prompt_key):
                cond = self._cache.prompt_get()
            else:
                cond = model.encode_prompt(
                    request.prompt, request.negative_prompt,
                    guidance_scale=guidance_scale,
                )
                self._cache.prompt_store(prompt_key, cond)

            params = SamplingParams(
                height=height, width=width, steps=steps, seed=seed,
                guidance_scale=guidance_scale, sampler=effective_sampler,
            )

            # Stage 2: sampling (denoise)
            if self._cache.sampling_hit(sampling_key):
                latents = self._cache.sampling_get()
            else:
                with torch.no_grad():
                    latents = self._denoise(cond, params)
                self._cache.sampling_store(sampling_key, latents)

            # Stage 3/4: upscale + decode (interleaved so cache hits skip upscale)
            if self._cache.decode_hit(decode_key):
                pixels = self._cache.decode_get()
            else:
                # Cache miss — run latent upscale+refine (if refined) then decode
                if refined:
                    latents = self._upscale_and_refine(latents, cond, params)
                pixels = model.decode(latents)  # fp32 GPU tensor [C,H,W]
                self._cache.decode_store(decode_key, pixels)

            # Qwen notch filter must run immediately after VAE decode (before any
            # pixel-domain upscaling), so the 2px grid pattern is removed at its
            # native resolution.
            if request.qwen_vae_enhance:
                pixels = nyquist_notch(pixels)

            # Pixel-domain upscaler (fast, not cached) + GPU resize to target size.
            pixels = self._pixel_upscalers.apply(
                pixel_upscaler, pixels, pixel_scale
            )
            pixels = resize_to_target(pixels, target_width, target_height)

            # Stage 5: postprocess (cheap — not cached)
            pixels = self.postprocess(
                pixels,
                film_grain_strength=request.film_grain,
                sharpening=request.sharpening,
            )
            image = pixels_to_pil(pixels)

            if (image.width, image.height) != (target_width, target_height):
                image = center_crop(image, target_width, target_height)

            # Attach PNG metadata
            pnginfo = build_pnginfo(
                model=model.name,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                upscale=request.upscale,
                upscale_factor=factor,
                upscale_type=upscale_type,
                sampler=effective_sampler,
                qwen_vae_enhance=request.qwen_vae_enhance,
                film_grain=request.film_grain,
                sharpening=request.sharpening,
                lora_specs=request.lora_specs,
                pixel_upscaler=pixel_upscaler,
            )
            image._pnginfo = pnginfo
            return image

    def _denoise(
        self,
        cond: Conditioning,
        params: SamplingParams,
    ) -> torch.Tensor:
        """Shared denoising pipeline over the model's ``schedule``.

        Builds the latent and schedule, then delegates the loop to the selected
        solver sampler (``euler`` or ``er_sde``). Each sampler calls
        ``denoise_step`` once per schedule step and runs integration in fp32.
        """
        model = self.model
        solver = create_sampler(params.sampler, model)

        latents = model.init_latents(params)
        x = model.prepare_latent(latents, cond, params)
        schedule = model.schedule(params)
        x = solver.sample(x, schedule, cond, params.guidance_scale, params.seed)
        return model.finalize_latent(x, params)

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
            # The raw VAE latent may be spatially larger than the pipeline latent
            # (Flux2 packs 2x2 into channels), so the target size goes through the
            # adaptor's spatial scale. For scale-1 formats this is the identity.
            target = adaptor.vae_target_size((scale * h, scale * w))
            raw_up = upscaler(raw, target)
            z_up = adaptor.from_vae_latent(raw_up.float()).to(model.dtype)

        # One short low-strength refine denoise at the upscaled size. The refine
        # runs on an independent schedule (see ``_refine``), so the original
        # ``steps`` count is deliberately NOT forwarded.
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
        """One low-strength refine denoise step on an already-clean latent."""
        model = self.model
        refine_steps = model.REFINE_STEPS
        denoise = model.REFINE_DENOISE
        new_steps = int(refine_steps / denoise)  # int(1/0.1) = 10

        # Last ``refine_steps`` steps of an independent ``new_steps`` schedule.
        refine_params = replace(params, steps=new_steps)
        full = model.schedule(refine_params)
        sub = full[-refine_steps:]
        strength = float(sub[0].t)  # sigma_hat == the step's timestep

        # ComfyUI CONST noise scaling: x = sigma*noise + (1-sigma)*z.
        generator = torch.Generator(device=model.device).manual_seed(params.seed)
        noise = torch.randn_like(z, generator=generator)
        noised = strength * noise + (1.0 - strength) * z

        x = model.prepare_latent(noised, cond, refine_params)
        solver = EulerSampler(model)
        x = solver.sample(
            x, sub, cond, params.guidance_scale, params.seed, desc="refining"
        )
        return model.finalize_latent(x, refine_params)

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
