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
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import torch
from PIL import Image

from thenoise.inference import inference, inference_lock
from thenoise.models.base import DiffusionModel, Conditioning
from thenoise.models.config import EncodePromptArgs, GenerateRequest, SamplingParams
from thenoise.samplers import Step, create_sampler
from thenoise.samplers.euler import EulerSampler
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.pipeline_cache import PipelineCache
from thenoise.utils.image_tensor import (
    center_crop,
    load_image,
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


# Largest side for the edit output when neither width nor height is given.
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
    ref_method: str


def _normalize_sigmas(values: Sequence[float]) -> Tuple[float, ...]:
    """Validate a user sigma list, returning the full grid (terminal 0 included).

    The values decrease from 1.0 toward 0.0 and the trailing 0.0 is implied, so
    ``len(grid) - 1`` is the step count. Strictly decreasing keeps the ER-SDE
    solver's ``logit(sigma)`` and ``1/(1-sigma)`` finite.
    """
    if not values:
        raise ValueError("sigmas must not be empty")
    try:
        grid = [float(s) for s in values]
    except (TypeError, ValueError):
        raise ValueError("sigmas must be a list of numbers") from None
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


def _sigma_steps(
    grid: Sequence[float],
    device: str,
    dtype: torch.dtype,
) -> list[Step]:
    """Sigma grid (descending, ending on 0) -> one ``Step`` per grid gap.

    Tensors are in the model's own device/dtype because ``denoise_step`` consumes
    ``Step.t`` the way it consumes its own ``schedule()`` output.
    """
    ts = torch.tensor(list(grid), device=device, dtype=dtype)
    return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(len(grid) - 1)]


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
    return _sigma_steps([sigma * (1 - i / steps) for i in range(steps + 1)], device, dtype)


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
        lora_specs: Optional[list[str]] = None,
        ref_key: Optional[Tuple] = None,
    ) -> Tuple:
        """Cache key for prompt conditioning.

        Includes ``lora_specs`` because the fused text is computed from the
        LoRA-adjusted DiT weights. The edit path appends ``ref_key`` so encoders
        whose conditioning depends on the input image invalidate correctly.
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

        Hashes each image's normalized pixel bytes (RGBA when it carries
        transparency) in order, plus the target size, since refs are resized and
        center-cropped to the working resolution.
        """
        digests = tuple(
            hashlib.md5(load_image(img).tobytes()).hexdigest() for img in images
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
        kv_cache: bool = False,
        sigmas: Optional[Tuple[float, ...]] = None,
    ) -> Tuple:
        """Cache key for the sampling (denoise stage).

        Embeds the prompt key so any prompt/guidance/LoRA change cascades, and the
        custom ``sigmas`` grid so two grids of the same length never share cached
        latents. The edit path also embeds ``ref_method`` and ``kv_cache``, both of
        which change the denoise output.
        """
        base = (prompt_key, width, height, steps, seed, sampler, sigmas)
        if ref_method is None:
            return ("sampling",) + base
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
        r = self._resolve_pipeline(request)
        with inference():
            return self._finalize(self._run(request, r), request, r)

    def edit(self, request: GenerateRequest) -> Image.Image:
        """Reference-latent instruction editing. Returns a single PIL image.

        Mirrors ``generate`` with a cached **reference** stage (the VAE-encoded
        input image, keyed by content) and image-aware prompt conditioning
        (``encode_prompt(..., image=...)``) so encoders can consume the image as
        vision tokens in addition to the reference latent.
        """
        model = self.model
        if not model.capability("edit"):
            raise ValueError(f"model '{model.name}' does not support image editing")
        images = self._edit_images(request)
        if not images:
            raise ValueError("edit requires an input image")

        # Without explicit width/height the first reference sets the output
        # aspect ratio, resized to ``_EDIT_DEFAULT_SIZE`` on its largest side.
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
        with inference():
            return self._finalize(
                self._run(local, r, ref_key=ref_key, ref_method=r.ref_method),
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
        and image-aware prompt conditioning.
        """
        model = self.model
        is_edit = ref_key is not None

        # The KV cache is a reference-latent optimization: it needs an edit request.
        if r.kv_cache and not is_edit:
            raise ValueError("kv_cache requires an edit request (a reference image)")
        if r.kv_cache and is_edit and not model.capability("kv_cache"):
            raise ValueError(f"model '{model.name}' does not support the reference-latent KV cache")

        prompt_key = self._cache_key_prompt(
            request.prompt, request.negative_prompt, r.guidance_scale,
            request.lora_specs, ref_key=ref_key,
        )
        sampling_key = self._cache_key_sampling(
            prompt_key, r.width, r.height, r.steps, r.seed, r.effective_sampler,
            ref_method=ref_method, kv_cache=r.kv_cache, sigmas=r.sigmas,
        )
        decode_key = self._cache_key_decode(sampling_key, r.refined)

        with self._lock:
            memory = model.memory

            # Stage 0: reference (image) latent — deterministic per image (edit).
            ref_latents: Optional[list[torch.Tensor]] = None
            if is_edit:
                if self._cache.reference_hit(ref_key):
                    ref_latents = self._cache.reference_get()
                else:
                    ref_latents = []
                    for img in self._edit_images(request):
                        # Scale each ref to cover the working size, center-cropping
                        # when the aspect ratio differs.
                        cover = resize_to_cover_center_crop(img, r.width, r.height)
                        pixels = pil_to_pixels(cover, model.pixel_channels)
                        ref_latents.append(model.encode_reference(pixels))  # [1,C,H,W]
                    self._cache.reference_store(ref_key, ref_latents)

            # Stage 1: prompt conditioning — text encoder only.
            if self._cache.prompt_hit(prompt_key):
                cond = self._cache.prompt_get()
            else:
                memory.ensure("text_encoder")
                cond_raw = model.encode_prompt(
                    EncodePromptArgs(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        guidance_scale=r.guidance_scale,
                        image=request.image if is_edit else None,
                        width=r.width,
                        height=r.height,
                    )
                )
                memory.offload("text_encoder")

            params = SamplingParams(
                height=r.height, width=r.width, steps=r.steps, seed=r.seed,
                guidance_scale=r.guidance_scale, sampler=r.effective_sampler,
                kv_cache=r.kv_cache,
            )

            # Stage 2: sampling — the dit block. The DiT is resident here, so
            # LoRA switching (which may requantize) and text fusion run in it too.
            if self._cache.sampling_hit(sampling_key):
                latents = self._cache.sampling_get()
            else:
                memory.ensure("dit")
                model.switch_loras(request.lora_specs, model.dit)
                if not self._cache.prompt_hit(prompt_key):
                    cond = model.fuse_text(cond_raw)
                    self._cache.prompt_store(prompt_key, cond)
                latents = self._denoise(
                    cond, params, ref_latents, ref_method or "index",
                    sigmas=r.sigmas,
                )
                self._cache.sampling_store(sampling_key, latents)

            # Stage 3/4: upscale + decode (interleaved so cache hits skip upscale).
            # The DiT is offloaded first, keeping the decode peak to VAE + activations.
            if self._cache.decode_hit(decode_key):
                pixels = self._cache.decode_get()
            else:
                if r.refined:
                    memory.ensure("dit")
                    latents = self._upscale_and_refine(latents, cond, params)
                memory.offload("dit")
                memory.ensure("vae")
                pixels = model.decode(latents)  # fp32 GPU [C,H,W]
                self._cache.decode_store(decode_key, pixels)

            # Leave every swappable component offloaded at rest.
            memory.offload("dit")
            return pixels

    def _denoise(
        self,
        cond: Conditioning,
        params: SamplingParams,
        ref_latents: Optional[list[torch.Tensor]] = None,
        ref_method: str = "index",
        sigmas: Optional[Tuple[float, ...]] = None,
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
            schedule = _sigma_steps(sigmas, model.device, model.dtype)
        else:
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
        # Precedence: explicit request > checkpoint marker > model default.
        width = model.pref("width", request.width)
        height = model.pref("height", request.height)
        steps = model.pref("steps", request.steps)
        guidance_scale = model.pref("guidance_scale", request.guidance_scale)
        effective_sampler = model.pref("sampler", request.sampler)
        ref_method = model.pref("ref_method", request.ref_method)
        kv_cache = model.pref("kv_cache", request.kv_cache)

        sigmas: Optional[Tuple[float, ...]] = None
        if request.sigmas is not None:
            sigmas = _normalize_sigmas(request.sigmas)
            steps = len(sigmas) - 1
            if request.steps is not None and request.steps != steps:
                logger.warning(
                    "sigmas=%s overrides steps=%s: running %d steps",
                    sigmas, request.steps, steps,
                )

        # kv_cache only makes sense on an edit request.
        if request.kv_cache is None and request.image is None:
            kv_cache = False

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

        # The KV cache freezes the reference K/V, which is only valid when those
        # tokens are conditioned at timestep zero. With the method left on auto,
        # pick the one that makes it valid; an explicit ``index`` stays explicit and
        # is rejected below.
        if kv_cache and request.ref_method is None:
            ref_method = "index_timestep_zero"
        if kv_cache and ref_method != "index_timestep_zero":
            raise ValueError(
                "kv_cache requires ref_method='index_timestep_zero': the cached "
                "reference K/V are only step-invariant when the reference tokens are "
                "conditioned at timestep zero"
            )

        # seed=-1 means "random"
        seed = request.seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)

        return _ResolvedRequest(
            width=width, height=height, steps=steps, sigmas=sigmas,
            guidance_scale=guidance_scale,
            factor=factor, upscale_type=upscale_type, target_width=target_width,
            target_height=target_height, refined=refined, pixel_scale=pixel_scale,
            effective_sampler=effective_sampler, seed=seed, pixel_upscaler=pixel_upscaler,
            kv_cache=kv_cache, ref_method=ref_method,
        )

    def _finalize(
        self,
        pixels: torch.Tensor,
        request: GenerateRequest,
        r: _ResolvedRequest,
    ) -> Image.Image:
        """Decoded pixels -> final PIL image (shared tail of generate/edit).

        Notch filter -> pixel upscaler -> resize -> postprocess -> PIL -> crop ->
        PNG metadata.
        """
        model = self.model

        # The notch filter runs at the native decoded resolution, before any
        # pixel-domain upscaling.
        if request.qwen_vae_enhance:
            pixels = nyquist_notch(pixels)

        # Pixel-domain upscaler + GPU resize to target size.
        pixels = self._pixel_upscalers.apply(
            r.pixel_upscaler, pixels, r.pixel_scale
        )
        pixels = resize_to_target(pixels, r.target_width, r.target_height)

        # Stage 5: postprocess
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
            sigmas=list(r.sigmas) if r.sigmas else None,
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
