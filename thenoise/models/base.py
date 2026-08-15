"""Abstract interface for a diffusion model adapter.

The base class owns the entire high-level pipeline — encode -> denoise -> decode
-> postprocess -> PIL — plus the inference lock, the denoising loop, and the
tensor/PIL conversions. Subclasses implement the model-specific kernels and load
their own VAE:

  * ``detect(f)``            — recognize this model's DiT from a safetensors handle.
  * ``encode_prompt(...)``   — text -> conditioning embeddings (cond + null).
  * ``init_latents(...)``    — seeded noise in the canonical 4D latent format.
  * ``prepare_latent(...)``  — canonical -> model-internal latent (once, pre-loop).
  * ``schedule(...)``        — the model's timestep/step-size schedule.
  * ``denoise_step(...)``    — one DiT forward + CFG, returning a velocity.
  * ``finalize_latent(...)`` — model-internal -> canonical latent (once, post-loop).
  * ``resolve_size(...)``    — per-model size rounding / validation.
  * ``_upscale_format(...)``  — required: the latent-format name for this
    model's VAE (selected by ``load_latent_upscaler``).

Both models use the same Qwen-Image VAE (z_dim=16, spatial compression 8), so
``init_latents`` produces and ``finalize_latent`` returns the canonical latent
format ``[B, C, H, W]`` (4D). The VAE's ``decode_to_pixels`` accepts that
directly (the VAE is 2D / single-frame; it no longer adds a frame axis).
Model-internal reshaping (e.g.
Anima's frame axis, Krea2's patchify) lives in ``prepare_latent``/``finalize_latent``
and runs ONCE around the loop, so the per-step ``denoise_step`` never re-converts
the latent.

The shared pipeline integrates the model's velocity by dispatching to a solver
sampler (``euler`` or ``er_sde``) selected by name. Each sampler calls
``denoise_step`` exactly once per schedule step.

Post-processing runs on the decoded pixels as an fp32 GPU tensor (bf16's ~7-bit
mantissa causes banding in image filters); the tensor is only cast to uint8 and
moved to CPU inside ``_to_pil``. Metadata that must live on the final PNG is a
separate concern and is added later, after the PIL conversion.

Upscaling has two modes, driven by ``upscale_factor`` (f in (0.0, 8.0]) and
``upscale_type``:

  * ``refined`` (default): the latent (Sesqui) upscaler runs ``UPSCALE_SCALE``x
    in latent space followed by a low-strength refine denoise. For f in (1, 2]
    that 2x is the whole upscale; for f in (2, ...] a pixel-domain Real-ESRGAN
    step (scale auto-detected from the model, 2/4/8) is added on the decoded
    image and the result is downscaled to f.
  * ``fast``: only the pixel-domain Real-ESRGAN step runs on the decoded image
    (no latent 2x multiplier), so f is capped at the detected ESRGAN scale.

ESRGAN requires an ``esrgan_path`` (optional CLI ``--esrgan``); without it only
``refined`` factors <= 2 are available. Max factor ranges follow the detected
model scale: ``refined`` up to ``UPSCALE_SCALE * esrgan_scale``, ``fast`` up to
``esrgan_scale``. ESRGAN is fast and is deliberately *not* pipeline-cached —
only the decoded VAE output is cached.

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
  Postprocess    | (not cached — cheap)                    | —
  ESRGAN/resize  | (not cached — fast)                     | —

LoRA switching
---------------
LoRAs are applied per-request via ``switch_loras()``. The base model is loaded
without any LoRA baked in. At request time, the requested LoRA(s) are loaded
from disk and their deltas are added to the model's parameters. When the next
request asks for different LoRAs, the old deltas are subtracted (undo) before
applying the new ones. This avoids reloading the entire model from disk.
"""
from __future__ import annotations

import glob
import os
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from thenoise.upscale import load_latent_upscaler, load_esrgan, detect_esrgan_scale
from thenoise.samplers import Step, create_sampler
from thenoise.samplers.euler import EulerSampler
from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model
from thenoise.utils.lora import LoRAApplyResult
from thenoise.utils.pipeline_cache import PipelineCache
from thenoise.utils.safetensors import WRAP_PREFIXES
from thenoise.postprocess.film_grain import film_grain
from thenoise.postprocess.nyquist import nyquist_notch
from thenoise.postprocess.rcas import rcas
from thenoise.utils.png import build_pnginfo


@dataclass
class Conditioning:
    """Bundle of (un)conditional embeddings produced by ``encode_prompt``.

    ``null``/``null_mask`` are ``None`` when guidance is off (CFG disabled), so
    ``denoise_step`` can skip the unconditional forward.
    """

    cond: torch.Tensor
    cond_mask: Optional[torch.Tensor] = None
    null: Optional[torch.Tensor] = None
    null_mask: Optional[torch.Tensor] = None


# Generic wrapper prefixes that repackagings (e.g. ComfyUI's "diffusion_model"
# export) prepend to *every* tensor name. Detection must strip these before
# matching on an architecture signature, otherwise a repackaged checkpoint is
# misidentified. The canonical list lives in ``thenoise.utils.safetensors``
# (shared with ``strip_wrap_prefixes`` used at load time) so detection and
# loading stay in sync.


def normalize_keys(keys):
    """Yield tensor names with any generic wrapper prefix stripped.

    Repackaged checkpoints often prefix every key with a shared wrapper such as
    ``model.diffusion_model.`` (ComfyUI) or ``net.``. Stripping it lets each
    ``detect`` match on the model's *own* distinctive key paths regardless of
    the wrapper, so raw and repackaged checkpoints resolve identically.
    """
    for k in keys:
        for p in WRAP_PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
                break
        yield k


def _center_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop ``image`` to ``(width, height)``."""
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


class DiffusionModel(ABC):
    """Base class for model adapters. Subclasses must set ``name``."""

    name: str = ""

    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    DEFAULT_STEPS = 28
    DEFAULT_GUIDANCE_SCALE = 0.0

    SAMPLER = "er_sde"

    # Canonical latent geometry (shared Qwen-Image VAE).
    LATENT_CHANNELS = 16
    _VAE_SCALE = 8

    UPSCALE_SCALE = 2
    REFINE_STEPS = 1
    REFINE_DENOISE = 0.1

    @staticmethod
    @abstractmethod
    def detect(f) -> bool:
        """Return True if the open safetensors handle ``f`` is this model's DiT."""

    def __init__(
        self,
        *,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_dir: Optional[str] = None,
        esrgan_path: Optional[str] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.lora_dir = lora_dir
        self.esrgan_path = esrgan_path
        self._lock = threading.Lock()

        torch._dynamo.config.recompile_limit = 64

        # LoRA state: cached LoRA state dicts for clean switching.
        # Stores small rank-reduced factors instead of full-sized delta tensors.
        self._active_lora_result: Optional[LoRAApplyResult] = None
        self._active_lora_spec: Optional[str] = None

        # Pipeline result cache (single-entry per stage, on-device).
        # Uses PipelineCache for cascade invalidation with immediate release
        # of downstream tensors when any stage is invalidated.
        self._cache = PipelineCache()

        # Lazy latent upscaler (only loaded if ``upscale`` is requested).
        # ``_upscale_format`` supplies the latent-format name matching the VAE.
        self._upscaler = None
        self._adaptor = None

        # Lazy pixel-domain Real-ESRGAN upscaler (only if ``esrgan_path`` set).
        self._esrgan = None
        self._esrgan_scale_val: Optional[int] = None  # detected scale, cached

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        guidance_scale: float,
    ) -> Conditioning:
        """Tokenize + encode prompt (and negative) into conditioning."""

    @abstractmethod
    def init_latents(self, height: int, width: int, seed: int) -> torch.Tensor:
        """Seed the canonical 4D latent ``[B, C, H//8, W//8]``."""

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Canonical -> model-internal latent. Runs ONCE before the loop.

        Override for reshaping (e.g. Krea2 patchify, Anima frame axis); default
        is the identity (canonical == internal).
        """
        return latents

    @abstractmethod
    def schedule(self, steps: int, height: int, width: int) -> list[Step]:
        """Build the model's denoising schedule (one ``Step`` per iteration)."""

    @abstractmethod
    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One DiT forward (+ CFG) returning the velocity in internal form."""

    def finalize_latent(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Model-internal -> canonical 4D latent. Runs ONCE after the loop.

        Override to invert ``prepare_latent``; default is the identity.
        """
        return latents

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Return the effective (width, height). Override to round/validate."""
        return width, height


    # --------------------------------------------------------------- LoRA
    def _parse_lora_spec(self, spec: str) -> Tuple[str, float]:
        """Parse a 'filename:weight' spec into (filename, weight).

        Auto-appends .safetensors
        """
        if ":" in spec:
            filename, weight_str = spec.rsplit(":", 1)
            weight = float(weight_str)
        else:
            filename = spec
            weight = 1.0

        filename = filename + ".safetensors"

        return filename, weight

    def _resolve_lora_path(self, filename: str) -> str:
        """Resolve a LoRA filename to an absolute path, guarded against traversal.

        Subdirectories are allowed, but .. components that would escape lora_dir
        raise ValueError.
        """
        if not self.lora_dir:
            raise ValueError("lora_dir is not set")

        base = os.path.abspath(self.lora_dir)
        candidate = os.path.abspath(os.path.join(self.lora_dir, filename))

        if not candidate.startswith(base + os.sep) and candidate != base:
            raise ValueError("LoRA path escapes lora_dir")

        return candidate

    def _get_lora_sd(self, filename: str) -> Dict[str, torch.Tensor]:
        """Load a LoRA state dict from disk."""
        filepath = self._resolve_lora_path(filename)

        logger = __import__("logging").getLogger(__name__)
        logger.info("Loading LoRA: %s", filepath)
        return load_file(filepath, device=self.device)

    def _make_lora_spec_hash(self, lora_specs: Optional[List[str]]) -> str:
        """Create a hash string for the current LoRA configuration."""
        if not lora_specs:
            return "__none__"
        return "|".join(sorted(lora_specs))

    def switch_loras(
        self,
        lora_specs: Optional[List[str]],
        dit: torch.nn.Module,
    ) -> None:
        """Switch active LoRAs on the DiT module (in-place, under the lock).

        Args:
            lora_specs: list of "filename:weight" strings, or None for base model.
            dit: the DiT model module whose parameters will be modified.

        Skips the switch if the requested config matches the current one.
        """
        new_spec = self._make_lora_spec_hash(lora_specs)
        if new_spec == self._active_lora_spec:
            return  # no-op: same LoRA config

        logger = __import__("logging").getLogger(__name__)

        # Undo any currently active LoRA
        if self._active_lora_result is not None:
            logger.debug("Undoing previous LoRA config")
            undo_lora_on_model(dit, self._active_lora_result, torch.device(self.device))
            self._active_lora_result = None

        # Apply new LoRAs
        if lora_specs and self.lora_dir is not None:
            lora_sds = []
            multipliers = []
            for spec in lora_specs:
                filename, weight = self._parse_lora_spec(spec)
                lora_sds.append(self._get_lora_sd(filename))
                multipliers.append(weight)

            self._active_lora_result = apply_lora_to_model(
                dit, lora_sds, multipliers, torch.device(self.device)
            )
            active_names = ", ".join(
                self._parse_lora_spec(s)[0] for s in lora_specs
            )
            logger.info("Applied LoRA(s): %s", active_names)
        else:
            logger.debug("Using base model (no LoRA)")

        self._active_lora_spec = new_spec

    def list_loras(self) -> List[str]:
        """List available LoRA names relative to lora_dir.

        Subdirectories are scanned recursively. Names are relative paths with the
        .safetensors suffix stripped (e.g. "12345_something" or "sub/style"), so
        they can be used directly as lora_specs (which auto-appends the suffix).
        """
        if not self.lora_dir:
            return []
        names = []
        for root, _dirs, files in os.walk(self.lora_dir):
            for name in sorted(files):
                if not name.endswith(".safetensors"):
                    continue
                rel = os.path.relpath(os.path.join(root, name), self.lora_dir)
                names.append(rel[: -len(".safetensors")])
        return sorted(names)

    def percent_to_sigma(self, percent: float) -> float:
        """Map a percent (0..1) to a sigma, used by the sampler's SNR offset.

        The ER-SDE solver needs ``sigma`` just below 1 (its ``sigma/(1-sigma)``
        blows up at exactly 1). Flow models override this with their shift
        the default is a linear fallback."""
        return 1.0 - percent

    # ---------------------------------------------------------- pipeline cache
    def _cache_key_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        lora_specs: Optional[List[str]] = None,
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

        Embeds the prompt key so any prompt/guidance/LoRA change cascades..
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
            self.UPSCALE_SCALE,
            self.REFINE_STEPS,
            self.REFINE_DENOISE,
        )

    # ------------------------------------------------------------ pipeline
    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        upscale: bool = False,
        upscale_factor: float = 1.0,
        upscale_type: str = "refined",
        sampler: Optional[str] = None,
        qwen_vae_enhance: bool = False,
        film_grain: float = 0.0,
        sharpening: float = 0.0,
        lora_specs: Optional[List[str]] = None,
    ) -> Image.Image:
        """Encode -> denoise -> decode -> postprocess. Returns a single PIL image.

        Each pipeline stage is cached (single-entry). Cache keys are computed from
        *resolved* parameters so that defaults are accounted for. A change at any
        stage invalidates that stage and all downstream stages automatically via
        the nested key structure.
        """
        # --- resolve defaults (cache keys must use actual resolved values) ---
        width = width or self.DEFAULT_WIDTH
        height = height or self.DEFAULT_HEIGHT
        steps = steps or self.DEFAULT_STEPS
        guidance_scale = (
            self.DEFAULT_GUIDANCE_SCALE
            if guidance_scale is None
            else guidance_scale
        )

        # Resolve upscale parameters: ``upscale`` is a legacy alias for a 2x
        # refined upscale; an explicit factor/type overrides it.
        if upscale and upscale_factor == 1.0:
            upscale_factor = float(self.UPSCALE_SCALE)
        factor, upscale_type = self._resolve_upscale(upscale_factor, upscale_type)
        width, height = self.resolve_size(width, height)
        target_width = width
        target_height = height
        if factor != 1.0:
            target_width = round(width * factor)
            target_height = round(height * factor)
        refined = upscale_type == "refined" and factor > 1.0
        esrgan_scale = self._esrgan_scale_for(factor, upscale_type)
        effective_sampler = sampler or self.SAMPLER

        # seed=-1 is treated as "random" (same as None)
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # --- compute cache keys (pure data, no model access) ---
        prompt_key = self._cache_key_prompt(
            prompt, negative_prompt, guidance_scale, lora_specs
        )
        sampling_key = self._cache_key_sampling(
            prompt_key, width, height, steps, seed, effective_sampler
        )
        decode_key = self._cache_key_decode(sampling_key, refined)

        # --- locked section: cache checks + model access ---
        with self._lock:
            self.switch_loras(lora_specs, self.dit)

            # Stage 1: prompt conditioning
            if self._cache.prompt_hit(prompt_key):
                cond = self._cache.prompt_get()
            else:
                cond = self.encode_prompt(
                    prompt, negative_prompt, guidance_scale=guidance_scale
                )
                self._cache.prompt_store(prompt_key, cond)

            # Stage 2: sampling (denoise)
            if self._cache.sampling_hit(sampling_key):
                latents = self._cache.sampling_get()
            else:
                with torch.no_grad():
                    latents = self._denoise(
                        cond, steps, height, width, seed,
                        guidance_scale, effective_sampler,
                    )
                self._cache.sampling_store(sampling_key, latents)

            # Stage 3/4: upscale + decode (interleaved so cache hits skip upscale)
            if self._cache.decode_hit(decode_key):
                pixels = self._cache.decode_get()
            else:
                # Cache miss — run latent upscale+refine (if refined) then decode
                if refined:
                    latents = self._upscale_and_refine(
                        latents, cond, steps, height, width, seed, guidance_scale
                    )
                pixels = self.decode(latents)  # fp32 GPU tensor [C,H,W]
                self._cache.decode_store(decode_key, pixels)

            # Qwen notch filter must run immediately after VAE decode (before any
            # pixel-domain upscaling), so the 2px grid pattern is removed at its
            # native resolution.
            if qwen_vae_enhance:
                pixels = nyquist_notch(pixels)

            # Pixel-domain ESRGAN (fast, not cached) + GPU resize to target size.
            pixels = self._esrgan_step(pixels, esrgan_scale)
            pixels = self._resize_to_target(pixels, target_width, target_height)

            # Stage 5: postprocess (cheap — not cached)
            pixels = self.postprocess(
                pixels,
                film_grain_strength=film_grain,
                sharpening=sharpening,
            )
            image = self._to_pil(pixels)

            if (image.width, image.height) != (target_width, target_height):
                image = _center_crop(image, target_width, target_height)

            # Attach PNG metadata
            pnginfo = build_pnginfo(
                model=self.name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                upscale=upscale,
                upscale_factor=factor,
                upscale_type=upscale_type,
                sampler=effective_sampler,
                qwen_vae_enhance=qwen_vae_enhance,
                film_grain=film_grain,
                sharpening=sharpening,
                lora_specs=lora_specs,
            )
            image._pnginfo = pnginfo
            return image

    def _denoise(
        self,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
        seed: int,
        guidance_scale: float,
        sampler: str,
    ) -> torch.Tensor:
        """Shared denoising pipeline over the model's ``schedule``.

        Builds the latent and schedule, then delegates the loop to the selected
        solver sampler (``euler`` or ``er_sde``). Each sampler calls
        ``denoise_step`` once per schedule step and runs integration in fp32.
        """
        sampler = sampler or self.SAMPLER
        solver = create_sampler(sampler, self)

        latents = self.init_latents(height, width, seed)
        x = self.prepare_latent(latents, cond, steps, height, width)
        schedule = self.schedule(steps, height, width)
        x = solver.sample(x, schedule, cond, guidance_scale, seed)
        return self.finalize_latent(x, height, width)

    # ------------------------------------------------------------- upscaling
    @abstractmethod
    def _upscale_format(self) -> str:
        """Return the latent-format name matching this model's VAE.

        Concrete subclasses must override this to return the name of their VAE's
        latent format (e.g. ``"wan21"`` for the shared Qwen-Image VAE). It is
        passed to ``load_latent_upscaler``, which selects the adaptor and weight file.
        """
        ...

    def _load_latent_upscaler(self):
        """Load the latent upscaler (once, lazily, under the lock)."""
        if self._upscaler is None:
            self._upscaler, self._adaptor = load_latent_upscaler(
                self._upscale_format(),
                device=self.device,
                dtype=self.dtype,
            )
        return self._upscaler, self._adaptor

    # ------------------------------------------------------------- upscale plan
    def _resolve_upscale(
        self,
        factor: float,
        upscale_type: str,
    ) -> tuple[float, str]:
        """Validate and return the effective (factor, type).

        ``upscale_factor`` must be in (0.0, 8.0]. ``fast`` mode has no latent
        2x multiplier so it is capped at 4.0. ESRGAN (required for ``fast`` and
        for ``refined`` factors above the latent 2x) must have been configured
        via ``esrgan_path``.
        """
        if upscale_type not in ("refined", "fast"):
            raise ValueError(
                f"upscale_type must be 'refined' or 'fast', got {upscale_type!r}"
            )
        if not 0.0 < factor <= 8.0:
            raise ValueError("upscale_factor must be in (0.0, 8.0]")
        esrgan = self._esrgan_scale  # detected scale (0 = no ESRGAN configured)
        if esrgan == 0:
            # No ESRGAN: only ``refined`` factors up to the latent 2x work.
            if factor > self.UPSCALE_SCALE:
                raise ValueError(
                    f"upscale_factor > {self.UPSCALE_SCALE} requires a "
                    "Real-ESRGAN model; pass --esrgan PATH (or run "
                    "scripts/download_esrgan.py)"
                )
            if upscale_type == "fast" and factor > 1.0:
                raise ValueError(
                    "upscale_type='fast' requires a Real-ESRGAN model; "
                    "pass --esrgan PATH (or run scripts/download_esrgan.py)"
                )
        else:
            # Max factor depends on the detected model scale: refined gets the
            # latent 2x multiplier, fast does not.
            max_refined = self.UPSCALE_SCALE * esrgan
            if upscale_type == "refined" and factor > max_refined:
                raise ValueError(
                    f"upscale_type='refined' with a {esrgan}x ESRGAN model is "
                    f"limited to factor {max_refined}"
                )
            if upscale_type == "fast" and factor > esrgan:
                raise ValueError(
                    f"upscale_type='fast' with a {esrgan}x ESRGAN model is "
                    f"limited to factor {esrgan}"
                )
        return factor, upscale_type

    @property
    def _esrgan_scale(self) -> int:
        """Detected scale of the configured ESRGAN model (0 if none), cached.

        Reads only the safetensors header on first use; the value is then reused.
        Tests may set ``_esrgan_scale_val`` directly to skip file access.
        """
        if self._esrgan_scale_val is None:
            if not self.esrgan_path:
                self._esrgan_scale_val = 0
            else:
                self._esrgan_scale_val = detect_esrgan_scale(self.esrgan_path)
        return self._esrgan_scale_val

    def _esrgan_scale_for(self, factor: float, upscale_type: str) -> int:
        """Pixel-domain ESRGAN scale for a (factor, type), or 0 to skip.

        ``refined`` gets a 2x from the latent path, so ESRGAN is only needed when
        the factor exceeds that 2x. ``fast`` has no latent multiplier and always
        needs ESRGAN for any upscale. Uses the detected model scale.
        """
        esrgan = self._esrgan_scale
        if esrgan == 0 or factor <= 1.0:
            return 0
        if upscale_type == "fast":
            return esrgan
        return esrgan if factor > self.UPSCALE_SCALE else 0

    def _load_esrgan(self):
        """Load the pixel-domain ESRGAN model (once, lazily, under the lock)."""
        if self._esrgan is None:
            if not self.esrgan_path:
                raise ValueError(
                    "no ESRGAN model configured; pass --esrgan PATH "
                    "(or run scripts/download_esrgan.py)"
                )
            self._esrgan, scale = load_esrgan(self.esrgan_path, device=self.device)
            self._esrgan_scale_val = scale
        return self._esrgan

    def _esrgan_step(self, pixels: torch.Tensor, scale: int) -> torch.Tensor:
        """Apply the pixel-domain ESRGAN upscale by ``scale``x (if > 0).

        The Real-ESRGAN model operates on RGB in [0, 1] (see its ``enhance``
        path), while the pipeline's decoded pixels are in [-1, 1]. Convert to
        [0, 1] before the model and back to [-1, 1] afterwards so downstream
        postprocessing / ``_to_pil`` stay unchanged.
        """
        if not scale:
            return pixels
        model = self._load_esrgan()
        x = (pixels.unsqueeze(0) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        with torch.no_grad():
            out = model.forward_tiled(x)
        out = out * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return out[0]

    @staticmethod
    def _resize_to_target(
        pixels: torch.Tensor, target_w: int, target_h: int
    ) -> torch.Tensor:
        """GPU bilinear resize of ``[C, H, W]`` to the target size (no-op if equal)."""
        c, h, w = pixels.shape
        if (w, h) == (target_w, target_h):
            return pixels
        with torch.no_grad():
            return F.interpolate(
                pixels.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )[0]


    def _upscale_and_refine(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
        seed: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Upscale the canonical latent ``UPSCALE_SCALE``x in latent space, then
        run a short low-strength refine denoise at the new size.

        Sesqui operates on raw VAE latents; the adaptor converts the canonical
        latent to/from that raw space. The refined result is the canonical latent
        at the upscaled spatial size, ready for ``decode``.
        """
        upscaler, adaptor = self._load_latent_upscaler()
        scale = self.UPSCALE_SCALE
        z = latents.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            # Adaptor math in fp32; the model runs in bf16.
            raw = adaptor.to_vae_latent(z).to(self.dtype)
            h, w = z.shape[-2:]
            raw_up = upscaler(raw, (scale * h, scale * w))
            z_up = adaptor.from_vae_latent(raw_up.float()).to(self.dtype)

        # One short low-strength refine denoise at the upscaled size. The refine
        # runs on an independent schedule (see ``_refine``), so the original
        # ``steps`` count is deliberately NOT forwarded.
        return self._refine(
            z_up, cond, scale * height, scale * width, seed, guidance_scale
        )

    def _refine(
        self,
        z: torch.Tensor,
        cond: Conditioning,
        height: int,
        width: int,
        seed: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """One low-strength refine denoise step on an already-clean latent."""
        refine_steps = self.REFINE_STEPS
        denoise = self.REFINE_DENOISE
        new_steps = int(refine_steps / denoise)  # int(1/0.1) = 10

        # Last ``refine_steps`` steps of an independent ``new_steps`` schedule.
        full = self.schedule(new_steps, height, width)
        sub = full[-refine_steps:]
        strength = float(sub[0].t)  # sigma_hat == the step's timestep

        # ComfyUI CONST noise scaling: x = sigma*noise + (1-sigma)*z.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn_like(z, generator=generator)
        noised = strength * noise + (1.0 - strength) * z

        x = self.prepare_latent(noised, cond, new_steps, height, width)
        solver = EulerSampler(self)
        x = solver.sample(x, sub, cond, guidance_scale, seed, desc="refining")
        return self.finalize_latent(x, height, width)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Shared Qwen-Image VAE decode.

        Accepts the canonical 4D latent ``[B, C, H, W]`` (the VAE is 2D /
        single-frame) and returns pixels ``[C, H, W]`` in [-1, 1] as an fp32
        GPU tensor, ready for tensor post-processing.
        """
        dev = torch.device(self.device)
        with torch.no_grad():
            pixels = self.vae.decode_to_pixels(latents.to(dev, dtype=self.vae.dtype))
        if pixels.ndim == 5:  # [B, C, 1, H, W] -> [B, C, H, W]
            pixels = pixels.squeeze(2)
        pixels = pixels.to(torch.float32)
        return pixels[0]  # [C, H, W] in [-1, 1]

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
            pixels = film_grain(pixels, strength=film_grain_strength/10.0)
        return pixels

    @staticmethod
    def _to_pil(pixels: torch.Tensor) -> Image.Image:
        x = torch.clamp(pixels, -1.0, 1.0)
        x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
        x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C
        return Image.fromarray(x)
