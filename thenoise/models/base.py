"""Abstract interface for a diffusion model adapter.

The base class concerns itself ONLY with actual generation, finishing at the
final VAE decode step. It owns the model kernels and the load/switch logic that
is inseparable from the model's own weights (the DiT, VAE, text encoder, and
LoRAs). All *pipeline orchestration* — encode -> denoise -> decode -> postprocess
-> PIL, the inference lock, the stage cache, upscale planning, pixel-domain
upscaling, PNG metadata — lives in ``thenoise.pipeline.PipelineController``.
Pixel-domain upscaling (a pixel-space / postprocessing concern that needs no
model) lives in ``thenoise.upscale.pixel.PixelUpscalerManager``.

Subclasses implement the model-specific kernels and load their own VAE:

  * ``detect(f)``            — recognize this model's DiT from a safetensors handle.
  * ``encode_prompt(...)``   — text -> conditioning embeddings (cond + null).
  * ``init_latents(params)`` — seeded noise in the canonical 4D latent format.
  * ``prepare_latent(...)``  — canonical -> model-internal latent (once, pre-loop).
  * ``schedule(params)``     — the model's timestep/step-size schedule.
  * ``denoise_step(...)``    — one DiT forward + CFG, returning a velocity.
  * ``finalize_latent(...)`` — model-internal -> canonical latent (once, post-loop).
  * ``resolve_size(...)``    — per-model size rounding / validation.
  * ``decode(...)``          — canonical latent -> pixels (the generation end).
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

The denoise loop itself (building the latent/schedule and dispatching to a solver
sampler) is orchestrated by the controller via ``thenoise.samplers``; each sampler
calls ``denoise_step`` exactly once per schedule step.

LoRA switching
---------------
LoRAs are applied per-request via ``switch_loras()`` and are a model concern
(they mutate the DiT's parameters). The base model is loaded without any LoRA
baked in. At request time, the requested LoRA(s) are loaded from disk and their
deltas are added to the model's parameters. When the next request asks for
different LoRAs, the old deltas are subtracted (undo) before applying the new
ones. This avoids reloading the entire model from disk.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.samplers import Step
from thenoise.samplers.euler import EulerSampler
from thenoise.upscale import load_latent_upscaler

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image
from thenoise.utils.model_dir import (
    ensure_safetensors,
    resolve_in_dir,
    list_safetensors,
)
from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model
from thenoise.utils.lora import LoRAApplyResult
from thenoise.utils.safetensors import WRAP_PREFIXES

logger = logging.getLogger(__name__)


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
    pooled: Optional[torch.Tensor] = None
    neg_pooled: Optional[torch.Tensor] = None


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

    # Reference-latent editing capability (image + instruction -> edited image).
    # Editing models set this True and override ``encode_reference``/``pack_reference_latent``.
    supports_edit: bool = False

    @staticmethod
    @abstractmethod
    def detect(f) -> bool:
        """Return True if the open safetensors handle ``f`` is this model's DiT."""

    def __init__(self, *, config: ModelConfig):
        self.device = config.device
        self.dtype = config.dtype
        self.dit_path = config.dit_path
        self.vae_path = config.vae_path
        self.text_encoder_path = config.text_encoder_path
        self.lora_dir = config.lora_dir

        torch._dynamo.config.recompile_limit = 64

        # LoRA state: cached LoRA state dicts for clean switching.
        # Stores small rank-reduced factors instead of full-sized delta tensors.
        self._active_lora_result: Optional[LoRAApplyResult] = None
        self._active_lora_spec: Optional[str] = None

        # Lazy latent upscaler (only loaded if upscale is requested).
        # ``_upscale_format`` supplies the latent-format name matching the VAE.
        self._upscaler = None
        self._adaptor = None

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        """Tokenize + encode prompt (and negative) into conditioning.

        Accepts a single ``EncodePromptArgs`` struct (prompt, negative_prompt,
        guidance_scale, image) so new knobs never change the signature. ``image``
        is only set in the edit path (``supports_edit`` models); multimodal
        encoders feed it as vision tokens in addition to any reference latent.
        """

    @abstractmethod
    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        """Seed the canonical 4D latent ``[B, C, H//8, W//8]``."""

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
        ref: Optional[torch.Tensor] = None,
        ref_method: str = "index",
    ) -> torch.Tensor:
        """Canonical -> model-internal latent. Runs ONCE before the loop.

        Override for reshaping (e.g. Krea2 patchify, Anima frame axis); default
        is the identity (canonical == internal).

        ``ref``/``ref_method`` are only passed in the edit path; editing models
        use them to stash the reference tokens+ids that ``denoise_step`` reads.
        """
        return latents

    @abstractmethod
    def schedule(self, params: SamplingParams) -> list[Step]:
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
        params: SamplingParams,
    ) -> torch.Tensor:
        """Model-internal -> canonical 4D latent. Runs ONCE after the loop.

        Override to invert ``prepare_latent``; default is the identity.
        """
        return latents

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Return the effective (width, height). Override to round/validate."""
        return width, height

    def percent_to_sigma(self, percent: float) -> float:
        """Map a percent (0..1) to a sigma, used by the sampler's SNR offset.

        The ER-SDE solver needs ``sigma`` just below 1 (its ``sigma/(1-sigma)``
        blows up at exactly 1). Flow models override this with their shift
        the default is a linear fallback."""
        return 1.0 - percent

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) into the canonical latent.

        Overridden by editing models (uses their VAE encoder)."""
        raise NotImplementedError(f"{self.name} does not support reference editing")

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "index",
        ref_index: int = 1,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Canonical reference latent -> model-internal (tokens, ids).

        ``ref_index`` is the 1-based position among the reference images (used to
        give each ref a distinct t-axis index). Overridden by editing models."""
        return None

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

        filename = ensure_safetensors(filename)

        return filename, weight

    def _resolve_lora_path(self, filename: str) -> str:
        """Resolve a LoRA filename to an absolute path, guarded against traversal.

        Subdirectories are allowed, but .. components that would escape lora_dir
        raise ValueError. Shared path logic lives in ``utils.model_dir``.
        """
        return resolve_in_dir(self.lora_dir, filename)

    def _get_lora_sd(self, filename: str) -> Dict[str, torch.Tensor]:
        """Load a LoRA state dict from disk."""
        filepath = self._resolve_lora_path(filename)

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

        Names are relative paths with the .safetensors suffix stripped (e.g.
        "12345_something" or "sub/style"), so they can be used directly as
        lora_specs (which auto-appends the suffix). Shared listing logic lives
        in ``utils.model_dir``.
        """
        return list_safetensors(self.lora_dir)

    # ------------------------------------------------------- latent upscaler
    @abstractmethod
    def _upscale_format(self) -> str:
        """Return the latent-format name matching this model's VAE.

        Concrete subclasses must override this to return the name of their VAE's
        latent format (e.g. ``"wan21"`` for the shared Qwen-Image VAE). It is
        passed to ``load_latent_upscaler``, which selects the adaptor and weight file.
        """
        ...

    def load_latent_upscaler(self):
        """Load the latent upscaler (once, lazily, under the lock)."""
        if self._upscaler is None:
            self._upscaler, self._adaptor = load_latent_upscaler(
                self._upscale_format(),
                device=self.device,
                dtype=self.dtype,
            )
        return self._upscaler, self._adaptor

    def supports_latent_upscale(self) -> bool:
        """True if this model can run latent (refined) upscaling.

        Defaults to ``_upscale_format()`` succeeding; models that cannot raise
        ``NotImplementedError`` there, so the pipeline degrades refined upscale
        to pixel-only.
        """
        try:
            self._upscale_format()
            return True
        except NotImplementedError:
            return False

    def refine_latents(
        self,
        z: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
    ) -> torch.Tensor:
        """One short low-strength refine denoise on an already-clean latent.

        Flow-model (CONST) default: blend a little noise in and euler over the
        tail of an independent schedule. Discrete-epsilon models (SDXL) override
        this with their EDM noise semantics.
        """
        refine_steps = self.REFINE_STEPS
        denoise = self.REFINE_DENOISE
        new_steps = int(refine_steps / denoise)  # int(1/0.1) = 10

        refine_params = replace(params, steps=new_steps)
        full = self.schedule(refine_params)
        sub = full[-refine_steps:]
        strength = float(sub[0].t)  # flow sigma == the tail step's timestep

        # ComfyUI CONST noise scaling: x = sigma*noise + (1-sigma)*z.
        generator = torch.Generator(device=self.device).manual_seed(params.seed)
        noise = torch.randn_like(z, generator=generator)
        noised = strength * noise + (1.0 - strength) * z

        x = self.prepare_latent(noised, cond, refine_params)
        solver = EulerSampler(self)
        x = solver.sample(
            x, sub, cond, params.guidance_scale, params.seed, desc="refining"
        )
        return self.finalize_latent(x, refine_params)

    # ------------------------------------------------------------ decode
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Shared VAE decode — the final generation step.

        Accepts the canonical 4D latent ``[B, C, H, W]`` (the VAE is 2D /
        single-frame) and returns pixels ``[C, H, W]`` in [-1, 1] as an fp32
        GPU tensor, ready for the controller's postprocessing.
        """
        dev = torch.device(self.device)
        with torch.no_grad():
            pixels = self.vae.decode_to_pixels(latents.to(dev, dtype=self.vae.dtype))
        if pixels.ndim == 5:  # [B, C, 1, H, W] -> [B, C, H, W]
            pixels = pixels.squeeze(2)
        pixels = pixels.to(torch.float32)
        return pixels[0]  # [C, H, W] in [-1, 1]


# Imported lazily to avoid a cycle: samplers import Conditioning/DiffusionModel.

__all__ = ["DiffusionModel", "Conditioning", "normalize_keys"]
