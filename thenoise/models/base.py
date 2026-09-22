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

The VAE also owns the pixel width (``pixel_channels``: 3 for the RGB family, 4 for
the RGBA Qwen-Image 2.1 one) and the adapter just reports it, so the pipeline can
ask for the right number of channels at the input boundary and hand the decoded
ones through to the PNG.

Every adapter works on the canonical latent format ``[B, C, H, W]`` (4D), which is
simply the VAE's own output format: ``C = vae.z_dim``, one latent cell per
``vae.spatial_compression`` pixels (16ch/8x for the shared Qwen-Image VAE, 128ch/16x
for the packed Flux.2 one). The geometry therefore lives on the VAE, never on the
adapter: a DiT's own input width is a different number (it patchifies the latent
further) and re-declaring the latent's shape is how the two drift apart.
``init_latents`` produces and ``finalize_latent`` returns that format, which the
VAE's ``decode_to_pixels`` accepts directly (the VAE is 2D / single-frame; it no
longer adds a frame axis).
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
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import load_file

from thenoise.dit.kvcache import KVCache
from thenoise.memory import MemoryManager
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.utils.device import get_device_memory
from thenoise.samplers import Step
from thenoise.upscale import load_latent_upscaler

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image
from thenoise.utils.model_dir import (
    ensure_safetensors,
    resolve_in_dir,
    list_safetensors,
)
from thenoise.utils.checkpoint import detect_checkpoint_prefs
from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model
from thenoise.utils.lora import LoRAApplyResult
from thenoise.utils.safetensors import unwrap_key

logger = logging.getLogger(__name__)

# Fraction of the compute device's VRAM that the resident *weights* (DiT + text
# encoder + VAE) may occupy while staying resident (``offload == load``, no moves).
# The remaining fraction is deliberately conservative headroom for the activation peak
# (denoise, and especially VAE decode at upscaled resolutions) plus the pipeline
# cache. This is a conservative value. The offload device can be forced via --offload-device
_RESIDENT_VRAM_FRACTION = 0.6


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


# Generic wrapper prefixes (``model.diffusion_model.`` / ``net.``) that repackagings
# prepend to *every* tensor name. Detection must strip these before matching on an
# architecture signature, otherwise a repackaged checkpoint is misidentified. The
# stripping itself is shared with loading via ``safetensors.unwrap_key`` so the two
# can never drift apart.


def normalize_keys(keys):
    """Yield tensor names with any generic wrapper prefix stripped.

    Repackaged checkpoints often prefix every key with a shared wrapper such as
    ``model.diffusion_model.`` (ComfyUI) or ``net.``. Stripping it lets each
    ``detect`` match on the model's *own* distinctive key paths regardless of
    the wrapper, so raw and repackaged checkpoints resolve identically.
    """
    for k in keys:
        yield unwrap_key(k)


class DiffusionModel(ABC):
    """Base class for model adapters. Subclasses must set ``name``."""

    name: str = ""

    # Generation preferences and their model default — layer 3 of the precedence
    # implemented by ``pref``: 1. explicit request (API/CLI), 2. what the loaded
    # checkpoint's markers imply (``checkpoint_prefs``), 3. these defaults.
    # Adapters override just the entries they differ on, e.g.
    #
    #     DEFAULT_PREFS = {**DiffusionModel.DEFAULT_PREFS, "steps": 4}
    #
    # so a preference added later keeps its base default instead of being dropped.
    # Asking for an unlisted name is a bug, so ``pref`` raises rather than silently
    # defaulting.
    DEFAULT_PREFS: ClassVar[Dict[str, Any]] = {
        "width": 1024,
        "height": 1024,
        "steps": 28,
        # CFG scale; <= 1.0 disables the unconditional forward.
        "guidance_scale": 0.0,
        # Default solver (see ``thenoise.samplers.SAMPLERS``).
        "sampler": "er_sde",
        # Reference conditioning method for editing.
        "ref_method": "index",
        # Reference-latent KV cache (edit only).
        "kv_cache": False,
    }

    UPSCALE_SCALE = 2
    REFINE_STEPS = 1
    REFINE_DENOISE = 0.1

    # Model capabilities — which optional generation features this adapter actually
    # implements. Adapters override just the entries they differ on, e.g.
    #
    #     CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True}
    #
    # so a capability added later keeps its base default instead of being dropped
    # (the same layering as ``DEFAULT_PREFS``). Asking for an unlisted name is a bug,
    # so ``capability`` raises rather than answering ``False``: silently disabling a
    # feature is indistinguishable from a model that genuinely lacks it. The dict is
    # reported verbatim by ``/health`` so the UI can gate its controls.
    CAPABILITIES: ClassVar[Dict[str, bool]] = {
        # Reference-latent editing: image + instruction -> edited image. An adapter
        # that sets it also overrides ``encode_reference``/``pack_reference_latent``.
        "edit": False,
        # Reference-latent KV cache: freeze the reference tokens' K/V across denoise
        # steps (ComfyUI ``FluxKVCache``) through the shared ``start_kv_caches`` /
        # ``kv_cache`` / ``end_kv_caches`` protocol. Validity is a separate question:
        # the frozen K/V stay step-invariant only under ``index_timestep_zero``.
        "kv_cache": False,
    }

    # The run's caches, created by ``start_kv_caches`` (``prepare_latent``) and
    # dropped by ``end_kv_caches`` (``finalize_latent``). The empty class default
    # keeps ``kv_cache`` answering ``None`` on adapters built without ``__init__``.
    _kv_caches: Optional[Dict[str, KVCache]] = None

    # Preferences implied by the loaded checkpoint's markers; replaced per instance
    # in ``__init__``. The empty class default keeps ``pref`` working on instances
    # built without ``__init__`` (tests, stubs).
    checkpoint_prefs: Dict[str, Any] = {}

    # The sub-projection stackings this model's modules use, as a
    # ``{fused: parts}`` spec (``FUSE_QKV``/``FUSE_GATE_UP`` in
    # ``thenoise.utils.lora``): a LoRA trained on the separate part names gets
    # fused onto the fused module before matching. Default: nothing is fused.
    lora_fusions: Dict[str, Tuple[str, ...]] = {}

    # Which end of the attention sequence this model's KV cache freezes: "suffix"
    # for a ``text, target, references`` layout, "prefix" for ``text + references,
    # target``. Passed to ``thenoise.dit.kvcache``.
    KV_CACHED_SLICE: ClassVar[str] = "suffix"

    def _lora_key_map(self, key: str) -> str:
        """Map a LoRA key to this model's schema.

        Training tools name LoRA targets differently from this repo's model
        schema (e.g. ComfyUI Flux.2 ``transformer_blocks``/``attn`` vs the
        repo's ``double_blocks``/``img_attn``, diffusers ``to_out.0`` vs the
        model's ``out``). Model families with a non-canonical schema override
        this. Default: identity (the generic naming conventions in
        ``thenoise.utils.lora`` already resolve sd-scripts / diffusers names).
        """
        return key

    @staticmethod
    @abstractmethod
    def detect(f) -> bool:
        """Return True if the open safetensors handle ``f`` is this model's DiT."""

    def __init__(self, *, config: ModelConfig):
        self.device = config.device
        self.offload_device = config.offload_device or self._detect_offload_device(config)
        self.dtype = config.dtype
        self.dit_path = config.dit_path
        self.vae_path = config.vae_path
        self.text_encoder_path = config.text_encoder_path
        self.lora_dir = config.lora_dir

        # Layer 2 of the preference precedence: the preferences implied by markers
        # in the DiT header (empty when it carries none). Model-agnostic by
        # construction — see ``thenoise.utils.checkpoint``.
        self.checkpoint_prefs = detect_checkpoint_prefs(config.dit_path)

        # Component placement: subclasses register ``dit`` / ``text_encoder`` /
        # ``vae``; the pipeline controller ensures/offloads them by name.
        self.memory = MemoryManager(self.device, self.offload_device)

        torch._dynamo.config.recompile_limit = 64

        # LoRA state: cached LoRA state dicts for clean switching.
        # Stores small rank-reduced factors instead of full-sized delta tensors.
        self._active_lora_result: Optional[LoRAApplyResult] = None
        self._active_lora_spec: Optional[str] = None

        # Lazy latent upscaler (only loaded if upscale is requested).
        # ``_upscale_format`` supplies the latent-format name matching the VAE.
        self._upscaler = None
        self._adaptor = None

    # ------------------------------------------------------------ preferences
    def pref(self, name: str, request_value: Any = None) -> Any:
        """Resolve a generation preference: request > checkpoint marker > model default.

        ``request_value`` is what the API/CLI carried, or ``None`` when the user did
        not ask (that is how "auto" is represented on the wire). Markers only fill
        in what the user did not ask for — an explicit request always wins, since
        unmarked-but-trained checkpoints (and LoRAs that change what a checkpoint
        expects) are common enough that vetoing on a missing marker would misfire.
        """
        if name not in self.DEFAULT_PREFS:
            raise KeyError(f"unknown preference {name!r}; known: {sorted(self.DEFAULT_PREFS)}")
        if request_value is not None:
            value, source = request_value, "request"
        else:
            detected = self.checkpoint_prefs.get(name)
            if detected is not None:
                value, source = detected, "checkpoint"
            else:
                value, source = self.DEFAULT_PREFS[name], "model default"
        logger.debug("%s = %s (%s)", name, value, source)
        return value

    # ------------------------------------------------------------ capabilities
    def capability(self, name: str) -> bool:
        """True when this adapter implements capability ``name`` (see ``CAPABILITIES``).

        Raises on an unlisted name, mirroring ``pref``: a typo in a capability check
        must not quietly read as "this model cannot do it".
        """
        if name not in self.CAPABILITIES:
            raise KeyError(f"unknown capability {name!r}; known: {sorted(self.CAPABILITIES)}")
        return self.CAPABILITIES[name]

    # ------------------------------------------------------------ devices
    def _detect_offload_device(self, config: ModelConfig) -> str:
        """Pick an offload device from safetensors size vs VRAM (or ``device``).

        The expected resident bytes are estimated from the combined sizes of the three
        checkpoint files (not 100%% accurate, close enough). If they fit the compute
        device's VRAM with ``_RESIDENT_VRAM_FRACTION`` headroom left over for
        activations we stay resident (``offload == load`` -> no moves); otherwise
        we offload to CPU.
        """
        total_vram = get_device_memory(config.device)
        if total_vram is None:
            return config.device
        resident = sum(
            self._file_size(p)
            for p in (config.dit_path, config.vae_path, config.text_encoder_path)
        )
        if resident <= _RESIDENT_VRAM_FRACTION * total_vram:
            return config.device
        return "cpu"

    @staticmethod
    def _file_size(path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def encode_prompt(self, args: EncodePromptArgs) -> "Conditioning":
        """Tokenize + encode prompt (and negative) into RAW conditioning.

        Text-encoder only (the DiT is NOT needed here). Accepts a single
        ``EncodePromptArgs`` struct (prompt, negative_prompt, guidance_scale,
        image) so new knobs never change the signature. ``image`` is only set in
        the edit path (models with the ``edit`` capability); multimodal encoders feed it as
        vision tokens in addition to any reference latent. Returns the raw
        conditioning, transformed into the model-internal conditioning by
        ``fuse_text`` (which runs with the DiT resident).
        """

    def fuse_text(self, cond: "Conditioning") -> "Conditioning":
        """DiT-side text fusion: raw conditioning -> model-internal conditioning.

        Runs with the DiT resident (inside the dit block), after the text encoder
        has been offloaded. Default is the identity for models whose prompt
        conditioning is already the final form (e.g. FluxKlein, ZImage); models
        that fuse the text stream through the DiT (Anima's
        ``_preprocess_text_embeds``, Krea2's ``fuse_text``) override it.
        """
        return cond

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

    # ------------------------------------------------------- reference KV cache
    def start_kv_caches(
        self,
        params: SamplingParams,
        has_reference: bool,
        has_uncond: bool,
    ) -> None:
        """Create this run's K/V caches, one per conditioning branch. Call in ``prepare_latent``.

        Fresh per run so a later request can never reuse a stale cache, and only for
        an edit run on a model implementing the fill/read protocol. One cache per
        branch because the branches can have different token counts (the uncond
        prompt is usually shorter), so their buffers cannot be shared; the uncond
        cache is only created when CFG is actually active.
        """
        if not (params.kv_cache and has_reference and self.capability("kv_cache")):
            self._kv_caches = None
            return
        caches = {"cond": KVCache("cond", self.KV_CACHED_SLICE)}
        if has_uncond:
            caches["uncond"] = KVCache("uncond", self.KV_CACHED_SLICE)
        self._kv_caches = caches

    def kv_cache(self, branch: str) -> Optional[KVCache]:
        """The cache of one conditioning branch (``cond`` / ``uncond``), else ``None``.

        ``None`` means the run has no cache at all (plain t2i, ``kv_cache`` off, or a
        model without support), which is the DiT's uncached path.
        """
        return None if self._kv_caches is None else self._kv_caches.get(branch)

    def end_kv_caches(self) -> None:
        """Drop the run's caches. Call in ``finalize_latent``, before the VAE decode.

        The controller offloads the DiT right after that, so releasing the cache
        here avoids holding its frozen K/V (easily GBs) through the decode.
        """
        self._kv_caches = None

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
                dit, lora_sds, multipliers, torch.device(self.device),
                dit_path=self.dit_path,
                key_map=self._lora_key_map,
                fusions=self.lora_fusions,
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

    # ------------------------------------------------------------ pixel format
    @property
    def pixel_channels(self) -> int:
        """Pixel channels this model's VAE consumes and emits (3 = RGB, 4 = RGBA)."""
        return getattr(getattr(self, "vae", None), "pixel_channels", 3)

    # ------------------------------------------------------------ decode
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Shared VAE decode — the final generation step.

        Accepts the canonical 4D latent ``[B, C, H, W]`` (the VAE is 2D /
        single-frame) and returns pixels ``[C, H, W]`` in [-1, 1] as an fp32
        GPU tensor, ready for the controller's postprocessing. ``C`` is the VAE's
        own width, so an RGBA decode's alpha reaches the PNG output.
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
