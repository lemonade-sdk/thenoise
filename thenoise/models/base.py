"""Abstract interface for a diffusion model adapter.

The base class owns the model kernels and the load/switch logic for the model's own
weights (DiT, VAE, text encoder, LoRAs), ending at the final VAE decode. Pipeline
orchestration lives in ``thenoise.pipeline.PipelineController``.

Adapters work on the canonical 4D latent ``[B, C, H, W]`` — the VAE's own output
format, ``C = vae.z_dim`` at ``vae.spatial_compression`` pixels per latent cell — so
latent geometry lives on the VAE. ``init_latents`` produces and ``finalize_latent``
returns that format. Model-internal reshaping lives in ``prepare_latent`` /
``finalize_latent``, which run once around the denoise loop, keeping per-step
``denoise_step`` a pure DiT forward.

LoRAs mutate the DiT's parameters, so they are a model concern: they are swapped per
request by ``switch_loras()``.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

from thenoise.dit.kvcache import KVCache
from thenoise.memory import MemoryManager
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.utils.device import get_device_memory
from thenoise.utils.image_tensor import ReferenceSizing
from thenoise.samplers import Step
from thenoise.upscale import LatentUpscaler

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

# Fraction of the compute device's VRAM the resident weights (DiT + text encoder +
# VAE) may occupy while staying resident. The rest is headroom for the activation
# peak (denoise, VAE decode) plus the pipeline cache. ``--offload-device`` forces it.
_RESIDENT_VRAM_FRACTION = 0.6

# What one stage of a request asks of that headroom, for the residency decision
# `MemoryManager.stage` makes before it runs: a part the latency machinery holds at any
# size (tiles, halos, feature caches) plus a part proportional to the pixels.
#
# Both are the worst case across codecs of a least-squares fit to the decode peaks in
# bench-scripts/snapshots/vae-gfx1151-baseline-linux.json, which reproduces every codec to
# 4%; the same fit without the fixed part is off by up to 51%, mostly by under-claiming
# small images. Those peaks include each codec's own weights, so the claim also sits above
# the transient alone: at 2048x2048 it runs 1.0x the widest codec's need and 3.6x the
# cheapest one's. A stage that claims too little gets the device handed back mid-run.
# No denoise peak has been measured, so the sampling loop borrows these two figures.
_STAGE_FLOOR_BYTES = 967 << 20
_STAGE_BYTES_PER_MEGAPIXEL = 1_015_000_000


@dataclass
class Conditioning:
    """Bundle of (un)conditional embeddings produced by ``encode_prompt``.

    ``null``/``null_mask`` are ``None`` when guidance is off.
    """

    cond: torch.Tensor
    cond_mask: Optional[torch.Tensor] = None
    null: Optional[torch.Tensor] = None
    null_mask: Optional[torch.Tensor] = None


def normalize_keys(keys):
    """Yield tensor names with any generic wrapper prefix stripped.

    Repackaged checkpoints prefix every key with a wrapper such as
    ``model.diffusion_model.`` or ``net.``; stripping it lets each ``detect`` match on
    the model's own distinctive key paths. Uses ``safetensors.unwrap_key`` so
    detection and loading cannot drift apart.
    """
    for k in keys:
        yield unwrap_key(k)


class DiffusionModel(ABC):
    """Base class for model adapters. Subclasses must set ``name``."""

    name: str = ""

    # Generation preferences and their model default; ``pref`` resolves the
    # precedence. Adapters override only the entries they differ on, merging so a
    # preference added later keeps its base default:
    #
    #     DEFAULT_PREFS = {**DiffusionModel.DEFAULT_PREFS, "steps": 4}
    #
    DEFAULT_PREFS: ClassVar[Dict[str, Any]] = {
        "width": 1024,
        "height": 1024,
        "steps": 28,
        # CFG scale; <= 1.0 disables the unconditional forward.
        "guidance_scale": 0.0,
        "sampler": "er_sde",
        "ref_method": "index",
        "sigmas": None
    }

    UPSCALE_SCALE = 2
    REFINE_STEPS = 1
    REFINE_DENOISE = 0.25

    # Which optional generation features this adapter implements. Adapters override
    # only the entries they differ on, merging as above. Reported verbatim by
    # ``/health`` so the UI can gate its controls.
    CAPABILITIES: ClassVar[Dict[str, bool]] = {
        # Reference-latent editing: image + instruction -> edited image. Requires
        # overriding ``encode_reference``/``pack_reference_latent``.
        "edit": False,
        # Freeze the reference tokens' K/V across denoise steps through the shared
        # ``start_kv_caches`` / ``kv_cache`` / ``end_kv_caches`` protocol.
        "kv_cache": False,
    }

    # The run's caches: created by ``start_kv_caches``, dropped by ``end_kv_caches``.
    # The empty class default keeps ``kv_cache`` answering ``None`` on adapters built
    # without ``__init__``.
    _kv_caches: Optional[Dict[str, KVCache]] = None

    # Preferences implied by the loaded checkpoint's markers; replaced per instance
    # in ``__init__``, with the empty class default covering adapters built without it.
    checkpoint_prefs: Dict[str, Any] = {}

    # Sub-projection stackings this model's modules use, as a ``{fused: parts}`` spec
    # (``FUSE_QKV``/``FUSE_GATE_UP`` in ``thenoise.utils.lora``): a LoRA trained on
    # the separate part names is fused onto the fused module before matching.
    lora_fusions: Dict[str, Tuple[str, ...]] = {}

    # Which end of the attention sequence this model's KV cache freezes: "suffix"
    # for a ``text, target, references`` layout, "prefix" for ``text + references,
    # target``.
    KV_CACHED_SLICE: ClassVar[str] = "suffix"

    def _lora_key_map(self, key: str) -> str:
        """Map a LoRA key to this model's schema.

        Training tools name LoRA targets differently from this repo's model schema;
        families with a non-canonical schema override this. Default: identity.
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

        # Preferences implied by markers in the DiT header (see
        # ``thenoise.utils.checkpoint``).
        self.checkpoint_prefs = detect_checkpoint_prefs(config.dit_path)

        # Component placement: subclasses register ``dit`` / ``text_encoder`` /
        # ``vae``; the pipeline controller ensures/offloads them by name.
        self.memory = MemoryManager(self.device, self.offload_device)

        torch._dynamo.config.recompile_limit = 64

        # Cached LoRA factors, for clean switching between requests.
        self._active_lora_result: Optional[LoRAApplyResult] = None
        self._active_lora_spec: Optional[str] = None

        # Lazy latent upscaler (only built if upscale is requested).
        self._upscaler: Optional[LatentUpscaler] = None

    # ------------------------------------------------------------ preferences
    def pref(self, name: str, request_value: Any = None) -> Any:
        """Resolve a generation preference: request > checkpoint marker > model default.

        ``request_value`` is what the API/CLI carried, or ``None`` when the user did
        not ask — that is how "auto" is represented on the wire.
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

        Raises ``KeyError`` on an unlisted name.
        """
        if name not in self.CAPABILITIES:
            raise KeyError(f"unknown capability {name!r}; known: {sorted(self.CAPABILITIES)}")
        return self.CAPABILITIES[name]

    # ------------------------------------------------------------ devices
    def _detect_offload_device(self, config: ModelConfig) -> str:
        """Pick an offload device from safetensors size vs VRAM (or ``device``).

        Resident bytes are estimated from the combined size of the three checkpoint
        files; if they fit the device VRAM with ``_RESIDENT_VRAM_FRACTION`` headroom
        left over for activations we stay resident, otherwise offload to CPU.
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

        Text-encoder only. ``fuse_text`` turns the result into the model-internal
        conditioning, with the DiT resident.
        """

    def fuse_text(self, cond: "Conditioning") -> "Conditioning":
        """Raw conditioning -> model-internal conditioning, with the DiT resident."""
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

        ``ref``/``ref_method`` are only passed in the edit path.
        """
        return latents

    @abstractmethod
    def schedule(self, params: SamplingParams) -> list[Step]:
        """Build the model's denoising schedule (one ``Step`` per iteration)."""

    def begin_sampling(self, schedule: list[Step]) -> None:
        """Called once per denoise pass with the schedule the solver is about to run.

        ``schedule()`` builds a model's *own* grid, but the pipeline may substitute an
        explicit sigma grid and the refine runs a short one of its own: this is where a
        kernel that needs the whole grid — a flow-map DiT embeds the timestep each step
        jumps to — learns the one it will actually be stepped through.
        """

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
        """Model-internal -> canonical 4D latent. Runs ONCE after the loop."""
        return latents

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Return the effective (width, height). Override to round/validate."""
        return width, height

    #: Area an edit is sampled at when the request names no size.
    EDIT_DEFAULT_AREA: ClassVar[int] = 1024 * 1024

    def default_edit_size(self, width: int, height: int) -> Tuple[int, int]:
        """The result size of an edit that names none, for a ``width`` x ``height`` reference.

        The reference's aspect ratio at :data:`EDIT_DEFAULT_AREA`; ``resolve_size`` still
        rounds the answer onto this model's latent cell. Override it when a model was
        trained at particular sizes rather than at any size.
        """
        scale = (self.EDIT_DEFAULT_AREA / (width * height)) ** 0.5
        return round(width * scale), round(height * scale)

    def percent_to_sigma(self, percent: float) -> float:
        """Map a percent (0..1) to a sigma, used by the sampler's SNR offset.

        The ER-SDE solver needs its first sigma strictly below 1, so flow models
        override this with their shift.
        """
        return 1.0 - percent

    # ------------------------------------------------------------ editing
    # How one reference is fitted for ``encode_reference``: aspect preserved, never
    # cropped, aligned to the model's latent cell. ``prepare_reference`` must stay a
    # pure function of the image and this struct — the pipeline caches the encoded
    # reference on it, and on the canvas size it hands ``fit_references``.
    REFERENCE_SIZING: ClassVar[ReferenceSizing] = ReferenceSizing()

    def prepare_reference(self, image: Image.Image) -> Image.Image:
        """Fit one reference image for :meth:`encode_reference`."""
        return self.REFERENCE_SIZING.apply(image)

    def fit_references(self, images: List[Image.Image],
                       target: Tuple[int, int]) -> List[Image.Image]:
        """Every reference of one edit, fitted for :meth:`encode_reference`.

        ``target`` is the size of the result. The default hands each image to
        :meth:`prepare_reference` and ignores it, because a reference normally keeps the
        shape it arrived in; a model whose edit lines up with its lone reference pixel
        for pixel overrides this to put that reference on the canvas's own grid.
        """
        return [self.prepare_reference(img) for img in images]

    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) into the canonical latent."""
        raise NotImplementedError(f"{self.name} does not support reference editing")

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "index",
        ref_index: int = 1,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Canonical reference latent -> model-internal (tokens, ids).

        ``ref_index`` is the 1-based position among the reference images.
        """
        return None

    # ------------------------------------------------------- reference KV cache
    def start_kv_caches(
        self,
        params: SamplingParams,
        has_reference: bool,
        has_uncond: bool,
    ) -> None:
        """Create this run's K/V caches, one per conditioning branch.

        Call in ``prepare_latent``. The branches can have different token counts, so
        they cannot share buffers; the uncond cache exists only under CFG.
        """
        if not (params.kv_cache and has_reference and self.capability("kv_cache")):
            self._kv_caches = None
            return
        caches = {"cond": KVCache("cond", self.KV_CACHED_SLICE)}
        if has_uncond:
            caches["uncond"] = KVCache("uncond", self.KV_CACHED_SLICE)
        self._kv_caches = caches

    def kv_cache(self, branch: str) -> Optional[KVCache]:
        """The cache of one conditioning branch (``cond`` / ``uncond``), else ``None``."""
        return None if self._kv_caches is None else self._kv_caches.get(branch)

    def end_kv_caches(self) -> None:
        """Drop the run's caches. Call in ``finalize_latent``, before the VAE decode."""
        self._kv_caches = None

    # --------------------------------------------------------------- LoRA
    def _parse_lora_spec(self, spec: str) -> Tuple[str, float]:
        """Parse a 'filename:weight' spec into (filename, weight), appending .safetensors."""
        if ":" in spec:
            filename, weight_str = spec.rsplit(":", 1)
            weight = float(weight_str)
        else:
            filename = spec
            weight = 1.0

        filename = ensure_safetensors(filename)

        return filename, weight

    def _resolve_lora_path(self, filename: str) -> str:
        """Resolve a LoRA filename against lora_dir, rejecting path traversal."""
        return resolve_in_dir(self.lora_dir, filename)

    def _get_lora_sd(self, filename: str) -> Dict[str, torch.Tensor]:
        """Load a LoRA state dict from disk."""
        filepath = self._resolve_lora_path(filename)

        logger.info("Loading LoRA: %s", filepath)
        return load_file(filepath, device=self.device)

    def _make_lora_spec_hash(self, lora_specs: Optional[List[str]]) -> str:
        """Stable key for a set of LoRA specs."""
        if not lora_specs:
            return "__none__"
        return "|".join(sorted(lora_specs))

    def switch_loras(
        self,
        lora_specs: Optional[List[str]],
        dit: torch.nn.Module,
    ) -> None:
        """Switch active LoRAs on the DiT (in-place, under the lock).

        No-op when the requested config matches the current one.
        """
        new_spec = self._make_lora_spec_hash(lora_specs)
        if new_spec == self._active_lora_spec:
            return

        if self._active_lora_result is not None:
            logger.debug("Undoing previous LoRA config")
            undo_lora_on_model(dit, self._active_lora_result)
            self._active_lora_result = None

        if lora_specs and self.lora_dir is not None:
            lora_sds = []
            multipliers = []
            for spec in lora_specs:
                filename, weight = self._parse_lora_spec(spec)
                lora_sds.append(self._get_lora_sd(filename))
                multipliers.append(weight)

            self._active_lora_result = apply_lora_to_model(
                dit, lora_sds, multipliers,
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
        """Available LoRA names relative to lora_dir, with .safetensors stripped."""
        return list_safetensors(self.lora_dir)

    # ------------------------------------------------------- latent upscaler
    def get_upscaler(self) -> LatentUpscaler:
        """This model's latent upscaler, built once on first use (under the lock)."""
        if self._upscaler is None:
            self._upscaler = self._create_upscaler()
        return self._upscaler

    @abstractmethod
    def _create_upscaler(self) -> LatentUpscaler:
        """Build this model's latent upscaler."""
        ...

    # ------------------------------------------------------------ pixel format
    @property
    def pixel_channels(self) -> int:
        """Pixel channels this model's VAE consumes and emits (3 = RGB, 4 = RGBA)."""
        return getattr(getattr(self, "vae", None), "pixel_channels", 3)

    # ------------------------------------------------------------ stage claims
    def activation_bytes(self, width: int, height: int) -> int:
        """What a stage moving this many pixels needs beside the weights it runs on.

        An over-estimate costs a weight set a round trip, an under-estimate costs the
        run. Overridable per codec once its own peak is measured. The pipeline asks at
        the final resolution: whatever the device holds at the end of a request it holds
        through the tail that follows.
        """
        return _STAGE_FLOOR_BYTES + int(_STAGE_BYTES_PER_MEGAPIXEL * width * height / 1e6)

    def denoise_activation_bytes(self, width: int, height: int) -> int:
        """The sampling loop's claim: the pixel-stage figures, borrowed, because that
        loop has never been measured. Its peak follows token count and head width rather
        than the pixel count of a VAE's feature maps, so a codec that knows better
        overrides this and not ``activation_bytes``.
        """
        return self.activation_bytes(width, height)

    # ------------------------------------------------------------ decode
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Shared VAE decode — the final generation step.

        Canonical 4D latent -> pixels ``[C, H, W]`` in [-1, 1] as an fp32 GPU tensor.
        """
        dev = torch.device(self.device)
        pixels = self.vae.decode_to_pixels(latents.to(dev, dtype=self.vae.dtype))
        if pixels.ndim == 5:  # [B, C, 1, H, W] -> [B, C, H, W]
            pixels = pixels.squeeze(2)
        pixels = pixels.to(torch.float32)
        return pixels[0]  # [C, H, W] in [-1, 1]


__all__ = ["DiffusionModel", "Conditioning", "normalize_keys"]
