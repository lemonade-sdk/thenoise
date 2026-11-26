"""The engine contract every adapter shares: the preferences and capabilities a model is
asked for, where its weights live, and how one denoise pass runs. ``PixelModel``, in
``thenoise.models.pixel_base``, derives further for a latent that decodes to pixels.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import torch

from thenoise.dit.kvcache import KVCache
from thenoise.memory import MemoryManager
from thenoise.models.lora_support import LoRASupport
from thenoise.utils.device_patches import apply_device_patches
from thenoise.utils.attention import ATTENTION_METHODS as ATTENTION_WORDS
from thenoise.utils.attention import DEFAULT_ATTENTION, LOSSY_ATTENTION
from thenoise.utils.dynamo import apply_recompile_limit
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.utils.image_tensor import ReferenceSizing, flatten_alpha
from thenoise.samplers import Step

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image
from thenoise.utils.checkpoint import checkpoint_bytes, detect_checkpoint_prefs
from thenoise.utils.lora_roles import DEFAULT_LORA_ROLE
from thenoise.utils.safetensors import Header, Signature

logger = logging.getLogger(__name__)


@dataclass
class Conditioning:
    """Bundle of (un)conditional embeddings produced by ``encode_prompt``.

    ``null``/``null_mask`` are ``None`` when guidance is off.
    """

    cond: torch.Tensor
    cond_mask: Optional[torch.Tensor] = None
    null: Optional[torch.Tensor] = None
    null_mask: Optional[torch.Tensor] = None


#: How a reference picture enters a run. The latent paths append VAE-encoded tokens
#: (``index_timestep_zero`` conditions them at timestep zero, which is what makes their
#: K/V step-invariant and cacheable); ``vision`` is the conditioner's own tower, so no
#: VAE stage, no reference latent, and nothing survives pixel for pixel.
LATENT_REF_METHODS = ("index", "index_timestep_zero")
VISION_REF_METHOD = "vision"


class DiffusionModel(LoRASupport, ABC):
    """Base class for model adapters. Subclasses must set ``name``."""

    name: str = ""

    #: What this adapter produces; ``thenoise.mediums`` maps it to a controller and routes.
    MEDIUM: ClassVar[str] = "image"

    #: Required components; ``()`` when embedded.
    REQUIRED_COMPONENTS: ClassVar[Tuple[str, ...]] = ("vae", "text_encoder")

    #: Components a release may ship as their own file, so a request can name one. A
    #: role outside this set is refused before anything loads.
    COMPONENT_ROLES: ClassVar[Tuple[str, ...]] = ()

    #: Model defaults behind ``pref``; an adapter merges the entries it differs on.
    DEFAULT_PREFS: ClassVar[Dict[str, Any]] = {
        # Every medium carries the field, whether its trunk has more than one way to
        # compute attention yet or not: a client asks the same question of all of them.
        "attention": DEFAULT_ATTENTION,
        "steps": 28,
        # CFG scale; <= 1.0 disables the unconditional forward.
        "guidance_scale": 0.0,
        "sampler": "er_sde",
        "sigmas": None
    }

    #: Features ``/health`` reports verbatim; a medium nests its own under its name.
    CAPABILITIES: ClassVar[Dict[str, Any]] = {"kv_cache": False}

    #: How this model can use a reference picture, best first; empty means none.
    REF_METHODS: ClassVar[Tuple[str, ...]] = ()

    #: What this model answers for ``attention``: the words in
    #: :data:`thenoise.utils.attention.ATTENTION_METHODS`, plus ``optimized`` where the
    #: architecture has a lossy path to offer.
    ATTENTION_METHODS: ClassVar[Tuple[str, ...]] = ATTENTION_WORDS

    #: Created by ``start_kv_caches``, dropped by ``end_kv_caches``.
    _kv_caches: Optional[Dict[str, KVCache]] = None

    #: Preferences implied by the loaded checkpoint's markers; set per instance.
    checkpoint_prefs: Dict[str, Any] = {}

    #: Which end of the sequence the KV cache freezes: ``"suffix"`` or ``"prefix"``.
    KV_CACHED_SLICE: ClassVar[str] = "suffix"

    #: The tensor names that make a checkpoint this adapter's DiT. The same declaration
    #: answers the catalog's ``detect`` and the loader's geometry read, so a file cannot
    #: be claimed by a loader that would then fail to build it. An adapter whose claim
    #: is not a plain name test — a depth window, a substring, a family its sibling also
    #: claims — overrides :meth:`detect` instead.
    SIGNATURE: ClassVar[Signature] = Signature("")

    @classmethod
    def detect(cls, header: Header) -> bool:
        """Whether this file is this model's DiT, as :attr:`SIGNATURE` states it."""
        return header.identifies(cls.SIGNATURE)

    @classmethod
    def dit_roles(cls, checkpoints: dict[str, Header]) -> dict[str, Header]:
        """The DiT files among the roles a request named.

        The runtime hands over every role it was given, and a role in
        :attr:`COMPONENT_ROLES` is a codec or a projection rather than a transformer, so
        saying anything about DiT files means looking past those.
        """
        return {role: header for role, header in checkpoints.items()
                if role not in cls.COMPONENT_ROLES}

    @classmethod
    def validate_paths(cls, checkpoints: dict[str, Header]) -> None:
        """Reject a set of DiT files this model cannot use, before anything loads.

        Every role the request named arrives as a parsed header, so an extra file is
        identified here rather than discovered halfway through its load. Adapters that
        need more than the primary checkpoint override this; the role names are theirs,
        so the runtime never has to know them.
        """
        if len(cls.dit_roles(checkpoints)) > 1:
            raise ValueError(f"{cls.name} takes one DiT checkpoint")

    @classmethod
    def require_same_model(cls, checkpoints: dict[str, Header]) -> None:
        """Refuse an extra DiT file this adapter would not have claimed on its own."""
        for role, header in cls.dit_roles(checkpoints).items():
            if role and not cls.detect(header):
                raise ValueError(
                    f"{header.path} does not match the {cls.SIGNATURE.name} signature: "
                    "every DiT file one model loads has to be that model's")

    def __init__(self, *, config: ModelConfig):
        self.device = config.device
        self.offload_device = config.offload_device
        self.dtype = config.dtype
        self.dit_paths = config.dit_paths
        self.dit_path = config.dit_path
        self.vae_path = config.vae_path
        self.text_encoder_path = config.text_encoder_path
        self.lora_dir = config.lora_dir

        #: The primary DiT's header, read once for the whole load: geometry, markers
        #: and the weight load all ask the same parsed copy.
        self.header = config.header()
        self.checkpoint_prefs = detect_checkpoint_prefs(self.header)

        # Subclasses register ``dit`` / ``vae`` / ``text_encoder`` for staging by name.
        self.memory = MemoryManager(self.device, self.offload_device)

        apply_recompile_limit()
        apply_device_patches(self.device)

        # Cached LoRA factors, for clean switching between requests.
        self._init_lora_state()

    # ------------------------------------------------------------ preferences
    @property
    def default_prefs(self) -> Dict[str, Any]:
        """What an unset request field falls back to.

        The class defaults with the loaded checkpoint's markers on top. A model that
        ships several releases behind one adapter layers its per-release table between
        the two — markers still win, because a checkpoint's own hint outranks a default
        for the family it was released under.
        """
        return {**self.DEFAULT_PREFS, **self.checkpoint_prefs}

    def pref(self, name: str, request_value: Any = None) -> Any:
        """Resolve a generation preference: request > checkpoint marker > model default.

        ``request_value`` is what the API/CLI carried, or ``None`` when the user did
        not ask — that is how "auto" is represented on the wire.
        """
        defaults = self.default_prefs
        if name not in defaults:
            raise KeyError(f"unknown preference {name!r}; known: {sorted(defaults)}")
        if request_value is not None:
            value, source = request_value, "request"
        else:
            value = defaults[name]
            source = ("checkpoint" if name in self.checkpoint_prefs else "model default")
        logger.debug("%s = %s (%s)", name, value, source)
        return value

    # ------------------------------------------------------------ capabilities
    def capability(self, name: str) -> Any:
        """Value for capability ``name`` (see ``CAPABILITIES``).

        Raises ``KeyError`` on an unlisted name.
        """
        if name not in self.CAPABILITIES:
            raise KeyError(f"unknown capability {name!r}; known: {sorted(self.CAPABILITIES)}")
        return self.CAPABILITIES[name]

    # ------------------------------------------------------------ devices
    def _load_target(self, path: str) -> str:
        """The device to build the weights at ``path`` on, the manager's decision.

        The file's own bytes are the hint: a component that owns a file reads all of it.
        """
        return str(self.memory.device_for(checkpoint_bytes(str(path))))

    # ------------------------------------------------------------ teardown
    def unload(self) -> None:
        """Drop whatever this instance holds that would outlive it.

        The runtime calls it before releasing the device on a swap. Doing nothing is the
        answer for an adapter whose weights are only reachable through itself; a codec's
        feature cache or residency bookkeeping is not, and clears itself here.
        """

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

    def percent_to_sigma(self, percent: float) -> float:
        """Map a percent (0..1) to a sigma, used by the sampler's SNR offset.

        The ER-SDE solver needs its first sigma strictly below 1, so flow models
        override this with their shift.
        """
        return 1.0 - percent

    # ------------------------------------------------------------ references
    #: Fitted before anything encodes it; the pipeline caches on this struct, so stay pure.
    REFERENCE_SIZING: ClassVar[ReferenceSizing] = ReferenceSizing()

    def prepare_reference(self, image: Image.Image) -> Image.Image:
        """Fit one reference picture, for a VAE latent or a conditioner's pixels."""
        return self.REFERENCE_SIZING.apply(image)

    def conditioned_images(self, args: EncodePromptArgs) -> List[Image.Image]:
        """``args.image``, single or several, as a conditioner that reads pictures
        wants them: RGB, and fitted the way this model fits any reference.

        The alpha is composited because a vision tower takes three channels; the fit is
        the one in :meth:`prepare_reference`, so a model shows its conditioner the same
        pixels it would hand its VAE.
        """
        if args.image is None:
            return []
        images = args.image if isinstance(args.image, list) else [args.image]
        return [flatten_alpha(self.prepare_reference(image)) for image in images]

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
        """Drop the run's caches. Call in ``finalize_latent``, before the decode."""
        self._kv_caches = None

    # ------------------------------------------------------- LoRA-able components
    def attention_methods(self) -> Tuple[str, ...]:
        """The ``attention`` words this model answers on the hardware it is loaded onto.

        :attr:`ATTENTION_METHODS` is the architecture's vocabulary; a lossy path is only
        worth offering a client when this box can run it.
        """
        if "optimized" not in self.ATTENTION_METHODS:
            return self.ATTENTION_METHODS
        blocked = self.attention_routing_blocked()
        return self.ATTENTION_METHODS if blocked is None else tuple(
            w for w in self.ATTENTION_METHODS if w != LOSSY_ATTENTION)

    def resolve_attention(self, method: str) -> str:
        """The attention a request asked for, checked against this model and this box.

        The words are the architecture's; whether this box runs the lossy one is only
        worth asking when the request asked for it.
        """
        if method not in self.ATTENTION_METHODS:
            raise ValueError(
                f"{self.name} does not answer {method!r} attention; it answers "
                f"{', '.join(self.ATTENTION_METHODS)}")
        blocked = (self.attention_routing_blocked()
                   if method == LOSSY_ATTENTION else None)
        if blocked is not None:
            raise ValueError(
                f"{self.name} cannot run optimized attention here: {blocked}")
        return method

    def attention_routing_blocked(self) -> Optional[str]:
        """Why this box cannot run the model's lossy attention path, ``None`` when it can.

        A model that offers ``optimized`` answers this from its own trunk width, which is a
        checkpoint fact rather than a class one.
        """
        return None

    def lora_components(self) -> dict[str, torch.nn.Module]:
        """This model's LoRA-able components, by the role a request names them with.

        The keys are the roles `switch_loras` accepts and the wire may name, so a model
        with a second component adds an entry here rather than a second list to keep in
        step. The name is the one the component is registered for residency under.
        """
        return {DEFAULT_LORA_ROLE: self.dit}

__all__ = ["DiffusionModel", "Conditioning"]
