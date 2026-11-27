"""Single-model runtime: loads exactly one DiffusionModel at a time, detected from the
DiT checkpoint, swapping (unload + GC) on reload. The VAE and encoder match the model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from thenoise.errors import ModelReloadNeeded
from thenoise import multigpu
from thenoise.inference import inference, inference_lock
from thenoise.utils.device import release_device_memory

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Runtime knobs, all passed on the CLI (bf16 is fixed for this engine).
    ``upscaler_dir`` is server configuration: pixel upscaling needs no diffusion model.
    """
    device: str = "cuda"      # ROCm torch aliases cuda -> hip
    offload_device: str = "cpu"   # the compute device itself keeps every weight set
    host: str = "127.0.0.1"
    port: int = 8000
    upscaler_dir: str = ""     # directory of pixel-domain upscaler models
    controlmap_dir: str = ""  # directory of control-map checkpoints
    devices: str = ""         # "cuda:0,cuda:1": stack GPUs for one model (multigpu.py)
    placement: str = ""       # "dit=split,text_encoder=cuda:1" (default: multigpu.default_placement)


@dataclass
class ModelPaths:
    """Checkpoint paths supplied by the CLI.

    ``lora_dir`` is model-bound: LoRAs mutate the DiT weights.
    """
    dit_path: str
    vae_path: str = ""
    text_encoder_path: str = ""
    lora_dir: str = ""
    dit_low_path: str = ""
    #: Component checkpoints by role: ``"audio_vae"``, ``"upscaler"``,
    #: ``"text_projection"``. Which roles a model reads is the adapter's; a role left
    #: out is looked for in the files that were given.
    extra_paths: dict[str, str] = field(default_factory=dict)

    @property
    def dit_paths(self) -> dict[str, str]:
        """The DiT files by role; what a role means is the adapter's, not ours."""
        if self.dit_low_path:
            return {"": self.dit_path, "low": self.dit_low_path}
        return {"": self.dit_path}


class NotLoadedError(RuntimeError):
    """Raised when no model is loaded."""


class Runtime:
    def __init__(self, settings):
        from .controlmap.manager import ControlMapManager
        from .controlmap_controller import ControlMapController
        from .upscale.pixel import PixelUpscalerManager
        from .upscale_controller import PixelUpscaleController

        self._settings = settings
        # Refused here, before anything loads. With --devices the first is the compute device.
        self._stack_spec = multigpu.parse_stack(settings.devices, settings.placement)
        self._device = str(self._stack_spec.primary or settings.device)
        self._spreads: dict[str, multigpu.Spread] = {}
        self._paths: Optional[ModelPaths] = None
        self._model: Any = None
        self._model_name: Optional[str] = None
        # Shared pixel-domain upscaler pool (model-free, server-level).
        self._pixel_upscalers = PixelUpscalerManager(
            upscaler_dir=settings.upscaler_dir, device=self._device
        )
        self._upscaler = PixelUpscaleController(self._pixel_upscalers)
        # Control-map checkpoints: model-free too, and independent of the upscalers.
        self._controlmaps = ControlMapManager(
            device=self._device, controlmap_dir=settings.controlmap_dir
        )
        self._controlmap = ControlMapController(self._controlmaps)
        self._pipeline = None

    def load(self, paths: ModelPaths) -> None:
        from .models import resolve
        from .models.config import ModelConfig
        from .mediums import controller_for
        # The catalog parse is the header the adapter and the loader go on to use, so a
        # checkpoint is opened once however many parts of a load want its geometry.
        header = resolve(paths.dit_path)
        cls = header.model_cls
        name = cls.name
        # Model declares its components.
        missing = [component.replace("_", "-") for component in cls.REQUIRED_COMPONENTS
                   if not getattr(paths, f"{component}_path", "")]
        if missing:
            raise ValueError(f"{name} needs --{' and --'.join(missing)}")

        config = ModelConfig(
            dit_paths=paths.dit_paths,
            headers={"": header},
            vae_path=paths.vae_path,
            text_encoder_path=paths.text_encoder_path,
            device=self._device,
            offload_device=self._settings.offload_device,
            lora_dir=paths.lora_dir or None,
            extra_paths=paths.extra_paths,
        )
        # A component file the adapter has no role for is refused, not left out quietly.
        unwanted = sorted(set(paths.extra_paths) - set(cls.COMPONENT_ROLES))
        if unwanted:
            raise ValueError(f"{name} has no use for --"
                             + ", --".join(role.replace("_", "-") for role in unwanted))
        cls.validate_paths(config.all_headers())
        dit_roles: list[str] = []
        if self._stack_spec.devices:   # without --devices a load is exactly what it was
            dit_roles = [f"dit_{role}" if role else "dit"
                         for role in cls.dit_roles(config.all_headers())]
            multigpu.check_dit_placement(self._stack_spec, dit_roles)
            roles = set(dit_roles) | {"vae", "text_encoder"} | set(cls.COMPONENT_ROLES)
            unknown = sorted(set(self._stack_spec.placement) - roles)
            if unknown:
                raise ValueError("--place: model has no component " + ", ".join(unknown))

        self._unload()  # swap: only one model resident at a time
        logger.info("Loading model '%s'", name)
        with inference():
            model = cls(config=config)
            self._stack(model, dit_roles)
        self._model = model
        self._model_name = name
        self._paths = paths
        # Every medium gets a controller, so `run` — the lock, the rebuild-and-retry of a
        # model that reports itself unfit — is one path whatever was loaded.
        self._pipeline = controller_for(model, self._pixel_upscalers)

    def _stack(self, model, dit_roles) -> None:
        """Spread ``model`` over ``--devices`` and move what ``--place`` names."""
        spec = self._stack_spec
        if not spec.devices:
            return
        plan = multigpu.default_placement(dit_roles, model.memory.component_names(), spec.devices)
        plan.update(spec.placement)
        logger.info("stacking %s over %s: %s", type(model).__name__,
                    ", ".join(map(str, spec.devices)), plan)
        self._spreads = multigpu.apply(model, spec, plan)
        if any(plan.get(role) == "split" for role in dit_roles):
            # Cache buffers belong to the primary device, not each split block's device.
            model.CAPABILITIES = {**model.CAPABILITIES, "kv_cache": False}

    def reload(self) -> None:
        """Rebuild the loaded model from its checkpoints, which clears an unfit state."""
        if self._paths is None:
            raise NotLoadedError("no model loaded")
        self.load(self._paths)

    def run(self, action):
        """``action(pipeline)``; again against a rebuilt model if it reports itself unfit.
        The action gets the controller, which a rebuild replaces.

        Under the inference lock throughout: a request keeps the model alive until it
        returns, so only a rebuild inside the lock gets its memory back.
        """
        with inference_lock:
            if self._pipeline is None:
                raise NotLoadedError("no model loaded")

            attempted = self._model
            try:
                return action(self.pipeline)
            except ModelReloadNeeded as needed:
                reason = str(needed)

            # Out of the handler, whose frames hold the old model. A changed instance
            # means somebody else already rebuilt it.
            stale = self._model is attempted
            attempted = None
            if stale:
                logger.warning("%s: rebuilding a model that reported itself unfit: %s",
                               self._model_name, reason)
                self.reload()
            return action(self.pipeline)

    def _unload(self) -> None:
        if self._model is None:
            return
        logger.info("Unloading model '%s'", self._model_name)
        self._pipeline = None
        self._model.unload()  # drops the references the instance itself holds
        self._model = None
        self._model_name = None
        for spread in self._spreads.values():
            spread.remove()
        self._spreads = {}
        release_device_memory(self._device)
        for dev in self._stack_spec.devices[1:]:
            release_device_memory(str(dev))

    @property
    def device(self) -> str:
        """The compute device: ``--devices``' first, else ``--device``."""
        return self._device

    @property
    def model(self) -> Any:
        if self._model is None:
            raise NotLoadedError("no model loaded")
        return self._model

    @property
    def pipeline(self):
        """The pipeline controller (None until a model is loaded)."""
        return self._pipeline

    @property
    def medium(self) -> Optional[str]:
        """Read off the model, so a swap cannot leave the replaced model's medium behind."""
        return None if self._model is None else self._model.MEDIUM

    @property
    def pixel_upscalers(self):
        """The shared pixel-domain upscaler pool (available without a model)."""
        return self._pixel_upscalers

    @property
    def upscaler(self):
        """Standalone pixel upscale controller (model-free, always available)."""
        return self._upscaler

    @property
    def controlmaps(self):
        """The control-map pool (available without a model)."""
        return self._controlmaps

    @property
    def controlmap(self):
        """Standalone control-map controller (model-free, always available)."""
        return self._controlmap

    @property
    def model_name(self) -> str:
        if self._model is None:
            raise NotLoadedError("no model loaded")
        return self._model_name

    def available(self) -> list[str]:
        return [self._model_name] if self._model else []

    def model_capabilities(self) -> dict:
        """What the loaded model can serve, as ``/health`` reports it (empty if none).

        The adapter's ``CAPABILITIES`` plus ``ref_methods``, which lives on the class
        as :attr:`DiffusionModel.REF_METHODS` because it describes a request field
        rather than a feature switch.
        """
        if self._model is None:
            return {}
        return {**self._model.CAPABILITIES,
                "ref_methods": list(self._model.REF_METHODS),
                "attention_methods": list(self._model.attention_methods())}

    def model_pixel_channels(self) -> int:
        """The loaded model's pixel width (3 = RGB, 4 = RGBA); 3 with no model."""
        if self._model is None:
            return 3
        return getattr(self._model, "pixel_channels", 3)

    def model_knobs(self) -> dict:
        """The loaded model's knob menus; empty with no model, or a medium of free numbers."""
        if self._model is None:
            return {}
        from .mediums import knobs_for

        return knobs_for(self.medium, self._model)

    def postprocess_options(self) -> type:
        """The options class behind ``GET /postprocess``: a shape's finish filters, or the
        pixel filters every other medium decodes into."""
        if self._model is None:
            from .postprocess.options import PostprocessOptions
            return PostprocessOptions
        from .mediums import tail_options

        return tail_options(self.medium)
