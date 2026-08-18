"""Single-model runtime: loads exactly one DiffusionModel at a time.

The runtime is deliberately thin: it detects the model class from the DiT
checkpoint, holds the single resident instance, and swaps (unloads + GCs) on
reload so only one set of weights is ever resident.

The text encoder and VAE are assumed to match the detected model; a wrong type
throws during load and we fail anyway.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Runtime knobs, all passed on the CLI (bf16 is fixed for this engine).

    ``upscaler_dir`` is server configuration (like host/port): pixel-domain
    upscaling is a pixel-space / postprocessing concern that needs no diffusion
    model, so its directory is NOT a model-load parameter.

    ``gallery_dir`` is a server configuration: when set, generated images are
    saved here and served in the gallery.
    """
    device: str = "cuda"      # ROCm torch aliases cuda -> hip
    host: str = "127.0.0.1"
    port: int = 8000
    upscaler_dir: str = ""     # directory of pixel-domain upscaler models
    gallery_dir: str = ""     # directory for saving/serving generated images


@dataclass
class ModelPaths:
    """Checkpoint paths supplied by the CLI.

    ``lora_dir`` stays here because LoRAs mutate the DiT weights (a model
    concern). Pixel-upscaler config is not model-bound, so ``upscaler_dir``
    lives on ``Settings`` instead.
    """
    dit_path: str
    vae_path: str
    text_encoder_path: str
    lora_dir: str = ""


class NotLoadedError(RuntimeError):
    """Raised when no model is loaded."""


class Runtime:
    def __init__(self, settings):
        from .upscale.pixel import PixelUpscalerManager

        self._settings = settings
        self._model: Any = None
        self._model_name: Optional[str] = None
        # Shared pixel-domain upscaler pool (model-free, server-level). Reused by
        # the pipeline controller and a future standalone /upscale endpoint.
        self._pixel_upscalers = PixelUpscalerManager(
            upscaler_dir=settings.upscaler_dir, device=settings.device
        )
        self._pipeline = None

    def load(self, paths: ModelPaths) -> None:
        from .models import resolve
        from .models.config import ModelConfig
        from .pipeline import PipelineController

        cls = resolve(paths.dit_path)
        name = cls.name

        config = ModelConfig(
            dit_path=paths.dit_path,
            vae_path=paths.vae_path,
            text_encoder_path=paths.text_encoder_path,
            device=self._settings.device,
            lora_dir=paths.lora_dir or None,
        )

        self._unload()  # swap: only one model resident at a time
        logger.info("Loading model '%s'", name)
        self._model = cls(config=config)
        self._model_name = name
        self._pipeline = PipelineController(self._model, self._pixel_upscalers)

    def _unload(self) -> None:
        if self._model is None:
            return
        logger.info("Unloading model '%s'", self._model_name)
        del self._model
        self._model = None
        self._model_name = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # torch may be unavailable (config-only tests)
            pass

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
    def pixel_upscalers(self):
        """The shared pixel-domain upscaler pool (available without a model)."""
        return self._pixel_upscalers

    @property
    def model_name(self) -> str:
        if self._model is None:
            raise NotLoadedError("no model loaded")
        return self._model_name

    def available(self) -> list[str]:
        return [self._model_name] if self._model else []
