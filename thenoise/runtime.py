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
    """Runtime knobs, all passed on the CLI (bf16 is fixed for this engine)."""
    device: str = "cuda"      # ROCm torch aliases cuda -> hip
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class ModelPaths:
    """Checkpoint paths supplied by the CLI."""
    dit_path: str
    vae_path: str
    text_encoder_path: str
    lora_dir: str = ""
    esrgan_path: str = ""  # optional pixel-domain Real-ESRGAN model


class NotLoadedError(RuntimeError):
    """Raised when no model is loaded."""


class Runtime:
    def __init__(self, settings):
        self._settings = settings
        self._model: Any = None
        self._model_name: Optional[str] = None

    def load(self, paths: ModelPaths) -> None:
        from .models import resolve

        cls = resolve(paths.dit_path)
        name = cls.name

        kwargs = dict(
            dit_path=paths.dit_path,
            vae_path=paths.vae_path,
            text_encoder_path=paths.text_encoder_path,
            device=self._settings.device,
        )
        kwargs["lora_dir"] = paths.lora_dir or None
        kwargs["esrgan_path"] = paths.esrgan_path or None

        self._unload()  # swap: only one model resident at a time
        logger.info("Loading model '%s'", name)
        self._model = cls(**kwargs)
        self._model_name = name

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
    def model_name(self) -> str:
        if self._model is None:
            raise NotLoadedError("no model loaded")
        return self._model_name

    def available(self) -> list[str]:
        return [self._model_name] if self._model else []
