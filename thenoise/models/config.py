"""Configuration dataclasses that wrap related generation fields.

These exist so that adding a new option never changes a method signature — new
fields are added to the relevant struct instead. There are three groups:

* ``ModelConfig``   — static load-time config for a ``DiffusionModel``.
* ``GenerateRequest`` — the complete user-facing generation request (the
  API/CLI surface).
* ``SamplingParams`` — the denoise-stage geometry + knobs passed to the model
  kernels (``init_latents``, ``prepare_latent``, ``schedule``,
  ``finalize_latent``) and to the controller's denoise/refine helpers.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image


@dataclass
class ModelConfig:
    """Static load-time configuration for a model.

    Deliberately model-bound: every field is needed to load the model's own
    weights. Pixel-upscaler configuration is NOT a model concern and lives on
    the server settings instead (it operates in pixel space, needs no model).
    """

    dit_path: str
    vae_path: str
    text_encoder_path: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    lora_dir: Optional[str] = None  # LoRAs mutate the DiT weights -> model concern
    checkpoint_path: Optional[str] = None  # single combined SDXL checkpoint (no split)
    sd_zsnr: Optional[bool] = None  # SDXL zero-terminal-SNR: None=auto (ztsnr marker), True=force on, False=force off


@dataclass
class EncodePromptArgs:
    """Arguments for ``encode_prompt``, bundled into one struct.

    A single structure (instead of passing each parameter separately) so adding a
    new knob — e.g. a future ``negative_prompt`` variant or an image — never
    changes the method signature. ``image`` is only set in the edit path
    (``supports_edit`` models); multimodal encoders feed it as vision tokens in
    addition to any reference latent.
    """

    prompt: str
    negative_prompt: str = ""
    guidance_scale: float = 0.0
    image: Optional[Image.Image] = None


@dataclass
class GenerateRequest:
    """The complete user-facing generation request.

    Mirrors the fields exposed by the HTTP API and CLI. Add new options here
    rather than to ``PipelineController.generate``'s signature.
    """

    prompt: str
    negative_prompt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    upscale: bool = False
    upscale_factor: float = 1.0
    upscale_type: str = "refined"
    sampler: Optional[str] = None
    qwen_vae_enhance: bool = False
    film_grain: float = 0.0
    sharpening: float = 0.0
    lora_specs: Optional[List[str]] = None
    pixel_upscaler: Optional[str] = None
    image: Optional[Union[Image.Image, List[Image.Image]]] = None


@dataclass(frozen=True)
class SamplingParams:
    """Denoise-stage geometry + knobs passed to the model kernels.

    ``seed`` is only consumed by ``init_latents``; the other kernels ignore the
    fields they do not need. Kept in one frozen struct so the kernel signatures
    stay stable as options are added.
    """

    height: int
    width: int
    steps: int
    seed: int
    guidance_scale: float
    sampler: str


__all__ = ["ModelConfig", "EncodePromptArgs", "GenerateRequest", "SamplingParams"]
