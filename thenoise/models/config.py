"""Configuration dataclasses that wrap related generation fields.

Grouping fields into structs means adding an option never changes a method
signature: ``ModelConfig`` is the load-time config, ``EncodePromptArgs`` the text
encoder's input, ``GenerateRequest`` the user-facing request and ``SamplingParams``
the denoise-stage geometry.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image


@dataclass
class ModelConfig:
    """Static load-time configuration for a model."""

    dit_path: str
    vae_path: str
    text_encoder_path: str
    device: str = "cuda"
    offload_device: str = ""  # empty = auto-detect from safetensors size vs VRAM
    dtype: torch.dtype = torch.bfloat16
    lora_dir: Optional[str] = None


@dataclass
class EncodePromptArgs:
    """Arguments for ``encode_prompt``, bundled into one struct.

    ``image`` is only set in the edit path; multimodal encoders feed it as vision
    tokens in addition to any reference latent.
    """

    prompt: str
    negative_prompt: str = ""
    guidance_scale: float = 0.0
    image: Optional[Union[Image.Image, List[Image.Image]]] = None


@dataclass
class GenerateRequest:
    """The complete user-facing generation request, mirroring the HTTP API and CLI."""

    prompt: str
    negative_prompt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    sigmas: Optional[List[float]] = None
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
    # Reference-latent KV cache (edit only); None = auto.
    kv_cache: Optional[bool] = None
    # Reference conditioning method for editing (edit only); None = auto.
    ref_method: Optional[str] = None


@dataclass(frozen=True)
class SamplingParams:
    """Denoise-stage geometry + knobs passed to the model kernels."""

    height: int
    width: int
    steps: int
    seed: int
    guidance_scale: float
    sampler: str
    # Reference-latent KV cache (edit only); resolved to a concrete bool.
    kv_cache: bool = False


__all__ = ["ModelConfig", "EncodePromptArgs", "GenerateRequest", "SamplingParams"]
