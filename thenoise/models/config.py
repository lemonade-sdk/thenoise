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

from thenoise.postprocess.options import PostprocessOptions, options_from
from thenoise.utils.options import declared_knobs

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image


@dataclass
class ModelConfig:
    """Static load-time configuration for a model.

    ``dit_paths`` holds the DiT checkpoint(s) by role: ``""`` is the primary one every
    model takes, and any further role (``"low"`` for a two-expert model) belongs to the
    adapter that asked for it.
    """

    dit_paths: dict[str, str]
    vae_path: str
    text_encoder_path: str
    offload_device: str      # where idle weights wait; the compute device offloads nothing
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    lora_dir: Optional[str] = None

    @property
    def dit_path(self) -> str:
        """The primary DiT."""
        return self.dit_paths[""]


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
@declared_knobs(PostprocessOptions)
class GenerateRequest:
    """The complete user-facing generation request, mirroring the HTTP API and CLI.

    The post-decode knobs are appended as ordinary flat fields by ``declared_knobs``, 
    so a new filter needs no new line here.
    """

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
    lora_specs: Optional[List[str]] = None
    pixel_upscaler: Optional[str] = None
    image: Optional[Union[Image.Image, List[Image.Image]]] = None
    # KV cache for supported models; T2I defaults off, editing may auto-enable.
    kv_cache: Optional[bool] = None
    # Reference conditioning method for editing (edit only); None = auto.
    ref_method: Optional[str] = None

    def postprocess_options(self) -> PostprocessOptions:
        """The post-decode knobs, as the filter chain wants them."""
        return options_from(self)


@dataclass(frozen=True)
class SamplingParams:
    """Denoise-stage geometry + knobs passed to the model kernels."""

    height: int
    width: int
    steps: int
    seed: int
    guidance_scale: float
    sampler: str
    # Resolved K/V caching preference for this sampling run.
    kv_cache: bool = False


@dataclass(frozen=True)
class VideoSamplingParams(SamplingParams):
    """A clip's denoise-stage geometry: the image fields plus how long the clip is.

    ``frames`` is the *resolved* count, already on the codec's temporal grid, because a
    model seeds ``(frames - 1) / 4 + 1`` latent frames and must not re-derive it.
    """

    frames: int = 1


__all__ = ["ModelConfig", "EncodePromptArgs", "GenerateRequest", "SamplingParams",
           "VideoSamplingParams"]
