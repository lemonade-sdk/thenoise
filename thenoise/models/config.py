"""Configuration dataclasses that wrap related generation fields.

Grouping fields into structs means adding an option never changes a method
signature: ``ModelConfig`` is the load-time config, ``EncodePromptArgs`` the text
encoder's input, ``GenerateRequest`` the user-facing request and ``SamplingParams``
the denoise-stage geometry.
"""
from __future__ import annotations

import os
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

from thenoise.mesh.octree import CHUNK
from thenoise.utils.attention import DEFAULT_ATTENTION
from thenoise.utils.lora_roles import LoraSpecs
from thenoise.utils.safetensors import Header, read_header

from thenoise.postprocess.options import PostprocessOptions, options_from
from thenoise.utils.options import declared_knobs

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image


def _resolved(path: str) -> str:
    """Files are identified by their resolved path: an all-in-one checkpoint arrives
    under several roles at once."""
    return os.path.realpath(os.path.expanduser(path))


@dataclass
class ModelConfig:
    """Static load-time configuration for a model.

    ``dit_paths`` holds the DiT checkpoint(s) by role: ``""`` is the primary one every
    model takes, and any further role (``"low"`` for a two-expert model) belongs to the
    adapter that asked for it.

    ``headers`` is those same files already parsed, which is what the runtime does to
    pick the model: handing them down is what keeps one checkpoint from being opened
    again by every layer that wants to know its geometry.
    """

    dit_paths: dict[str, str]
    vae_path: str
    text_encoder_path: str
    offload_device: str      # where idle weights wait; the compute device offloads nothing
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    lora_dir: Optional[str] = None
    #: Component checkpoints by role, beside the transformer, VAE and text encoder.
    extra_paths: dict[str, str] = field(default_factory=dict)
    #: Headers handed in already parsed, by role. The runtime passes the one it read to
    #: pick the model; anything a role has here is never opened again.
    headers: Optional[dict[str, Header]] = None
    #: Parsed headers by resolved file, so a file is opened once however many roles
    #: share it.
    _read: dict = field(default_factory=dict, repr=False, compare=False)
    _given: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def dit_path(self) -> str:
        """The primary DiT."""
        return self.dit_paths[""]

    @property
    def paths(self) -> dict[str, str]:
        """Every checkpoint file the request named, by role."""
        return {**self.dit_paths, **self.extra_paths}

    def header(self, role: str = "") -> Header:
        """Role ``role``'s parsed header, read once per file however many roles share it."""
        given = self.headers or {}
        if role in given:
            return given[role]
        if not self._given:
            self._given = {_resolved(h.path): h for h in given.values()}
        path = _resolved(self.paths[role])
        if path not in self._read:
            self._read[path] = self._given.get(path) or read_header(path)
        return self._read[path]

    def all_headers(self) -> dict[str, Header]:
        """Every named checkpoint's header, keyed as :attr:`paths` is."""
        return {role: self.header(role) for role in self.paths}


@dataclass
class EncodePromptArgs:
    """Arguments for ``encode_prompt``, bundled into one struct.

    ``image`` is only set when the request brought one: a multimodal encoder feeds it
    as vision tokens, on its own where the model has no reference latent and as well as
    one where it has.
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
    lora_specs: LoraSpecs = None
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
    flow_shift: float = 1.0
    fps: int = 24
    #: ``auto`` / ``full`` / ``optimized``, as the request asked and the model answered.
    attention: str = DEFAULT_ATTENTION


@dataclass(frozen=True)
class MeshSamplingParams(SamplingParams):
    """A shape's denoise-stage geometry: its token count, a shape having no pixel grid."""

    num_latents: int = 4096


@dataclass(frozen=True)
class MeshDecodeParams:
    """The volume-and-surface knobs, which no sampler sees."""

    # No defaults here: a model's own defaults table is where these are written down.
    octree_resolution: int
    bounds: float
    mc_level: float
    chunk: int = CHUNK


__all__ = ["ModelConfig", "EncodePromptArgs", "GenerateRequest", "SamplingParams",
           "VideoSamplingParams", "MeshSamplingParams", "MeshDecodeParams"]
