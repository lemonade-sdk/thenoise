"""Qwen-Image 2.1 adapter — single-stream DiT with a causal text/reference prefix.

Qwen3-VL-8B conditioner, Wan-2.2-layout 64-channel/16x RGBA VAE. The latent is the
DiT's own input: ``img_in`` takes the VAE's 64 channels directly and one token is one
latent cell, so there is no pack/unpack step and ``prepare_latent``/
``finalize_latent`` only move it.

Editing feeds the reference in twice: as vision tokens to the text encoder, and as a
VAE latent spliced into the DiT's text stream at the slot the tokenizer recorded. The
whole text + reference prefix is modulated at ``t = 0`` and is causally upstream of
the target, so its K/V are mathematically step-invariant. Runtime results can still
depend on GEMM shape: on gfx1151 the BF16 path changes rounding when cached sampling
projects only the target tokens, so T2I cache is enabled only for INT8-ConvRot.
That timestep-zero conditioning is architectural, so ``index_timestep_zero`` is the
only reference method.

The two halves of an edit are kept in step by construction: both are fitted by the
same ``prepare_reference``, so the vision tokens removed and the latent tokens
replacing them describe the same picture (one vision token per 32x32 pixels, one
latent cell per 16x16 — hence the 32-pixel alignment of both).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import List, Optional, Tuple

import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
from comfy_kitchen.registry import registry

from thenoise.dit.qwen_image21 import sampling as qwen21_sampling
from thenoise.dit.qwen_image21.encoder import (
    QwenImage21TextEncoder,
    load_qwen_image21_text_encoder,
)
from thenoise.dit.qwen_image21.models import QwenImage21Sequence
from thenoise.dit.qwen_image21.utils import is_qwen_image21_key
from thenoise.dit.qwen_image21.utils import load_qwen_image21_dit
from thenoise.models.base import (
    Conditioning,
    DiffusionModel,
    Step,
    normalize_keys,
)
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, Qwen21TranscodeUpscaler
from thenoise.utils.image_tensor import flatten_alpha, ReferenceSizing
from thenoise.utils.lora import FUSE_GATE_UP
from thenoise.utils.math import round_up
from thenoise.vae import load_wan22_vae

logger = logging.getLogger(__name__)


@dataclass
class QwenImage21Conditioning(Conditioning):
    """``Conditioning`` plus the per-branch reference-image token slots.

    ``cond_slots[i]`` is the index in ``cond`` where the ``i``-th reference latent is
    spliced into the text stream. They are per-branch because the two prompts have
    different lengths.
    """

    cond_slots: Optional[List[int]] = None
    null_slots: Optional[List[int]] = None


class QwenImage21Model(DiffusionModel):
    name = "qwen_image21"

    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 8,
        "guidance_scale": 1.0,
        "sampler": "euler",
        "ref_method": "index_timestep_zero",
        "sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
    }

    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": True}

    # The constructor enables text-to-image caching only for the validated
    # INT8-ConvRot profile and measured runtime. On gfx1151, BF16 GEMMs produce
    # materially different rounding when the target token dimension is reduced.
    KV_CACHE_WITHOUT_REFERENCE = False

    # The T2I cache A/B was validated on this exact runtime and kernel path. Keep the
    # execution qualification together so other adapters and reference editing are
    # not constrained by this Qwen-specific numerical result.
    T2I_CACHE_TORCH_VERSION = "2.14.0"
    T2I_CACHE_ROCM_BUILD = "rocm10.1"
    T2I_CACHE_HIP_VERSION_PREFIX = "7.16."
    T2I_CACHE_GFX_ARCH = "gfx1151"
    T2I_CACHE_GEMM_BACKEND = "hip"
    T2I_CACHE_COMFY_KITCHEN_VERSION = "0.2.40"

    # The step-invariant slice is the LEADING text/reference prefix (target last).
    KV_CACHED_SLICE = "prefix"

    # 32 px: a reference latent replaces whole 32x32 vision tokens, as in ``resolve_size``.
    REFERENCE_SIZING = ReferenceSizing(align=32)

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a Qwen-Image 2.1 DiT: its per-run ``modulation``
        and zero-centred ``txt_in.text_norm``."""
        keys = set(normalize_keys(f.keys()))
        return is_qwen_image21_key(keys)

    @staticmethod
    def _supports_tested_int8_projection(projection: torch.nn.Module) -> bool:
        """Whether a projection uses the exact per-row ConvRot profile we measured."""
        weight = getattr(projection, "weight", None)
        if not isinstance(weight, QuantizedTensor):
            return False
        if weight.layout_cls is not TensorWiseINT8Layout:
            return False
        params = weight.params
        scale = params.scale
        scale_shapes = {
            (projection.out_features,),
            (projection.out_features, 1),
        }
        return (
            weight.storage_dtype is torch.int8
            and weight.dtype is torch.bfloat16
            and tuple(weight.storage_shape) == tuple(weight.shape)
            and tuple(weight.shape) == (projection.out_features, projection.in_features)
            and tuple(params.orig_shape) == tuple(weight.shape)
            and params.transposed is False
            and params.is_weight is True
            and params.convrot is True
            and params.convrot_groupsize == 256
            and scale.dtype is torch.float32
            and tuple(scale.shape) in scale_shapes
        )

    @staticmethod
    def _supports_text_prefix_cache(dit: torch.nn.Module) -> bool:
        """Whether every changed projection has the measured INT8-ConvRot layout."""
        blocks = getattr(dit, "transformer_blocks", ())
        if not blocks:
            return False
        for block in blocks:
            projections = [
                block.attn.to_q,
                block.attn.to_k,
                block.attn.to_v,
                block.attn.to_out[0],
            ]
            if block.img_mlp.fused:
                projections.append(block.img_mlp.gate_up)
            else:
                projections.extend((block.img_mlp.proj, block.img_mlp.gate_layer))
            projections.append(block.img_mlp.out)
            if any(
                not QwenImage21Model._supports_tested_int8_projection(projection)
                for projection in projections
            ):
                return False
        return True

    @staticmethod
    def _text_prefix_cache_runtime_reason(device: torch.device | str) -> Optional[str]:
        """Explain why this process is outside the measured T2I-cache runtime."""
        device = torch.device(device)
        if device.type != "cuda":
            return "the tested T2I prefix cache requires a ROCm GPU"
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            return "the tested T2I prefix cache requires the ROCm/HIP runtime"

        version, separator, build = torch.__version__.partition("+")
        if (
            version != QwenImage21Model.T2I_CACHE_TORCH_VERSION
            or not separator
            or not build.startswith(QwenImage21Model.T2I_CACHE_ROCM_BUILD)
            or not str(torch.version.hip).startswith(
                QwenImage21Model.T2I_CACHE_HIP_VERSION_PREFIX
            )
        ):
            return (
                "the tested T2I prefix cache requires PyTorch 2.14.0 with "
                "ROCm 10.1 / HIP 7.16"
            )

        try:
            kitchen_version = package_version("comfy-kitchen")
        except PackageNotFoundError:
            return "could not verify the comfy-kitchen version for the tested T2I prefix cache"
        if kitchen_version != QwenImage21Model.T2I_CACHE_COMFY_KITCHEN_VERSION:
            return (
                "the tested T2I prefix cache requires comfy-kitchen "
                f"{QwenImage21Model.T2I_CACHE_COMFY_KITCHEN_VERSION}, not {kitchen_version}"
            )

        try:
            props = torch.cuda.get_device_properties(device)
            arch = props.gcnArchName.split(":", 1)[0]
        except Exception:
            return "could not verify the GPU architecture for the tested T2I prefix cache"
        if arch != QwenImage21Model.T2I_CACHE_GFX_ARCH:
            return (
                "the tested T2I prefix cache is qualified only for gfx1151, "
                f"not {arch}"
            )

        try:
            backend = registry.get_capable_backend("int8_linear")
        except Exception:
            return "could not verify the selected INT8 GEMM backend"
        if backend != QwenImage21Model.T2I_CACHE_GEMM_BACKEND:
            return (
                "the tested T2I prefix cache requires the HIP int8_linear backend, "
                f"not {backend}"
            )
        return None

    @staticmethod
    def _has_runtime_lora_branch(dit: torch.nn.Module) -> bool:
        """Runtime low-rank branches change the GEMM path used by T2I caching."""
        return any(
            getattr(module, "_lora_down", None) is not None
            or getattr(module, "_lora_up", None) is not None
            for module in dit.modules()
        )

    def _text_prefix_cache_ineligibility_reason(self) -> Optional[str]:
        profile_supported = getattr(
            self,
            "_text_prefix_cache_profile_supported",
            self.KV_CACHE_WITHOUT_REFERENCE,
        )
        if not profile_supported:
            return getattr(
                self,
                "KV_CACHE_WITHOUT_REFERENCE_REASON",
                "the checkpoint does not use the validated INT8-ConvRot profile",
            )

        reason = self._text_prefix_cache_runtime_reason(self.device)
        if reason:
            return reason
        if self._has_runtime_lora_branch(self.dit):
            return (
                "the T2I prefix cache is disabled while a quantized LoRA uses a "
                "runtime projection branch"
            )
        return None

    def resolve_text_prefix_cache(self) -> bool:
        """Return whether this Qwen T2I request can use the validated cache path.

        Unsupported T2I cache requests fall back to the original uncached forward.
        Editing uses its existing reference-cache path and does not call this gate.
        """
        reason = self._text_prefix_cache_ineligibility_reason()
        if reason:
            logger.warning("Qwen-Image 2.1 T2I prefix cache disabled: %s", reason)
            return False
        return True

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.dit = load_qwen_image21_dit(
            config.dit_path, device=self._load_target(config.dit_path), dtype=config.dtype
        )
        qkv_projections = [
            projection
            for block in self.dit.transformer_blocks
            for projection in (block.attn.to_q, block.attn.to_k, block.attn.to_v)
        ]
        int8_qkv = bool(qkv_projections) and all(
            self._supports_tested_int8_projection(projection)
            for projection in qkv_projections
        )
        self._text_prefix_cache_profile_supported = self._supports_text_prefix_cache(self.dit)
        if not self._text_prefix_cache_profile_supported:
            self.KV_CACHE_WITHOUT_REFERENCE_REASON = (
                "Qwen-Image 2.1 text-to-image KV caching requires the validated "
                "INT8-ConvRot projection profile; BF16 caching remains disabled "
                "because gfx1151 GEMM rounding changed final latents and pixels"
            )
        else:
            self.KV_CACHE_WITHOUT_REFERENCE_REASON = self._text_prefix_cache_runtime_reason(
                self.device
            )
        self.KV_CACHE_WITHOUT_REFERENCE = self.KV_CACHE_WITHOUT_REFERENCE_REASON is None
        if int8_qkv:
            if not self.dit.fuse_qkv_projections():
                logger.warning("INT8 Qwen-Image 2.1 Q/K/V fusion is unsupported; using separate projections")
            else:
                logger.info("Fused Qwen-Image 2.1 INT8 Q/K/V projections")
        if not self.KV_CACHE_WITHOUT_REFERENCE:
            logger.info("Qwen-Image 2.1 T2I prefix caching is unavailable: %s",
                        self.KV_CACHE_WITHOUT_REFERENCE_REASON)

        # A fused-MLP checkpoint needs a LoRA's split SwiGLU names fused onto gate_up.
        self.lora_fusions = FUSE_GATE_UP if self.dit.params.fused_mlp else {}

        # Per-run state, filled in by ``prepare_latent``.
        self._seqs: dict[str, QwenImage21Sequence] = {}

        logger.info("Loading Qwen-Image 2.1 text encoder (Qwen3-VL-8B) from %s", config.text_encoder_path)
        self.text_encoder: QwenImage21TextEncoder = load_qwen_image21_text_encoder(
            config.text_encoder_path, dtype=config.dtype, device=self._load_target(config.text_encoder_path)
        )

        self.vae = load_wan22_vae(self.vae_path, device=self._load_target(self.vae_path))
        if self.vae.z_dim != self.dit.in_channels:
            raise ValueError(
                f"the VAE's {self.vae.z_dim}ch latent does not feed this DiT's "
                f"{self.dit.in_channels}ch input"
            )

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info("Qwen-Image 2.1 model ready on %s (%s)", config.device, config.dtype)

    def switch_loras(self, lora_specs, dit: torch.nn.Module) -> None:
        """Keep Q/K/V weight aliases valid when LoRA weights are changed or restored."""
        if not lora_specs and self._active_lora_result is None:
            return super().switch_loras(lora_specs, dit)
        if self._make_lora_spec_hash(lora_specs) == self._active_lora_spec:
            return super().switch_loras(lora_specs, dit)

        was_fused = any(
            block.attn.to_qkv is not None for block in dit.transformer_blocks
        )
        if was_fused:
            dit.unfuse_qkv_projections()
        super().switch_loras(lora_specs, dit)
        if was_fused and not self._lora_state_unfit:
            dit.fuse_qkv_projections()

    # ------------------------------------------------------------ kernels
    def _encoder_images(self, args: EncodePromptArgs) -> Optional[list]:
        """``args.image`` (single or list) as RGB, at the size the VAE sees.

        An alpha is composited onto white: the vision tower takes RGB only.
        """
        if args.image is None:
            return None
        images = args.image if isinstance(args.image, list) else [args.image]
        if not images:
            return None
        return [flatten_alpha(self.prepare_reference(img)) for img in images]

    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        """Prompt (and, when editing, the references) -> embeddings + image slots.

        One prompt at a time needs no padding, so there is no attention mask: the
        DiT splices the reference latents into the vision tokens' places.
        """
        images = self._encoder_images(args)
        cond, cond_slots = self.text_encoder(args.prompt, images)
        null = null_slots = None
        if args.guidance_scale > 1.0:
            null, null_slots = self.text_encoder(args.negative_prompt, images)
        return QwenImage21Conditioning(
            cond=cond, null=null, cond_slots=cond_slots, null_slots=null_slots
        )

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        shape = (
            1, self.vae.z_dim,
            params.height // self.vae.spatial_compression,
            params.width // self.vae.spatial_compression,
        )
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
        ref: Optional[torch.Tensor] = None,
        ref_method: str = "index",
    ) -> torch.Tensor:
        """Build each branch's causal sequence once for the run and keep the latent.

        The sequence holds the *projected* text + reference tokens and their RoPE
        table, so a denoise step is only the target-image tokens plus the block loop.
        Each branch gets its own, keyed ``cond`` / ``uncond`` like the KV caches.
        """
        cache_params = params
        if params.kv_cache and ref is None and not self.resolve_text_prefix_cache():
            # Keep the requested settings immutable to callers, but safely run the
            # original uncached path for an unqualified T2I request.
            cache_params = replace(params, kv_cache=False)
        dev = torch.device(self.device)
        x = latents.to(device=dev, dtype=self.dtype)

        refs = None
        if ref is not None:
            refs = [self.pack_reference_latent(r, ref_method, ref_index=i + 1)[0]
                    for i, r in enumerate(ref)]

        self._seqs = {
            "cond": self.dit.build_sequence(
                x, cond.cond, refs, getattr(cond, "cond_slots", None),
                name="seq", dtype=self.dtype,
            )
        }
        if cond.null is not None:
            self._seqs["uncond"] = self.dit.build_sequence(
                x, cond.null, refs, getattr(cond, "null_slots", None),
                name="seq_uncond", dtype=self.dtype,
            )

        self.start_kv_caches(
            cache_params,
            has_reference=(
                refs is not None
                or (cache_params.kv_cache and self._seqs["cond"].prefix is not None)
            ),
            has_uncond=cache_params.guidance_scale > 1.0 and "uncond" in self._seqs,
        )
        return x

    def schedule(self, params: SamplingParams) -> list[Step]:
        # One token per latent cell, so the shift's token count is the 16x grid.
        comp = self.vae.spatial_compression
        image_seq_len = (params.height // comp) * (params.width // comp)
        ts = qwen21_sampling.get_schedule(params.steps, image_seq_len)
        return [Step(t=ts[i], delta=ts[i] - ts[i + 1]) for i in range(params.steps)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One DiT forward (+ CFG) returning the velocity, in canonical latent form."""
        dev = torch.device(self.device)
        t_full = torch.full((1,), float(t), dtype=latents.dtype, device=dev)
        with torch.autocast(device_type=dev.type, dtype=self.dtype):
            pos = self.dit(latents, t_full, self._seqs["cond"], self.kv_cache("cond"))
            if guidance_scale > 1.0 and "uncond" in self._seqs:
                neg = self.dit(latents, t_full, self._seqs["uncond"], self.kv_cache("uncond"))
                v = neg + guidance_scale * (pos - neg)
            else:
                v = pos
        return v

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        self.end_kv_caches()
        self._seqs = {}
        return latents

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Align to 32 pixels: two latent cells, and one Qwen3-VL vision token.

        The edit path needs the extra factor: the reference latent replaces whole
        32x32 vision tokens.
        """
        align = 2 * self.vae.spatial_compression
        return round_up(width, align), round_up(height, align)

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) -> canonical reference latent."""
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "index_timestep_zero",
        ref_index: int = 1,
    ) -> Tuple[torch.Tensor, None]:
        """Validate the reference method and hand back the latent unchanged."""
        if method != "index_timestep_zero":
            raise ValueError(
                f"unsupported ref_latents_method {method!r}; Qwen-Image 2.1 always "
                "conditions its text/reference prefix at timestep zero"
            )
        dev = torch.device(self.device)
        return latents.to(device=dev, dtype=self.dtype), None

    # -------------------------------------------------------------- upscaling
    def _create_upscaler(self) -> LatentUpscaler:
        return Qwen21TranscodeUpscaler(self.vae, device=self.device, dtype=self.dtype)


__all__ = ["QwenImage21Conditioning", "QwenImage21Model"]
