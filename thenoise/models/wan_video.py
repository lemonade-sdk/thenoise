"""Wan 2.2 generation from Comfy-Org checkpoints: one adapter, one variant per
released transformer (TI2V-5B, or the two files of T2V-A14B)."""
from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import torch

from thenoise.dit.wan_video.checkpoint import load_wan_native_transformer
from thenoise.dit.wan_video.model import CHANNELS, Wan22Params, detect_params
from thenoise.dit.wan_video.sampling import FLOW_SHIFT, get_sigmas, wan_frame_grid
from thenoise.models.base import Conditioning
from thenoise.models.config import EncodePromptArgs
from thenoise.models.video_base import Segment, VideoModel
from thenoise.samplers import Step, sigma_steps
from thenoise.text_encoders.wan_video import load_wan_text_encoder
from thenoise.utils.image_tensor import cover_crop
from thenoise.utils.wan_weights import wan_dit_key
from thenoise.vae.wan_video import detect_latent_width, load_wan_video_vae, TEMPORAL_COMPRESSION
from thenoise.video.spec import ResolvedVideo, VideoSpec

log = logging.getLogger("uvicorn.error")


@dataclasses.dataclass(frozen=True)
class Experts:
    """A two-expert schedule: where the sigma hands over, and how the low-noise half
    differs."""

    boundary: float
    flow_shift: float
    #: Upstream ships guidance 4.0 high and 3.0 low.
    low_cfg_ratio: float


@dataclasses.dataclass(frozen=True)
class Variant:
    """What a released Wan 2.2 transformer needs to be driven as."""

    name: str
    prefs: dict
    i2v: bool
    experts: Experts | None


#: One entry per released transformer, keyed by the variant name `CHANNELS` gives the
#: shape `detect_params` reads. `DEFAULT_PREFS` covers what the variants share; these
#: cover what a variant decides.
VARIANTS: dict[str, Variant] = {variant.name: variant for variant in (
    Variant(
        name="ti2v_5b",
        prefs=dict(width=1280, height=704, frames=33, fps=24, steps=12,
                   guidance_scale=5.0, sampler="unipc"),
        i2v=True, experts=None),
    Variant(
        name="t2v_14b",
        prefs=dict(width=832, height=480, frames=33, fps=16, steps=40,
                   guidance_scale=4.0, sampler="euler"),
        i2v=False,
        experts=Experts(boundary=0.875, flow_shift=12.0, low_cfg_ratio=3.0 / 4.0)),
)}


def variant_for(params: Wan22Params) -> Variant:
    """The policy for the variant a checkpoint says this is."""
    for variant in VARIANTS.values():
        if CHANNELS[variant.name] == params.in_channels:
            return variant
    raise ValueError(f"no Wan 2.2 policy for a {params.in_channels}-channel transformer")


class Wan22Model(VideoModel):
    """Wan 2.2 from Comfy-Org checkpoints.

    One adapter covers TI2V-5B and the two-expert T2V-A14B: the DiT header says which
    variant this is, and ``VARIANTS`` says what follows from that — request defaults,
    whether I2V exists at all, which codec, and whether a second expert file is
    required. I2V with a CLIP image tower would be a third variant, not this file.
    """

    name = "wan22"
    # A clip is made bigger by being generated bigger, not by upscaling a latent.
    CAPABILITIES = {**VideoModel.CAPABILITIES, "t2v": True, "upscale": False}
    # The conditioned I2V frame is denoised at timestep zero, which is what the edit
    # contract already calls `index_timestep_zero`; there is no other method to offer.
    DEFAULT_PREFS = {**VideoModel.DEFAULT_PREFS, "ref_method": "index_timestep_zero"}

    @staticmethod
    def detect(f) -> bool:
        """Any Wan 2.2 transformer: which variant it is comes from its shapes, later."""
        from thenoise.utils.safetensors import unwrap_key
        keys = {unwrap_key(k) for k in f.keys()}
        return {"patch_embedding.weight", "blocks.0.self_attn.q.weight",
                "text_embedding.0.weight"}.issubset(keys)

    @classmethod
    def validate_paths(cls, dit_paths: dict[str, str]) -> None:
        """A two-expert model is two files, so its second one is not optional."""
        experts = variant_for(detect_params(dit_paths[""])).experts
        if experts is None:
            if len(dit_paths) > 1:
                raise ValueError(
                    "--dit-low carries the low-noise half of a two-expert transformer; "
                    f"{dit_paths['']} has one expert")
            return
        if "low" not in dit_paths:
            raise ValueError(f"{dit_paths['']} is one expert of a two-expert "
                             "transformer: give the other with --dit-low")
        low = Path(dit_paths["low"]).resolve()
        if low == Path(dit_paths[""]).resolve():
            raise ValueError("--dit and --dit-low must be different checkpoints")
        if not low.is_file():
            raise FileNotFoundError(f"low-noise DiT checkpoint not found: {low}")

    def _lora_key_map(self, key: str) -> str:
        """Map native ComfyUI Wan LoRA projection names to our DiT module paths."""
        for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha"):
            if not key.endswith(suffix):
                continue
            stem = key[:-len(suffix)]
            if stem.startswith("lora_unet_"):
                raw = stem.removeprefix("lora_unet_")
                match = re.fullmatch(r"blocks_(\d+)_(self_attn|cross_attn)_(q|k|v|o)", raw)
                ffn = re.fullmatch(r"blocks_(\d+)_ffn_(0|2)", raw)
                if match:
                    stem = f"blocks.{match[1]}.{match[2]}.{match[3]}"
                elif ffn:
                    stem = f"blocks.{ffn[1]}.ffn.{ffn[2]}"
                else:
                    return key  # unknown training schema: never guess a projection
            stem = (stem.removeprefix("transformer.")
                    .removeprefix("model.diffusion_model.")
                    .removeprefix("diffusion_model."))
            return wan_dit_key(stem + ".weight").removesuffix(".weight") + suffix
        return key

    def __init__(self, *, config):
        super().__init__(config=config)
        self.params: Wan22Params = detect_params(self.dit_path)
        self.variant = variant_for(self.params)
        # `i2v` is the one capability a variant, rather than the model, decides.
        self.CAPABILITIES = {**self.CAPABILITIES, "i2v": self.variant.i2v}
        # Both headers are read before either module is built: the clip's latent is the
        # only thing that crosses between them and the normalization table is keyed by
        # its width, so a mismatched `--vae` is refused here rather than mid-run.
        codec_width = detect_latent_width(self.vae_path)
        if codec_width != self.params.in_channels:
            raise ValueError(
                f"--vae is a {codec_width}-channel codec, and {self.dit_path} denoises a "
                f"{self.params.in_channels}-channel latent: these two checkpoints are "
                "from different Wan releases")

        # The run's I2V conditioning: (first-frame latent, per-token timestep mask).
        self._clip = None
        self.dit = load_wan_native_transformer(self.dit_path, self._load_target(self.dit_path))
        self.low_dit = None
        if self.variant.experts is not None:
            low_path = self.dit_paths["low"]
            self.low_dit = load_wan_native_transformer(low_path, self._load_target(low_path))
        self.text_encoder, self.tokenizer = load_wan_text_encoder(
            self.text_encoder_path, self._load_target(self.text_encoder_path))
        self.vae = load_wan_video_vae(self.vae_path, self._load_target(self.vae_path))
        self.memory.register("dit", self.dit)
        if self.low_dit is not None:
            self.memory.register("dit_low", self.low_dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)
        log.info("Wan 2.2 %s loaded (%s)", self.variant.name,
                 "two experts" if self.low_dit is not None else "one transformer")

    # ------------------------------------------------------------ latent upscaler
    def _create_upscaler(self):
        raise NotImplementedError(f"{self.name} has no latent upscaler")

    # ------------------------------------------------------------ preferences
    @property
    def default_prefs(self) -> dict[str, Any]:
        """The class defaults under this variant's, a checkpoint marker still last."""
        return {**self.DEFAULT_PREFS, **self.variant.prefs, **self.checkpoint_prefs}

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Rounded down onto the 32 pixels a latent cell covers after patching."""
        size = (width // 32 * 32, height // 32 * 32)
        if 0 in size:
            raise ValueError("Wan video requires width and height of at least 32")
        return size

    def resolve_video(self, spec: VideoSpec) -> ResolvedVideo:
        """Apply this variant's defaults and geometric restrictions."""
        resolved = super().resolve_video(spec)
        if resolved.sampler not in ("euler", "unipc"):
            raise ValueError("Wan 2.2 supports only euler or unipc samplers")
        if resolved.image is not None and not self.variant.i2v:
            raise ValueError("Wan 2.2 T2V-A14B does not support I2V (requires CLIP vision)")
        return dataclasses.replace(
            resolved,
            frame_count=wan_frame_grid(resolved.frame_count,
                                       compression=TEMPORAL_COMPRESSION,
                                       upward=resolved.explicit_frames),
        )

    # ------------------------------------------------------------ teardown
    def _clear_weights(self) -> None:
        self.vae = self.text_encoder = self.tokenizer = None
        self.dit = self.low_dit = None

    def switch_loras(self, lora_specs, dit: torch.nn.Module) -> None:
        # LoRA bookkeeping is one DiT's: `LoRASupport` records a single active spec, so a
        # second expert would be left out in silence. Role-keyed specs (plan C14) are the
        # way to lift this.
        if lora_specs and self.variant.experts is not None:
            raise ValueError("Wan 2.2 T2V-A14B expert-specific LoRAs are not supported")
        super().switch_loras(lora_specs, dit)

    # ------------------------------------------------------------ kernels
    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        """UMT5 features for the prompt, and for the negative one when guidance runs."""
        return Conditioning(cond=self._embed(args.prompt),
                            null=(self._embed(args.negative_prompt)
                                  if args.guidance_scale > 1 else None))

    def _embed(self, prompt: str) -> torch.Tensor:
        ids, mask = self.tokenizer.encode(prompt)
        features = self.text_encoder(ids.to(self.device), mask.to(self.device))
        return features.to(dtype=self.dit.dtype)

    def fit_references(self, images, target):
        """I2V's initial frame fills the canvas the clip is sampled on."""
        return [cover_crop(image, target[0], target[1]) for image in images]

    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """One fitted frame, ``[C, H, W]`` in [-1, 1] -> the canonical latent."""
        return self.vae.encode_pixels_to_latents(pixels[None, :, None].to(self.device))

    def init_latents(self, params) -> torch.Tensor:
        """The clip's noise, at the codec's spatial and temporal compression."""
        shape = (1, self.vae.z_dim,
                 (params.frames - 1) // self.vae.temporal_compression + 1,
                 params.height // self.vae.spatial_compression,
                 params.width // self.vae.spatial_compression)
        rng = torch.Generator(device="cpu").manual_seed(params.seed)
        latents = torch.randn(shape, generator=rng, dtype=torch.float32)
        return latents.to(self.device)

    def prepare_latent(self, latents, cond, params, ref=None, ref_method="index"):
        """Seed the clip; with a reference, condition its first frame at timestep zero."""
        if not ref:
            return latents
        if ref_method != "index_timestep_zero":
            raise ValueError(
                "Wan 2.2 I2V conditions its first frame at timestep zero; it has no"
                f" '{ref_method}' path")
        if len(ref) != 1:
            raise ValueError(f"Wan 2.2 I2V takes one initial frame, got {len(ref)}")
        condition = ref[0]
        if condition.shape != latents[:, :, :1].shape:
            raise ValueError("I2V first-frame latent shape mismatch")
        patch_t, patch_h, patch_w = self.dit.params.patch_size
        b, _, t, h, w = latents.shape
        mask = torch.ones((b, 1, t, 1, 1), device=latents.device, dtype=torch.float32)
        mask[:, :, 0] = 0
        latents = latents.clone()
        latents[:, :, :1] = condition.to(latents.dtype)
        # A token's timestep is zero on the conditioned frame, so the mask is downsampled
        # to the token grid the DiT sees.
        patch_mask = mask.expand(-1, -1, -1, h, w)
        patch_mask = patch_mask[:, 0, ::patch_t, ::patch_h, ::patch_w].flatten(1)
        self._clip = (condition.to(self.dit.dtype), patch_mask)
        return latents

    def finalize_latent(self, latents, params):
        """Hand the conditioned frame back: it never took a denoising step.

        Every forward already re-injected it, so this only decides what the codec sees.
        """
        if self._clip is not None:
            latents[:, :, :1] = self._clip[0].to(latents.dtype)
            self._clip = None
        return latents

    def schedule(self, params) -> list[Step]:
        experts = self.variant.experts
        shift = experts.flow_shift if experts is not None else FLOW_SHIFT
        return sigma_steps(get_sigmas(params.steps, shift=shift),
                           self.device, torch.float32)

    def segments(self, params, schedule, guidance_scale) -> list[Segment]:
        """The two halves of the sigma range, one expert each.

        Both spans are handed over as separate stages and nothing here says they cannot
        share a device: the memory manager keeps the second expert resident when there is
        room and displaces the first when there is not.
        """
        experts = self.variant.experts
        if experts is None:
            return super().segments(params, schedule, guidance_scale)
        spans, start = [], 0
        steps = list(schedule)
        while start < len(steps):
            high = float(steps[start].t) > experts.boundary
            end = start + 1
            while end < len(steps) and (float(steps[end].t) > experts.boundary) == high:
                end += 1
            spans.append(Segment(
                stage="dit" if high else "dit_low",
                module=self.dit if high else self.low_dit,
                steps=tuple(steps[start:end]),
                guidance_scale=(guidance_scale if high
                                else guidance_scale * experts.low_cfg_ratio)))
            start = end
        return spans

    def denoise_step(self, latents, t, cond, guidance_scale: float, i: int) -> torch.Tensor:
        module = self._segment.module
        x = latents.to(module.dtype)
        condition, patch_mask = (None, None) if self._clip is None else self._clip
        if condition is not None:
            x = x.clone()
            x[:, :, :1] = condition
        timestep = float(t) * 1000.0
        if patch_mask is None:
            # A per-batch timestep keeps the modulation [B, 6, D] instead of the token
            # grid's, which is the difference between fitting and not fitting.
            timestep = torch.full((x.shape[0],), timestep,
                                  device=x.device, dtype=torch.float32)
        else:
            timestep = patch_mask * timestep
        vel = self._velocity(module, x, timestep, cond.cond)
        if cond.null is None:
            return vel
        uncond = self._velocity(module, x, timestep, cond.null)
        return uncond + guidance_scale * (vel - uncond)

    @staticmethod
    def _velocity(module, x, timestep, context) -> torch.Tensor:
        return module(x, timestep=timestep, encoder_hidden_states=context).float()

    # ------------------------------------------------------------ codec
    def decode_frames(self, latents: torch.Tensor) -> Iterator[torch.Tensor]:
        for video in self.vae.decode_to_pixel_frames(latents):
            for i in range(video.shape[2]):
                yield video[0, :, i].float()
