"""Wan 2.2 generation from Comfy-Org checkpoints: one adapter, one variant per
released transformer (TI2V-5B, or the two files of T2V-A14B)."""
from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import torch

from thenoise.dit.wan_video import sampling as wan_sampling
from thenoise.dit.wan_video.checkpoint import load_wan_native_transformer
from thenoise.dit.wan_video.model import (
    CHANNELS,
    SIGNATURE as WAN_SIGNATURE,
    Wan22Params,
    detect_params,
)
from thenoise.dit.wan_video.sparse import wan_plan, window_sigmas
from thenoise.models.base import Conditioning
from thenoise.models.config import EncodePromptArgs, VideoSamplingParams
from thenoise.models.pixel_base import expect_latent_width
from thenoise.models.video_base import Segment, VideoModel
from thenoise.samplers import Step, sigma_steps
from thenoise.utils.attention import DEFAULT_ATTENTION
from thenoise.text_encoders.wan_video import load_wan_text_encoder
from thenoise.utils.image_tensor import cover_crop
from thenoise.utils.safetensors import Header
from thenoise.utils.sol_attention import SolAttention
from thenoise.utils.wan_weights import wan_dit_key
from thenoise.vae.wan_video import detect_latent_width, load_wan_video_vae, TEMPORAL_COMPRESSION
from thenoise.video.spec import ResolvedVideo, VideoSpec, frame_grid

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ClipReference:
    """What a reference contributes to every forward pass of one clip.

    The two halves are the two ways Wan 2.2 conditions on an image, and a variant uses
    exactly one of them.
    """

    #: TI2V-5B: a latent frame re-injected before every forward and never denoised.
    first_frame: torch.Tensor | None = None
    #: TI2V-5B: per-token timestep, zero on the conditioned frame.
    timestep_mask: torch.Tensor | None = None
    #: I2V-A14B: mask and reference clip, concatenated onto the noise.
    channels: torch.Tensor | None = None


@dataclasses.dataclass(frozen=True)
class Experts:
    """A two-expert schedule: where the sigma hands over, and how the low-noise half
    differs."""

    boundary: float
    #: The low-noise half's CFG as a fraction of the request's. T2V-A14B ships 3.0
    #: against a high-noise 4.0; the I2V pair keeps the same value on both halves.
    low_cfg_ratio: float


@dataclasses.dataclass(frozen=True)
class Variant:
    """What a released Wan 2.2 transformer needs to be driven as."""

    name: str
    prefs: dict
    i2v: bool
    experts: Experts | None
    #: How a reference enters this transformer, when it takes one at all. Unset means
    #: the model default, which is the right answer for every variant but the one
    #: whose width says the reference is concatenated.
    ref_method: str | None = None

    @property
    def flf2v(self) -> bool:
        """Whether this transformer can be told where its clip ends."""
        return self.ref_method == "concat_channels"


#: The channels of a concatenated reference that say which latent frames are given.
REFERENCE_MASK_CHANNELS = 4


#: One entry per released transformer, keyed by the variant name `CHANNELS` gives the
#: shape `detect_params` reads. `DEFAULT_PREFS` covers what the variants share; these
#: cover what a variant decides.
VARIANTS: dict[str, Variant] = {variant.name: variant for variant in (
    Variant(
        name="ti2v_5b",
        prefs=dict(width=1280, height=704, frames=33, fps=24, steps=12,
                   guidance_scale=5.0, sampler="unipc", flow_shift=5.0),
        i2v=True, experts=None),
    Variant(
        name="t2v_14b",
        prefs=dict(width=832, height=480, frames=33, fps=16, steps=40,
                   guidance_scale=4.0, sampler="euler", flow_shift=12.0),
        i2v=False,
        experts=Experts(boundary=0.875, low_cfg_ratio=3.0 / 4.0)),
    Variant(
        name="i2v_14b",
        prefs=dict(width=832, height=480, frames=33, fps=16, steps=40,
                   guidance_scale=3.5, sampler="euler", flow_shift=5.0),
        i2v=True,
        # The pair's own release defaults, which the T2V pair does not share: one is not
        # the other, so a change to either is its own decision.
        experts=Experts(boundary=0.900, low_cfg_ratio=1.0),
        ref_method="concat_channels"),
)}


def variant_for(params: Wan22Params) -> Variant:
    """The policy for the variant a checkpoint says this is."""
    for variant in VARIANTS.values():
        if CHANNELS[variant.name] == params.in_channels:
            return variant
    raise ValueError(f"no Wan 2.2 policy for a {params.in_channels}-channel transformer")


def _given_mask(reference: torch.Tensor, ends: int) -> torch.Tensor:
    """The 4 channels naming which latent frames of ``reference`` are given. Upstream's
    layout: both ends given marks all four channels of the first latent frame, and of the
    last only the sub-channel its final pixel frame lands in."""
    mask = torch.zeros((1, REFERENCE_MASK_CHANNELS, *reference.shape[2:]),
                       device=reference.device, dtype=reference.dtype)
    mask[:, :, 0] = 1.0
    if ends > 1:
        mask[:, 3, -1] = 1.0
    return mask


class Wan22Model(VideoModel):
    """Wan 2.2 from Comfy-Org checkpoints.

    One adapter covers TI2V-5B and the two A14B pairs: the DiT header says which
    variant this is, and ``VARIANTS`` says what follows from that — request defaults,
    whether I2V exists at all, how a reference enters, which codec, and whether a
    second expert file is required. Wan 2.1's I2V, which adds a CLIP vision tower and
    an `img_emb` projection to the same trunk, would be a different file.
    """

    name = "wan22"
    # A clip is made bigger by being generated bigger, not by upscaling a latent.
    CAPABILITIES = {**VideoModel.CAPABILITIES, "upscale": False,
                    "video": {**VideoModel.CAPABILITIES["video"], "t2v": True}}
    # The answer for the variants that say nothing: the conditioned I2V frame is
    # denoised at timestep zero, which is what the edit contract calls
    # `index_timestep_zero`.
    DEFAULT_PREFS = {**VideoModel.DEFAULT_PREFS, "ref_method": "index_timestep_zero"}
    ATTENTION_METHODS = (*VideoModel.ATTENTION_METHODS, "optimized")

    def pref(self, name: str, request_value=None):
        """Which transformer this is decides how a reference enters, so that one
        setting comes from the variant: nothing the caller asks for can make a
        16-channel trunk take a concatenated one."""
        if name == "ref_method" and self.variant.ref_method is not None:
            return self.variant.ref_method
        return super().pref(name, request_value)

    #: Any Wan 2.2 transformer: which variant it is comes from its shapes, later.
    SIGNATURE = WAN_SIGNATURE

    @classmethod
    def validate_paths(cls, checkpoints: dict[str, Header]) -> None:
        """A two-expert model is two files, so its second one is not optional."""
        cls.require_same_model(checkpoints)
        primary = checkpoints[""]
        experts = variant_for(detect_params(primary)).experts
        if experts is None:
            if len(checkpoints) > 1:
                raise ValueError(
                    "--dit-low carries the low-noise half of a two-expert transformer; "
                    f"{primary.path} has one expert")
            return
        if "low" not in checkpoints:
            raise ValueError(f"{primary.path} is one expert of a two-expert "
                             "transformer: give the other with --dit-low")
        if Path(checkpoints["low"].path).resolve() == Path(primary.path).resolve():
            raise ValueError("--dit and --dit-low must be different checkpoints")

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
        self.params: Wan22Params = detect_params(self.header)
        self.variant = variant_for(self.params)
        # `i2v` and `flf2v` are the capabilities a variant, rather than the model, decides.
        self.CAPABILITIES = {**self.CAPABILITIES,
                             "video": {**self.CAPABILITIES["video"],
                                       "i2v": self.variant.i2v,
                                       "flf2v": self.variant.flf2v, "ref2v": False}}
        # Both headers are read before either module is built: the clip's latent is the
        # only thing that crosses between them and the normalization table is keyed by
        # its width, so a mismatched `--vae` is refused here rather than mid-run.
        codec_width = detect_latent_width(self.vae_path)
        # A transformer whose reference is concatenated is wider than the codec by the
        # mask and the reference latent itself: noise + 4 + reference.
        expect_latent_width(self.params.in_channels, codec_width,
                            also=(2 * codec_width + REFERENCE_MASK_CHANNELS,))

        self._clip: ClipReference | None = None
        self.dit = load_wan_native_transformer(self.header, self._load_target(self.dit_path))
        self.low_dit = None
        if self.variant.experts is not None:
            low = config.header("low")
            self.low_dit = load_wan_native_transformer(low, self._load_target(low.path))
        self.text_encoder, self.tokenizer = load_wan_text_encoder(
            self.text_encoder_path, self._load_target(self.text_encoder_path))
        self.vae = load_wan_video_vae(self.vae_path, self._load_target(self.vae_path))
        self.memory.register("dit", self.dit)
        if self.low_dit is not None:
            self.memory.register("dit_low", self.low_dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)
        logger.info("Wan 2.2 %s loaded (%s)", self.variant.name,
                 "two experts" if self.low_dit is not None else "one transformer")

    # ------------------------------------------------------------ latent upscaler
    def _create_upscaler(self):
        raise NotImplementedError(f"{self.name} has no latent upscaler")

    def lora_components(self) -> dict[str, torch.nn.Module]:
        """Each expert takes its own LoRAs, by the name `Segment.stage` answers for it.

        A variant with one transformer has no `dit_low`, and a component that is not
        there is not a role a request can name.
        """
        return {role: module for role, module in
                {"dit": self.dit, "dit_low": self.low_dit}.items() if module is not None}

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
        """This variant's defaults and sizes, and the frame count the codec can pack."""
        resolved = super().resolve_video(spec)
        return dataclasses.replace(
            resolved,
            frame_count=frame_grid(
                resolved.frame_count, compression=TEMPORAL_COMPRESSION,
                upward=resolved.explicit_frames),
        )

    def default_edit_size(self, width: int, height: int) -> tuple[int, int]:
        """A reference's aspect ratio at the area this variant defaults to.

        The image answer scales to `EDIT_DEFAULT_AREA`, which is a still-image
        number: a 14B clip is asked for 832x480 and should not inherit a reference's
        2K area.
        """
        prefs = self.variant.prefs
        scale = (prefs["width"] * prefs["height"] / (width * height)) ** 0.5
        return round(width * scale), round(height * scale)

    # ------------------------------------------------------------ teardown
    def _clear_weights(self) -> None:
        self.vae = self.text_encoder = self.tokenizer = None
        self.dit = self.low_dit = None

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

    def fuse_text(self, cond: Conditioning) -> Conditioning:
        """UMT5 features -> the cross-attention context, once per segment.

        The projection belongs to the expert, so the controller calls it after
        ``switch_loras`` with that expert resident, and every step of the segment reuses
        the answer instead of paying for the same GEMMs again.
        """
        module = self._segment.module
        return Conditioning(cond=module.fuse_text(cond.cond.to(module.dtype)),
                            null=(None if cond.null is None
                                  else module.fuse_text(cond.null.to(module.dtype))))

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
        dev = self.device
        rng = torch.Generator(device=dev).manual_seed(params.seed)
        return torch.randn(shape, generator=rng, device=dev, dtype=torch.float32)

    def encode_references(self, images, frames=1):
        """One latent per frame, or the 20 channels a concatenated reference contributes.
        Upstream's clip is the frames given and mid-grey (zeros here) everywhere else, and
        only here is it known how many frames were given."""
        if self.pref("ref_method") != "concat_channels":
            return super().encode_references(images, frames)
        first, *rest = images
        clip = first.new_zeros((frames, *first.shape))
        clip[0] = first
        if rest:
            last, = rest
            clip[-1] = last
        reference = self.vae.encode_pixels_to_latents(
            clip.permute(1, 0, 2, 3)[None].to(self.device))
        return [torch.cat([_given_mask(reference, ends=1 + len(rest)), reference], dim=1)]

    def prepare_latent(self, latents, cond, params, ref=None, ref_method="index"):
        """Seed the clip with the reference, however this variant takes one."""
        if not ref:
            # A run that died mid-schedule would otherwise condition the next clip.
            self._clip = None
            return latents
        if ref_method == "concat_channels":
            return self._concat_reference(latents, ref[0])
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
        self._clip = ClipReference(first_frame=condition.to(self.dit.dtype),
                                   timestep_mask=patch_mask)
        return latents

    def _concat_reference(self, latents, reference):
        """The 20 channels every step concatenates with, as ``encode_references`` cut them.
        The noise is left alone: these condition the run rather than replacing frames."""
        want = (latents.shape[0], self.params.in_channels - latents.shape[1],
                *latents.shape[2:])
        if tuple(reference.shape) != want:
            raise ValueError(
                f"{self.variant.name} reference conditioning is {tuple(reference.shape)} "
                f"for a clip latent of {tuple(latents.shape)}")
        self._clip = ClipReference(channels=reference.to(self.dit.dtype))
        return latents

    def finalize_latent(self, latents, params):
        """Hand the conditioned frame back: it never took a denoising step.

        Every forward already re-injected it, so this only decides what the codec sees.
        """
        if self._clip is not None:
            if self._clip.first_frame is not None:
                latents[:, :, :1] = self._clip.first_frame.to(latents.dtype)
            self._clip = None
        return latents

    def schedule(self, params) -> list[Step]:
        return sigma_steps(wan_sampling.get_sigmas(params.steps, shift=params.flow_shift),
                           self.device, torch.float32)

    def percent_to_sigma(self, percent: float) -> float:
        """Percent -> sigma under this variant's flow shift."""
        return wan_sampling.percent_to_sigma(percent, shift=self.pref("flow_shift"))

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
                                else guidance_scale * experts.low_cfg_ratio),
                params=params))
            start = end
        return spans

    def denoise_step(self, latents, t, cond, guidance_scale: float, i: int) -> torch.Tensor:
        module = self._segment.module
        method = self._attention
        x = latents.to(module.dtype)
        clip = ClipReference() if self._clip is None else self._clip
        if clip.channels is not None:
            x = torch.cat([x, clip.channels.to(x.dtype)], dim=1)
        elif clip.first_frame is not None:
            x = x.clone()
            x[:, :, :1] = clip.first_frame
        timestep = float(t) * 1000.0
        if clip.timestep_mask is None:
            # A per-batch timestep keeps the modulation [B, 6, D] instead of the token
            # grid's, which is the difference between fitting and not fitting.
            timestep = torch.full((x.shape[0],), timestep,
                                  device=x.device, dtype=torch.float32)
        else:
            timestep = clip.timestep_mask * timestep
        sigma = float(t)
        vel = self._velocity(module, x, timestep, cond.cond,
                             self._sol(module, x, sigma, method))
        if cond.null is None:
            return vel
        uncond = self._velocity(module, x, timestep, cond.null,
                                self._sol(module, x, sigma, method))
        return uncond + guidance_scale * (vel - uncond)

    def _sol(self, module, x: torch.Tensor, sigma: float,
             method: str = DEFAULT_ATTENTION) -> SolAttention | None:
        """Whether this step's self-attention routes, and with what sinks.

        The patchified geometry is all the trunk's policy needs beside the sigma: the token
        count decides whether routing is worth its overhead, the tokens in a frame where the
        first frame's exact range ends. Both come out of the latent shape the step already
        has, so nothing new is plumbed through the sampler.
        """
        pt, ph, pw = module.params.patch_size
        _, _, t, h, w = x.shape
        return wan_plan(tokens=(t // pt) * (h // ph) * (w // pw),
                        tokens_per_frame=(h // ph) * (w // pw), dtype=x.dtype,
                        head_dim=module.params.head_dim, device=x.device, sigma=sigma,
                        window=window_sigmas(self.percent_to_sigma), method=method)

    @staticmethod
    def _velocity(module, x, timestep, context,
                  sol: SolAttention | None = None) -> torch.Tensor:
        return module(x, timestep=timestep, encoder_hidden_states=context, sol=sol).float()

    # ------------------------------------------------------------ codec
    def decode_frames(self, latents: torch.Tensor,
                      params: VideoSamplingParams) -> Iterator[torch.Tensor]:
        for video in self.vae.decode_to_pixel_frames(latents):
            for i in range(video.shape[2]):
                yield video[0, :, i].float()
