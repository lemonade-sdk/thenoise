"""Flux.3 adapter — mode-routed flow DiT on the Flux.2 latent, Qwen3-VL-4B conditioned.

The released ``f3-image`` checkpoints are step-distilled **flow maps**: 15 Euler
steps at guidance 1.0 (the reference default), where every modulation additionally
carries the timestep the step jumps to. Latent geometry is Flux.2's packed 128-channel latent at 16 pixels per
token, so the VAE, the token packing and the latent upscaler are all Flux.2's; what is
new is the DiT (``dit/flux3``) and its Qwen3-VL-4B conditioner (``dit/flux3/encoder``).

Editing gives the DiT one or more reference latents. They are embedded by their own
``emb_in.image_cond``, modulated at **timestep zero**, and placed on negative RoPE
``l`` axes so they never collide with the target's positions. The joint sequence is
``[image, references, text]``.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from thenoise.dit.flux3.encoder import load_flux3_text_encoder
from thenoise.dit.flux3.models import (
    Flux3Params,
    image_ids,
    latent_to_tokens,
    ref_ids,
    text_ids,
    tokens_to_latent,
)
from thenoise.dit.flux3.sampling import get_schedule, target_times
from thenoise.dit.flux3.utils import detect_params, load_flux3_dit
from thenoise.models.base import Conditioning, DiffusionModel, Step, normalize_keys
from thenoise.models.config import EncodePromptArgs, ModelConfig, SamplingParams
from thenoise.upscale import LatentUpscaler, SesquiLSRUpscaler
from thenoise.utils.math import round_up
from thenoise.vae import load_flux2_vae

logger = logging.getLogger(__name__)


class Flux3Model(DiffusionModel):
    name = "flux3"

    # The reference samples this distilled flow map at 15 steps, guidance 1.0 — more
    # than the other distilled models here, because a flow map is conditioned on the
    # step size as well as the noise level. A base (CFG) release would want guidance > 1.
    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "steps": 15,
        "guidance_scale": 1.0,
        "sampler": "euler",
    }

    # Reference-latent editing, no KV cache (see the module docstring).
    CAPABILITIES = {**DiffusionModel.CAPABILITIES, "edit": True, "kv_cache": False}

    # The DiT fuses q/k/v even where the checkpoint does not; a LoRA trained on the
    # separate projections is stacked onto it before matching.
    lora_fusions = {"qkv": ("q_proj", "k_proj", "v_proj")}

    #: Flux.3's mode routing is unique among the models here: per-mode blocks and
    #: per-mode joint modulations. (The same fingerprint ComfyUI detects on.)
    _FINGERPRINT = ("content_mode_blocks.image.0.attn_out.", "single_stream_modulations.txt.")

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is a Flux.3 DiT: its per-mode blocks and modulations."""
        keys = list(normalize_keys(f.keys()))
        return all(
            any(key.startswith(prefix) for key in keys) for prefix in Flux3Model._FINGERPRINT
        )

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)

        self.params: Flux3Params = detect_params(config.dit_path)
        # A flow map modulates every stream with the timestep each step jumps to.
        self.flow_map = self.params.flow_map

        logger.info("Loading Flux.3 DiT from %s", config.dit_path)
        self.dit = load_flux3_dit(
            config.dit_path, self.params, device=self.offload_device, dtype=config.dtype
        )

        logger.info(
            "Loading Flux.3 text encoder (Qwen3-VL-4B) from %s", config.text_encoder_path
        )
        self.text_encoder = load_flux3_text_encoder(
            config.text_encoder_path, dtype=config.dtype, device=self.offload_device
        )

        logger.info("Loading Flux.3 VAE (Flux.2) from %s", self.vae_path)
        self.vae = load_flux2_vae(self.vae_path, device=self.device)

        self.memory.register("dit", self.dit)
        self.memory.register("text_encoder", self.text_encoder)
        self.memory.register("vae", self.vae)

        logger.info(
            "Flux.3 model ready on %s (%s)%s",
            config.device, config.dtype, " [flow map]" if self.flow_map else "",
        )

    # ------------------------------------------------------------ kernels
    def encode_prompt(self, args: EncodePromptArgs) -> Conditioning:
        """Encode the prompt (and the negative, under CFG).

        Reference images do NOT reach the conditioner: Flux.3 conditions on reference
        *latents*, which the pipeline hands to ``prepare_latent``.
        """
        cond = self.text_encoder(args.prompt)  # [1, L, 8 * 2560]
        null = None
        if args.guidance_scale > 1.0:
            null = self.text_encoder(args.negative_prompt)
        return Conditioning(cond=cond, null=null)

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
        ref: Optional[List[torch.Tensor]] = None,
        ref_method: str = "index",
    ) -> torch.Tensor:
        """Pack the latent into tokens and stash conditioning + positions, ONCE.

        Positions are built once for the whole ``[image, references, text]`` sequence
        and cached as rotary frequencies under the stream they belong to (the DiT
        concatenates them per step). The timestep-zero streams then run their early
        blocks here — but only when they are step-invariant, i.e. when the DiT is not
        a flow map.
        """
        dev = torch.device(self.device)
        x = latent_to_tokens(latents.to(device=dev, dtype=self.dtype))

        self._txt = cond.cond.to(device=dev, dtype=self.dtype)
        self._un_txt = None if cond.null is None else cond.null.to(device=dev, dtype=self.dtype)
        self._branches = ("cond", "uncond") if self._un_txt is not None else ("cond",)

        # Each reference goes on its own negative ``l`` axis: 1st -> -1, 2nd -> -2, ...
        self._ref_tokens = None
        ref_shapes: List[Tuple[int, int]] = []
        if ref is not None:
            tokens = []
            for i, ref_latent in enumerate(ref, start=1):
                ref_tokens, _ = self.pack_reference_latent(ref_latent, ref_method, i)
                tokens.append(ref_tokens)
                ref_shapes.append((ref_latent.shape[-2], ref_latent.shape[-1]))
            self._ref_tokens = torch.cat(tokens, dim=1)

        self.dit.pe_embedder.clear()
        self.dit.pe_embedder.store(
            "img",
            image_ids(latents.shape[-2], latents.shape[-1], device=dev),
            dtype=self.dtype,
        )
        self.dit.pe_embedder.store("txt", text_ids(self._txt.shape[1], device=dev), dtype=self.dtype)
        if self._un_txt is not None:
            self.dit.pe_embedder.store(
                "txt_uncond", text_ids(self._un_txt.shape[1], device=dev), dtype=self.dtype
            )
        if self._ref_tokens is not None:
            self.dit.pe_embedder.store("ref", ref_ids(ref_shapes, device=dev), dtype=self.dtype)

        # The timestep-zero streams, kept across steps when they do not change.
        self._txt_states: Dict[str, torch.Tensor] = {}
        self._ref_states: Optional[torch.Tensor] = None
        if not self.flow_map:
            # Row 1 of the timestep-zero vector is exactly ``time_in(0)``: what the
            # text and reference early blocks are driven by for the whole run.
            with torch.autocast(device_type=dev.type, dtype=self.dtype):
                vec_zero = self.dit.timestep_vec(torch.zeros(1, device=dev, dtype=self.dtype))[1:]
                self._ref_states = self._reference_states(vec_zero)
                for branch in self._branches:
                    self._txt_states[branch] = self._text_states(branch, vec_zero)
        return x

    def _text_states(self, branch: str, vec_zero: torch.Tensor) -> torch.Tensor:
        """One CFG branch's text stream. Each branch has its own context, so it has
        its own stream and its own position table."""
        ctx = self._txt if branch == "cond" else self._un_txt
        pe = self.dit.pe_embedder["txt" if branch == "cond" else "txt_uncond"]
        return self.dit.text_states(ctx, vec_zero, pe)

    def _reference_states(self, vec_zero: torch.Tensor) -> Optional[torch.Tensor]:
        """The reference stream — shared by both branches, which differ only in text."""
        if self._ref_tokens is None:
            return None
        return self.dit.reference_states(
            self._ref_tokens, vec_zero, self.dit.pe_embedder["ref"]
        )

    def begin_sampling(self, schedule: list[Step]) -> None:
        """Remember where each step is going: a flow map embeds its target timestep."""
        self._targets = target_times(schedule)

    def schedule(self, params: SamplingParams) -> list[Step]:
        image_ts = get_schedule(params.steps)
        # Step.t is the flow timestep (1 -> 0); delta = t_i - t_{i+1}.
        return [Step(t=image_ts[i], delta=image_ts[i] - image_ts[i + 1])
                for i in range(params.steps)]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One Flux.3 forward (+ CFG), returning the raw output as velocity.

        The Flux.3 flow ODE integrates ``x += (t_prev - t_curr) * v``, which is the
        shared Euler update ``x -= delta * v`` when ``v`` is the model's raw output.
        """
        dev = torch.device(self.device)
        t_full = torch.full((1,), float(t), dtype=latents.dtype, device=dev)
        target = None
        if self.flow_map:
            target = torch.full((1,), self._targets[i], dtype=latents.dtype, device=dev)
        # Row 0 modulates the target at ``t``, row 1 the rest at timestep zero (plus
        # the jump target on a flow map).
        vec = self.dit.timestep_vec(t_full, target)

        pe_img = self.dit.pe_embedder["img"]
        pe_ref = self.dit.pe_embedder["ref"] if self._ref_tokens is not None else None
        with torch.autocast(device_type=dev.type, dtype=self.dtype):
            # A flow map's timestep-zero rows move with every step; otherwise the
            # streams prepared once per run still stand.
            if self.flow_map:
                refs = self._reference_states(vec[1:])
                txt = {b: self._text_states(b, vec[1:]) for b in self._branches}
            else:
                refs, txt = self._ref_states, self._txt_states

            pos = self.dit(
                latents, vec, pe_img, txt["cond"], self.dit.pe_embedder["txt"], refs, pe_ref
            )
            if guidance_scale > 1.0 and "uncond" in txt:
                neg = self.dit(
                    latents, vec, pe_img, txt["uncond"],
                    self.dit.pe_embedder["txt_uncond"], refs, pe_ref,
                )
                v = neg + guidance_scale * (pos - neg)
            else:
                v = pos
        return v

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Unpack the target tokens back to the canonical packed latent."""
        return tokens_to_latent(
            latents,
            params.height // self.vae.spatial_compression,
            params.width // self.vae.spatial_compression,
        )

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # One token per latent cell: the packed latent already is the DiT's grid.
        align = self.vae.spatial_compression
        return round_up(width, align), round_up(height, align)

    # ------------------------------------------------------------ editing
    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Input pixels (``[C,H,W]`` in [-1, 1]) -> packed latent ``[1, 128, H//16, W//16]``."""
        return self.vae.encode_pixels_to_latents(pixels.unsqueeze(0))

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "index",
        ref_index: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Canonical reference latent -> (tokens, ids) at ``l`` axis ``-ref_index``.

        ``ref_index`` is the 1-based position, so the refs land on -1, -2, ... —
        negative because the target holds ``l = 0`` and the text spans the positive
        ones. Flux.3 has no ``index`` vs ``index_timestep_zero`` choice: references
        are modulated at timestep zero whatever the caller asks for, so both names
        select this one packing.
        """
        if method not in ("index", "index_timestep_zero"):
            raise ValueError(
                f"unsupported ref_latents_method {method!r}; expected 'index' or "
                "'index_timestep_zero' (Flux.3 always conditions its references at "
                "timestep zero, so they are the same packing here)"
            )
        dev = torch.device(self.device)
        tokens = latent_to_tokens(latents.to(device=dev, dtype=self.dtype))
        return tokens, image_ids(
            latents.shape[-2], latents.shape[-1], -ref_index, device=dev
        )

    # -------------------------------------------------------------- upscaling
    def _create_upscaler(self) -> LatentUpscaler:
        return SesquiLSRUpscaler("flux2", device=self.device, dtype=self.dtype)


__all__ = ["Flux3Model"]
