"""Mage-Flow DiT — the 12-layer member of the Qwen-Image dual-stream family.

The block is Qwen-Image's and is reused verbatim
(:class:`thenoise.dit.qwen_image.models.QwenImageTransformerBlock`); only ``eps``
differs (1e-6, for the block LayerNorms and the attention QK RMSNorms). Three things
around the block make this model:

  * **patch 1** — one token per latent cell, the raw 128-channel VAE latent with no
    2x2 packing (see :func:`latent_to_tokens`);
  * **unrotated text** — the text stream sits at position 0, where RoPE is the
    identity, so one matrix is stored for it instead of a per-token table;
  * **a bf16-rounded timestep table** — see :class:`MageTimestepProjEmbeddings`.

The timestep embedding has a single row, so reference tokens are modulated at ``t``
like the target ones and their K/V cannot be cached: no ``kv``, no
``timestep_zero_index``, no per-token modulation mask.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from thenoise.dit.qwen_image.models import (
    AdaLayerNormContinuous,
    QwenImageTransformerBlock,
    TimestepEmbedding,
)
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.dynamo import mark_token_axis
from thenoise.utils.loader import load_dit
from thenoise.utils.rope import RopeCache, matrix_rope
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

#: Layer-norm / RMS-norm epsilon of the reference implementation (Qwen-Image runs the
#: same block at 1e-5).
_EPS = 1e-6


@dataclass
class MageFlowParams:
    """The DiT's geometry, read out of the checkpoint header (see ``utils.detect_params``)."""

    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 12
    num_heads: int = 24
    head_dim: int = 128
    context_dim: int = 2560
    axes_dims: Tuple[int, int, int] = (16, 56, 56)
    #: Latent cells per token. 1 = the raw latent pixel is the token (no 2x2 packing).
    patch_size: int = 1

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.head_dim


def latent_to_tokens(latents: torch.Tensor) -> torch.Tensor:
    """Canonical latent ``[B, C, H, W]`` -> tokens ``[B, H*W, C]``.

    A Mage token IS a latent pixel, so unlike ``pack_latents`` there is no 2x2 merge:
    the token width stays ``C``.
    """
    batch, channels, height, width = latents.shape
    return latents.movedim(1, -1).reshape(batch, height * width, channels)


def tokens_to_latent(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Tokens ``[B, H*W, C]`` -> canonical latent ``[B, C, H, W]`` (``H``/``W`` in cells)."""
    batch, _, channels = tokens.shape
    return tokens.reshape(batch, height, width, channels).movedim(-1, 1)


class MageTimestepProjEmbeddings(nn.Module):
    """Timestep embedding whose frequency table is rounded to the timestep dtype.

    The shared ``timestep_embedding`` keeps ``exp(-log(10000) * i / half)`` in fp32.
    This model rounds it to bf16 first, and the rounding is audible: amplified by the
    x1000 time factor it is worth close to a radian at the top of the table. So it is
    reproduced literally — round the table, then multiply — rather than approximated.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        half = 128
        # ``timestep`` is the (shifted) sigma in [0, 1], forced to bf16 before embedding.
        t = timestep.to(torch.bfloat16)
        exponent = -math.log(10000) * torch.arange(
            half, dtype=torch.float32, device=t.device
        ) / half
        freqs = torch.exp(exponent).to(t.dtype)
        angles = t[:, None].float() * freqs[None, :]
        angles = 1000.0 * angles
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)  # ``flip_sin_to_cos``
        return self.timestep_embedder(emb.to(dtype=hidden_states.dtype))


class MageFlowTransformer2DModel(nn.Module):
    """The Mage-Flow DiT: 12 dual-stream blocks over patch-1 latent tokens."""

    def __init__(self, params: Optional[MageFlowParams] = None):
        super().__init__()
        self.params = params or MageFlowParams()
        params = self.params
        self.out_channels = params.out_channels
        self.inner_dim = params.inner_dim
        self.num_heads = params.num_heads
        self.head_dim = params.head_dim
        self.patch_size = params.patch_size

        self.pe_embedder = RopeCache(matrix_rope(list(params.axes_dims), 10000))
        self.time_text_embed = MageTimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_norm = RMSNorm(params.context_dim, eps=_EPS)
        self.img_in = QuantizedLinear(params.in_channels, self.inner_dim)
        self.txt_in = QuantizedLinear(params.context_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    heads=params.num_heads,
                    attention_head_dim=params.head_dim,
                    eps=_EPS,
                )
                for _ in range(params.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=_EPS
        )
        # patch_size 1: one output value per latent cell, so no patch**2 factor.
        self.proj_out = QuantizedLinear(self.inner_dim, self.out_channels, bias=True)
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        img_pe: torch.Tensor = None,
        txt_pe: torch.Tensor = None,
        ref_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One DiT forward, returning the velocity of the *target* tokens only.

        ``hidden_states`` are the target latent tokens and ``ref_tokens`` the edit's
        references, packed the same way and carrying their own positions inside
        ``img_pe``. Both stay in the sequence for every step.

        Returns ``[B, num_target_tokens, out_channels]`` — still tokens; see
        :func:`tokens_to_latent` for the canonical latent.
        """
        num_img_tokens = hidden_states.shape[1]
        if ref_tokens is not None:
            hidden_states = torch.cat([hidden_states, ref_tokens], dim=1)

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        temb = self.time_text_embed(timestep, hidden_states)

        for block in self.transformer_blocks:
            mark_token_axis(hidden_states, encoder_hidden_states, img_pe, txt_pe)
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                img_pe=img_pe,
                txt_pe=txt_pe,
            )

        # The references carry no prediction: drop them before the output head.
        hidden_states = self.norm_out(hidden_states[:, :num_img_tokens], temb)
        return self.proj_out(hidden_states)


def load_mage_flow_dit(
    dit_path: str,
    params: MageFlowParams,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> MageFlowTransformer2DModel:
    """Build the DiT on meta from the checkpoint's own geometry, then load its weights.

    The released exports need no key remapping: the names are the module names, and
    ``load_dit`` reconstructs the quantized layout from the ``.weight_scale`` siblings.
    """
    with init_empty_weights():
        model = MageFlowTransformer2DModel(params)
    load_dit(model, dit_path, device=device, dtype=dtype)
    logger.info("Loaded Mage-Flow DiT from %s", dit_path)
    return model


__all__ = [
    "MageFlowParams",
    "MageFlowTransformer2DModel",
    "MageTimestepProjEmbeddings",
    "latent_to_tokens",
    "load_mage_flow_dit",
    "tokens_to_latent",
]
