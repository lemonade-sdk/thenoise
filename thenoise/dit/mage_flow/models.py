"""Mage-Flow DiT — the 12-layer member of the Qwen-Image dual-stream family.

It reuses :class:`thenoise.dit.qwen_image.models.QwenImageTransformerBlock` at
``eps`` 1e-6, and adds:

  * **patch 1** — one token per latent cell of the raw 128-channel VAE latent
    (see :func:`latent_to_tokens`);
  * **unrotated text** — the text stream sits at position 0, where RoPE is the
    identity, so one matrix is stored for it;
  * **a bf16-rounded timestep table** — see :class:`MageTimestepProjEmbeddings`.

The timestep embedding has a single row, so reference tokens are modulated at ``t``
like the target ones and their K/V cannot be cached.

The **low-rank modulation** export variant (``modulation_rank`` in
:class:`MageFlowParams`) feeds the per-block AdaLN heads a shared, rank-wide FP32
projection of the timestep embedding — see
:class:`thenoise.utils.fp32_linear.Fp32Linear`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights

from thenoise.dit.qwen_image.models import (
    AdaLayerNormContinuous,
    QwenImageTransformerBlock,
    TimestepEmbedding,
)
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.dynamo import mark_token_axis
from thenoise.utils.fp32_linear import Fp32Linear
from thenoise.utils.loader import load_dit
from thenoise.utils.rope import RopeCache, matrix_rope
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

#: Layer-norm / RMS-norm epsilon of the reference implementation.
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
    #: Latent cells per token.
    patch_size: int = 1
    #: Width of the shared modulation vector, or ``None`` for the dense heads.
    modulation_rank: Optional[int] = None

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.head_dim


def latent_to_tokens(latents: torch.Tensor) -> torch.Tensor:
    """Canonical latent ``[B, C, H, W]`` -> tokens ``[B, H*W, C]``, one per latent cell."""
    batch, channels, height, width = latents.shape
    return latents.movedim(1, -1).reshape(batch, height * width, channels)


def tokens_to_latent(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Tokens ``[B, H*W, C]`` -> canonical latent ``[B, C, H, W]`` (``H``/``W`` in cells)."""
    batch, _, channels = tokens.shape
    return tokens.reshape(batch, height, width, channels).movedim(-1, 1)


class MageTimestepProjEmbeddings(nn.Module):
    """Timestep embedding whose frequency table is rounded to the timestep dtype.

    The rounding is audible: amplified by the x1000 time factor it is worth close to
    a radian at the top of the table. So it is reproduced literally — round the
    table, then multiply.
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


class MageLowRankBlock(QwenImageTransformerBlock):
    """Qwen-Image block driven by the shared low-rank modulation vector.

    ``img_mod``/``txt_mod`` keep their ``Sequential`` slotting — the checkpoint calls
    them ``img_mod.1`` — with slot 0 ``Identity`` (the SiLU moves to the shared
    down-projection) and slot 1 an FP32 rank-wide projection.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        attention_head_dim: int,
        eps: float,
        modulation_rank: int,
        out_dtype: torch.dtype,
    ):
        super().__init__(dim=dim, heads=heads, attention_head_dim=attention_head_dim, eps=eps)
        for mod in (self.img_mod, self.txt_mod):
            out_features = mod[1].out_features
            mod[0] = nn.Identity()
            mod[1] = Fp32Linear(modulation_rank, out_features, bias=True, out_dtype=out_dtype)


def _make_block(
    params: MageFlowParams, dim: int, rank: Optional[int], compute_dtype: torch.dtype
) -> QwenImageTransformerBlock:
    """One fresh transformer block: the dense one, or the low-rank export's."""
    if rank is None:
        return QwenImageTransformerBlock(
            dim=dim, heads=params.num_heads, attention_head_dim=params.head_dim, eps=_EPS
        )
    return MageLowRankBlock(
        dim=dim,
        heads=params.num_heads,
        attention_head_dim=params.head_dim,
        eps=_EPS,
        modulation_rank=rank,
        out_dtype=compute_dtype,
    )


class MageFlowTransformer2DModel(nn.Module):
    """The Mage-Flow DiT: 12 dual-stream blocks over patch-1 latent tokens."""

    def __init__(self, params: Optional[MageFlowParams] = None, compute_dtype: torch.dtype = torch.bfloat16):
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

        # The low-rank export feeds every block's heads from one shared FP32
        # down-projection of ``silu(temb)``; the factors are calibrated in FP32,
        # hence ``Fp32Linear``.
        rank = params.modulation_rank
        self.modulation_down = None if rank is None else Fp32Linear(self.inner_dim, rank)
        self.transformer_blocks = nn.ModuleList(
            _make_block(params, self.inner_dim, rank, compute_dtype) for _ in range(params.num_layers)
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

        ``ref_tokens`` are packed like the target tokens and carry their positions
        inside ``img_pe``; both stay in the sequence for every step.
        """
        num_img_tokens = hidden_states.shape[1]
        if ref_tokens is not None:
            hidden_states = torch.cat([hidden_states, ref_tokens], dim=1)

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        temb = self.time_text_embed(timestep, hidden_states)

        # ``norm_out`` is modulated by the full-width ``temb``; only the blocks can be
        # driven by the rank-wide vector, which consumes the SiLU the dense heads
        # apply internally.
        block_temb = temb if self.modulation_down is None else self.modulation_down(F.silu(temb).float())

        for block in self.transformer_blocks:
            mark_token_axis(hidden_states, encoder_hidden_states, img_pe, txt_pe)
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=block_temb,
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

    The released exports need no key remapping; ``load_dit`` reconstructs the
    quantized layout from the ``.weight_scale`` siblings.

    A low-rank export loads with ``dtype=None``: it is a full-precision file and a
    bulk cast would round the FP32 modulation factors.
    """
    with init_empty_weights():
        model = MageFlowTransformer2DModel(params, compute_dtype=dtype)
    load_dit(
        model,
        dit_path,
        device=device,
        dtype=None if params.modulation_rank is not None else dtype,
    )
    logger.info("Loaded Mage-Flow DiT from %s", dit_path)
    return model


__all__ = [
    "MageFlowParams",
    "MageFlowTransformer2DModel",
    "MageLowRankBlock",
    "MageTimestepProjEmbeddings",
    "latent_to_tokens",
    "load_mage_flow_dit",
    "tokens_to_latent",
]
