# Lumina S3-DiT core — the shared single-stream transformer (ported from diffusers,
# text-to-image path only).
#
# Copyright Alibaba Z-Image / HF, Apache-2.0.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Sequence

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import AttentionParams, attention
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rope import RopeCache, apply_rope, matrix_rope
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.positions import grid_positions
from thenoise.utils.sequence import (
    alignment_padding_mask,
    make_key_padding_mask,
    pad_len_to_multiple,
    pad_to_batch,
    pad_to_length,
)
from thenoise.utils.setup_logging import setup_logging
from thenoise.utils.timestep import timestep_embedding

setup_logging()
import logging

logger = logging.getLogger(__name__)


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

#: How the alignment padding is filled, and whether attention reads it.
#:   ``learned``     — pads are learned tokens, attended.
#:   ``zero_masked`` — pads are zeros, excluded from attention.
PAD_MODES = ("learned", "zero_masked")


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            QuantizedLinear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            QuantizedLinear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        # The sinusoidal embedding is computed in fp32; cast it to the MLP's dtype.
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class Attention(nn.Module):
    """Multi-head attention: fused QKV, per-head QK-RMSNorm, RoPE."""

    def __init__(self, dim, n_heads, eps):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = QuantizedLinear(dim, 3 * dim, bias=False)
        self.out = QuantizedLinear(dim, dim, bias=False)
        self.qk_norm = QKNorm(self.head_dim, eps=eps)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        dim = hidden_states.shape[-1]
        q, k, v = self.qkv(hidden_states).split([dim, dim, dim], dim=-1)

        query = q.unflatten(-1, (self.n_heads, -1)).transpose(1, 2)
        key = k.unflatten(-1, (self.n_heads, -1)).transpose(1, 2)
        value = v.unflatten(-1, (self.n_heads, -1)).transpose(1, 2)

        query, key = self.qk_norm(query, key)

        if freqs_cis is not None:
            query, key = apply_rope(query, key, freqs_cis)

        params = None
        if attention_mask is not None and attention_mask.ndim == 2:
            # Expand the [B, S] key-padding mask to SDPA's [B, 1, 1, S] (True = attend).
            params = AttentionParams(attention_mask=attention_mask[:, None, None, :])

        hidden_states = attention([query, key, value], attn_params=params, drop_rate=0.0)
        return self.out(hidden_states)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = QuantizedLinear(dim, hidden_dim, bias=False)
        self.w2 = QuantizedLinear(hidden_dim, dim, bias=False)
        self.w3 = QuantizedLinear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LuminaTransformerBlock(nn.Module):
    def __init__(self, layer_id, dim, n_heads, n_kv_heads, norm_eps, modulation=True):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim=dim, n_heads=n_heads, eps=norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(QuantizedLinear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    @torch.compile(fullgraph=True)
    def forward(self, x, attn_mask, freqs_cis, adaln_input=None):
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = QuantizedLinear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            QuantizedLinear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        scale = scale.unsqueeze(1)
        x = self.norm_final(x) * scale
        return self.linear(x)


@dataclass
class TokenStream:
    """One forward pass's worth of one token stream's layout.

    ``feats`` holds the real tokens only; padding is applied later, once the
    features are at DiT width, which is the only point where the pad fill is well
    defined. ``pos_ids`` already covers the padded length: the real coordinate grid
    then ``(0, 0, 0)`` per alignment pad, so RoPE depends on the geometry alone.

    ``extra`` (caption stream only) is a second conditioning tensor, already at DiT
    width, concatenated after ``cap_embedder``; it counts towards ``valid``.
    """

    feats: list[torch.Tensor]
    valid: list[int]
    padded: list[int]
    pos_ids: list[torch.Tensor]
    extra: Optional[list[torch.Tensor]] = None
    sizes: list = field(default_factory=list)

    @property
    def segments(self) -> list[tuple[int, int]]:
        """Per-item ``(valid, padded)`` pairs."""
        return list(zip(self.valid, self.padded))


class LuminaTransformer2DModel(nn.Module):
    def __init__(
        self,
        patch_size=2,
        f_patch_size=1,
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        cap_feat_dim=2560,
        rope_theta=256.0,
        axes_dims=(32, 48, 48),
        pad_mode="learned",
    ):
        super().__init__()
        if pad_mode not in PAD_MODES:
            raise ValueError(f"unknown pad_mode {pad_mode!r}; expected one of {PAD_MODES}")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.pad_mode = pad_mode

        self.x_embedder = QuantizedLinear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
        self.final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)

        self.noise_refiner = nn.ModuleList(
            [
                LuminaTransformerBlock(1000 + lid, dim, n_heads, n_kv_heads, norm_eps, modulation=True)
                for lid in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                LuminaTransformerBlock(lid, dim, n_heads, n_kv_heads, norm_eps, modulation=False)
                for lid in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), QuantizedLinear(cap_feat_dim, dim, bias=True))

        self.x_pad_token = nn.Parameter(torch.zeros(1, dim)) if pad_mode == "learned" else None
        self.cap_pad_token = nn.Parameter(torch.zeros(1, dim)) if pad_mode == "learned" else None

        self.layers = nn.ModuleList(
            [
                LuminaTransformerBlock(lid, dim, n_heads, n_kv_heads, norm_eps)
                for lid in range(n_layers)
            ]
        )
        self.axes_dims = list(axes_dims)
        self.rope_embedder = RopeCache(matrix_rope(axes_dims, rope_theta))

    # ------------------------------------------------------------ patchify
    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        return grid_positions(size, start=start, dtype=torch.int32, device=device)

    def _patchify_image(self, image, patch_size, f_patch_size):
        """``[C, F, H, W]`` latent -> ``[F_t*H_t*W_t, C*pF*pH*pW]`` patch tokens.

        The token order matches the coordinate grid, so token ``i`` and position
        ``i`` describe the same cell.
        """
        pH, pW, pF = patch_size, patch_size, f_patch_size
        C, F, H, W = image.size()
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return image, (F, H, W), (F_tokens, H_tokens, W_tokens)

    def _pad_positions(self, pos_grid_size, pos_start, pad_len, device):
        """Token positions for a padded stream: the real grid, then zeros for pads."""
        ori_pos_ids = self.create_coordinate_grid(size=pos_grid_size, start=pos_start, device=device)
        if pad_len <= 0:
            return ori_pos_ids
        pad_pos_ids = self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device).repeat(pad_len, 1)
        return torch.cat([ori_pos_ids, pad_pos_ids], dim=0)

    def prepare_rope(
        self,
        x,
        cap_feats,
        cap_extra=None,
        patch_size=None,
        f_patch_size=None,
        key="",
        clear=True,
    ):
        """Compute and cache the RoPE frequencies for one conditioning branch.

        The positions depend only on the latent/caption shapes, which are fixed for
        a prompt, so the frequencies are built once and reused by ``forward``.
        """
        patch_size = patch_size or self.patch_size
        f_patch_size = f_patch_size or self.f_patch_size
        x_stream, cap_stream = self.patchify_and_embed(
            x, cap_feats, patch_size, f_patch_size, cap_extra
        )
        if clear:
            self.rope_embedder.clear()
        self.rope_embedder.store(f"img{key}", torch.cat(x_stream.pos_ids, dim=0).unsqueeze(0))
        self.rope_embedder.store(f"cap{key}", torch.cat(cap_stream.pos_ids, dim=0).unsqueeze(0))

    def patchify_and_embed(
        self,
        all_image,
        all_cap_feats,
        patch_size=None,
        f_patch_size=None,
        all_cap_extra=None,
    ) -> tuple[TokenStream, TokenStream]:
        """Patchify the latents and lay out the ``(image, caption)`` ``TokenStream``s.

        The image stream starts at ``t = cap_padded + 1`` on the temporal axis, which
        makes the caption's padded length part of the geometry.
        """
        device = all_image[0].device
        if all_cap_extra is not None and len(all_cap_extra) != len(all_cap_feats):
            raise ValueError(f"cap_extra has {len(all_cap_extra)} items, caption has {len(all_cap_feats)}")

        x_feats, x_pos_ids, x_valid, x_padded, x_sizes = [], [], [], [], []
        cap_feats, cap_pos_ids, cap_valid, cap_padded = [], [], [], []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            # The caption block is [caption, extra], padded as a whole.
            cap_len = len(cap_feat)
            if all_cap_extra is not None:
                cap_len += len(all_cap_extra[i])
            cap_pad = pad_len_to_multiple(cap_len, SEQ_MULTI_OF) - cap_len
            cap_feats.append(cap_feat)
            cap_pos_ids.append(self._pad_positions((cap_len, 1, 1), (1, 0, 0), cap_pad, device))
            cap_valid.append(cap_len)
            cap_padded.append(cap_len + cap_pad)

            img_patches, size, (F_t, H_t, W_t) = self._patchify_image(image, patch_size, f_patch_size)
            img_len = len(img_patches)
            img_pad = pad_len_to_multiple(img_len, SEQ_MULTI_OF) - img_len
            x_feats.append(img_patches)
            x_pos_ids.append(self._pad_positions((F_t, H_t, W_t), (cap_padded[-1] + 1, 0, 0), img_pad, device))
            x_valid.append(img_len)
            x_padded.append(img_len + img_pad)
            x_sizes.append(size)

        x_stream = TokenStream(x_feats, x_valid, x_padded, x_pos_ids, sizes=x_sizes)
        cap_stream = TokenStream(cap_feats, cap_valid, cap_padded, cap_pos_ids, extra=all_cap_extra)
        return x_stream, cap_stream

    def _prepare_stream(self, feats, stream: TokenStream, pe, device):
        """Batch a padded stream, split its frequencies and mask it.

        ``feats`` are the per-item DiT-width features padded out to
        ``stream.padded``, so the frequencies split in lockstep.
        """
        # Drop the batch dim of ``pe`` and split the frequencies back per sample.
        positions = list(pe.squeeze(0).split(stream.padded, dim=0))
        feats, positions, _ = pad_to_batch(feats, positions)
        mask = self._attention_mask([[seg] for seg in stream.segments], device)
        return feats, positions, mask

    def _attention_mask(self, item_segments: Sequence[Sequence[tuple[int, int]]], device):
        """Attention mask for one stream, from its per-item ``(valid, padded)`` segments.

        A ``learned`` pad slot holds a real embedding, so only the batch padding
        beyond an item's padded length is masked. A ``zero_masked`` pad slot also has
        to be punched out of the middle of the sequence.
        """
        if self.pad_mode == "learned":
            return make_key_padding_mask([sum(p for _, p in segs) for segs in item_segments], device)
        return alignment_padding_mask([list(segs) for segs in item_segments], device)

    def unpatchify(self, x, size, patch_size, f_patch_size):
        pH = pW = patch_size
        pF = f_patch_size
        result = []
        for i in range(len(x)):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
            result.append(x[i])
        return result

    # ------------------------------------------------------------ forward
    def forward(self, x, t, cap_feats, cap_extra=None, patch_size=None, f_patch_size=None, rope_key=""):
        """Denoise one step.

        ``t`` has shape ``(B,)`` and is in ``[0, 1]`` (``1 - sigma``). ``cap_extra``
        is an optional per-sample second conditioning tensor ``[seq, dim]``,
        concatenated after ``cap_embedder``. ``rope_key`` selects the ``prepare_rope``
        entry pair (``img``/``cap`` + suffix).

        Returns per-sample velocity tensors ``[C, F, H, W]``.
        """
        patch_size = patch_size or self.patch_size
        f_patch_size = f_patch_size or self.f_patch_size
        device = x[0].device

        adaln_input = self.t_embedder(t)

        x_stream, cap_stream = self.patchify_and_embed(
            x, cap_feats, patch_size, f_patch_size, cap_extra
        )

        # X embed & refine
        x = self.x_embedder(torch.cat(x_stream.feats, dim=0))
        x = pad_to_length(
            list(x.split(x_stream.valid, dim=0)), x_stream.padded, pad_token=self.x_pad_token
        )
        x, x_freqs, x_mask = self._prepare_stream(
            x, x_stream, self.rope_embedder[f"img{rope_key}"], device
        )
        for layer in self.noise_refiner:
            x = layer(x, x_mask, x_freqs, adaln_input)

        # Cap embed & refine; extra conditioning is already dim-wide and is spliced
        # in before the block is padded.
        cap = self.cap_embedder(torch.cat(cap_stream.feats, dim=0))
        cap = list(cap.split([len(c) for c in cap_stream.feats], dim=0))
        if cap_stream.extra is not None:
            cap = [torch.cat([c, e], dim=0) for c, e in zip(cap, cap_stream.extra)]
        cap = pad_to_length(cap, cap_stream.padded, pad_token=self.cap_pad_token)
        cap, cap_freqs, cap_mask = self._prepare_stream(
            cap, cap_stream, self.rope_embedder[f"cap{rope_key}"], device
        )
        for layer in self.context_refiner:
            cap = layer(cap, cap_mask, cap_freqs)

        # Unified sequence: [x, cap]. Both are already padded to their own multiple,
        # so their segments carry over into the unified mask.
        unified, unified_freqs, unified_segments = [], [], []
        for i in range(len(x_stream.padded)):
            x_len, cap_len = x_stream.padded[i], cap_stream.padded[i]
            unified.append(torch.cat([x[i][:x_len], cap[i][:cap_len]]))
            unified_freqs.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))
            unified_segments.append([x_stream.segments[i], cap_stream.segments[i]])

        unified, unified_freqs, _ = pad_to_batch(unified, unified_freqs)
        unified_mask = self._attention_mask(unified_segments, device)

        # Main transformer layers
        for layer in self.layers:
            unified = layer(unified, unified_mask, unified_freqs, adaln_input)

        unified = self.final_layer(unified, c=adaln_input)

        return self.unpatchify(list(unified.unbind(dim=0)), x_stream.sizes, patch_size, f_patch_size)
