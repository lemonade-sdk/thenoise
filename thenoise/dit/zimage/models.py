# Z-Image (S3-DiT) model architecture.
# Ported from the diffusers ``ZImageTransformer2DModel`` for thenoise inference,
# stripped to the text-to-image (basic, non-omni) path: no Omni/SigLIP, no
# ControlNet, no gradient checkpointing. Weight key names are unchanged so the
# official / ComfyUI checkpoints load as-is.
#
# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. Licensed under
# the Apache-2.0 License.

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import AttentionParams, attention
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


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

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # The sinusoidal embedding is computed in fp32 for precision; cast it to the
        # projection weight dtype (bf16) before the MLP to avoid a dtype mismatch.
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class Attention(nn.Module):
    """Multi-head attention with fused QKV, QK-RMSNorm and RoPE (complex multiply).

    Matches the ComfyUI / Lumina Z-Image layout: a single fused ``qkv`` projection,
    per-head RMSNorm on query/key, and an ``out`` projection.
    """

    def __init__(self, dim, n_heads, qk_norm, eps):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = QuantizedLinear(dim, 3 * dim, bias=False)
        self.out = QuantizedLinear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()

    def _apply_rotary_emb(self, x, freqs_cis):
        # Real-arithmetic RoPE (no complex ops) so torch.compile can generate
        # code; complex view/multiply falls back to eager in inductor.
        x_dtype = x.dtype
        x = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, L, H, D/2, 2]
        x1, x2 = x[..., 0], x[..., 1]                # real and imaginary parts
        cos = freqs_cis[..., 0].unsqueeze(2)         # [L, 1, D/2]
        sin = freqs_cis[..., 1].unsqueeze(2)
        # (x1 + i*x2) * (cos + i*sin) -> real and imaginary parts.
        x1_out = x1 * cos - x2 * sin
        x2_out = x1 * sin + x2 * cos
        x_out = torch.stack([x1_out, x2_out], dim=-1).flatten(3)
        # Cast back to the activation dtype (bf16); the int8 ``out`` projection has
        # no bf16 ``weight`` to reference for the dtype.
        return x_out.to(x_dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        dim = hidden_states.shape[-1]
        q, k, v = self.qkv(hidden_states).split([dim, dim, dim], dim=-1)

        # q/k/v are [B, L, H, D]; the shared attention() util handles the SDPA
        # [B, H, L, D] transpose and (future) attention-backend swapping.
        query = q.unflatten(-1, (self.n_heads, -1))
        key = k.unflatten(-1, (self.n_heads, -1))
        value = v.unflatten(-1, (self.n_heads, -1))

        query = self.q_norm(query)
        key = self.k_norm(key)

        if freqs_cis is not None:
            query = self._apply_rotary_emb(query, freqs_cis)
            key = self._apply_rotary_emb(key, freqs_cis)

        params = None
        if attention_mask is not None and attention_mask.ndim == 2:
            # SDPA expects [B, H, L, S]; expand the [B, S] key-padding mask to
            # [B, 1, 1, S] (bool: True = attend).
            params = AttentionParams(attention_mask=attention_mask[:, None, None, :])

        hidden_states = attention([query, key, value], attn_params=params, drop_rate=0.0)
        return self.out(hidden_states)


class FeedForward(nn.Module):
    """SwiGLU feedforward: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = QuantizedLinear(dim, hidden_dim, bias=False)
        self.w2 = QuantizedLinear(hidden_dim, dim, bias=False)
        self.w3 = QuantizedLinear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(nn.Module):
    def __init__(self, layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim=dim, n_heads=n_heads, qk_norm=qk_norm, eps=norm_eps)
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


class RopeEmbedder:
    def __init__(self, theta=256.0, axes_dims=(16, 56, 56), axes_lens=(64, 128, 128)):
        self.theta = theta
        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim, end, theta):
        with torch.device("cpu"):
            # Precompute cos/sin pairs as real tensors (no complex ops) so the
            # compiled attention path avoids inductor's complex fallback.
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device="cpu", dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, self.theta)
        if self.freqs_cis[0].device != device:
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        # Each entry is [L, dim/2, 2]; concatenate along the dim/2 axis.
        return torch.cat(result, dim=1)


class ZImageTransformer2DModel(nn.Module):
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
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=(32, 48, 48),
        axes_lens=(1024, 512, 512),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale

        # ComfyUI / Lumina layout: plain (single patch config) embedder + final layer.
        self.x_embedder = QuantizedLinear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
        self.final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(1000 + lid, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True)
                for lid in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(lid, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
                for lid in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), QuantizedLinear(cap_feat_dim, dim, bias=True))

        self.x_pad_token = nn.Parameter(torch.zeros(1, dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, dim))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(lid, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for lid in range(n_layers)
            ]
        )
        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    # ------------------------------------------------------------ patchify
    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify_image(self, image, patch_size, f_patch_size):
        pH, pW, pF = patch_size, patch_size, f_patch_size
        C, F, H, W = image.size()
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return image, (F, H, W), (F_tokens, H_tokens, W_tokens)

    def _pad_with_ids(self, feat, pos_grid_size, pos_start, device):
        ori_len = len(feat)
        pad_len = (-ori_len) % SEQ_MULTI_OF
        total_len = ori_len + pad_len

        ori_pos_ids = self.create_coordinate_grid(size=pos_grid_size, start=pos_start, device=device).flatten(0, 2)
        if pad_len > 0:
            pad_pos_ids = (
                self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device).flatten(0, 2).repeat(pad_len, 1)
            )
            pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat(
                [
                    torch.zeros(ori_len, dtype=torch.bool, device=device),
                    torch.ones(pad_len, dtype=torch.bool, device=device),
                ]
            )
        else:
            pos_ids = ori_pos_ids
            padded_feat = feat
            pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)

        return padded_feat, pos_ids, pad_mask, total_len

    def patchify_and_embed(self, all_image, all_cap_feats, patch_size, f_patch_size):
        device = all_image[0].device
        all_img_out, all_img_size, all_img_pos_ids, all_img_pad_mask = [], [], [], []
        all_cap_out, all_cap_pos_ids, all_cap_pad_mask = [], [], []

        for image, cap_feat in zip(all_image, all_cap_feats):
            cap_out, cap_pos_ids, cap_pad_mask, cap_len = self._pad_with_ids(
                cap_feat, (len(cap_feat) + (-len(cap_feat)) % SEQ_MULTI_OF, 1, 1), (1, 0, 0), device
            )
            all_cap_out.append(cap_out)
            all_cap_pos_ids.append(cap_pos_ids)
            all_cap_pad_mask.append(cap_pad_mask)

            img_patches, size, (F_t, H_t, W_t) = self._patchify_image(image, patch_size, f_patch_size)
            img_out, img_pos_ids, img_pad_mask, _ = self._pad_with_ids(
                img_patches, (F_t, H_t, W_t), (cap_len + 1, 0, 0), device
            )
            all_img_out.append(img_out)
            all_img_size.append(size)
            all_img_pos_ids.append(img_pos_ids)
            all_img_pad_mask.append(img_pad_mask)

        return (
            all_img_out,
            all_cap_out,
            all_img_size,
            all_img_pos_ids,
            all_cap_pos_ids,
            all_img_pad_mask,
            all_cap_pad_mask,
        )

    def _prepare_sequence(self, feats, pos_ids, inner_pad_mask, pad_token, device):
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        feats_cat = torch.cat(feats, dim=0)
        mask = torch.cat(inner_pad_mask).unsqueeze(-1)
        feats_cat = torch.where(mask, pad_token, feats_cat)
        feats = list(feats_cat.split(item_seqlens, dim=0))

        freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))

        feats = nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = nn.utils.rnn.pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]

        if all(seq == max_seqlen for seq in item_seqlens):
            attn_mask = None
        else:
            attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(item_seqlens):
                attn_mask[i, :seq_len] = 1

        return feats, freqs_cis, attn_mask, item_seqlens

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
    def forward(self, x, t, cap_feats, patch_size=None, f_patch_size=None):
        """Denoise one step.

        Args:
            x: list of per-sample image latents ``[C, F, H, W]``.
            t: timestep tensor, shape ``(B,)``, in ``[0, 1]`` (``1 - sigma``). Scaled by
                ``self.t_scale`` (1000) for the sinusoidal embedding.
            cap_feats: list of per-sample caption embeddings ``[seq, cap_feat_dim]``.

        Returns:
            list of per-sample velocity tensors ``[C, F, H, W]`` (the flow direction).
        """
        patch_size = patch_size or self.patch_size
        f_patch_size = f_patch_size or self.f_patch_size
        device = x[0].device

        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])

        (x, cap_feats, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask) = self.patchify_and_embed(
            x, cap_feats, patch_size, f_patch_size
        )

        # X embed & refine
        x_seqlens = [len(xi) for xi in x]
        x = self.x_embedder(torch.cat(x, dim=0))
        x, x_freqs, x_mask, _ = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, device
        )
        for layer in self.noise_refiner:
            x = layer(x, x_mask, x_freqs, adaln_input)

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))
        cap_feats, cap_freqs, cap_mask, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, device
        )
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs)

        # Unified sequence: [x, cap]
        bsz = len(x_seqlens)
        unified = []
        unified_freqs = []
        for i in range(bsz):
            x_len, cap_len = x_seqlens[i], cap_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))
        unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]
        max_seqlen = max(unified_seqlens)

        unified = nn.utils.rnn.pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = nn.utils.rnn.pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)
        if all(seq == max_seqlen for seq in unified_seqlens):
            unified_mask = None
        else:
            unified_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(unified_seqlens):
                unified_mask[i, :seq_len] = 1

        # Main transformer layers
        for layer in self.layers:
            unified = layer(unified, unified_mask, unified_freqs, adaln_input)

        unified = self.final_layer(unified, c=adaln_input)

        return self.unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size)
