# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Lightricks and Hugging Face contributors.
"""Native LTX-2.3/2.5 audiovisual DiT, without Diffusers runtime imports.

Numerics, parameter names and forward contract follow the Apache-2.0 LTX-2
reference (Lightricks/LTX-2) and the Apache-2.0 LTX transformer implementation
in huggingface/diffusers (transformer_ltx2.py). This is a *model module*, not
another model loader, sampler, offloader, or inference pipeline. TheNoise owns
all those facilities, including attention/SDPA, timestep and RoPE operations.

The checkpoint-facing class name is retained for the model factory. Only
native PyTorch modules and existing TheNoise utilities are used at runtime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from thenoise.dit.quantized import QuantizedLinear

from thenoise.utils.attention import AttentionParams, attention
from thenoise.utils.qk_norm import QKNorm
from thenoise.dit.ltx_video.rope import AVRope
from thenoise.utils.rope import apply_rope_split_half
from thenoise.utils.timestep import timestep_embedding

#: How each stream's trunk is divided. A checkpoint states the trunk; it never states
#: the head split, and the released family divides the same way in both streams.
VIDEO_HEAD_DIM = 128
AUDIO_HEAD_DIM = 64


class _TimeFeatures(nn.Module):
    """Only the PixArt time embedding branch that LTX actually uses.

    The nested ``timestep_embedder.linear_{1,2}`` names are checkpoint ABI.
    ``timestep_embedding`` is the existing TheNoise sinusoidal implementation.
    Call with time_factor=1 because LTX adapters pass scaled timesteps already.
    """

    def __init__(self, width: int):
        super().__init__()
        self.timestep_embedder = nn.Module()
        self.timestep_embedder.linear_1 = QuantizedLinear(256, width)
        self.timestep_embedder.linear_2 = QuantizedLinear(width, width)
        self.timestep_embedder.act = nn.SiLU()

    def forward(self, time: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        x = timestep_embedding(time, 256, time_factor=1).to(dtype=dtype)
        t = self.timestep_embedder
        return t.linear_2(t.act(t.linear_1(x)))


class _AdaLayerNormSingle(nn.Module):
    def __init__(self, width: int, mod_params: int):
        super().__init__()
        self.emb = _TimeFeatures(width)
        self.silu = nn.SiLU()
        self.linear = QuantizedLinear(width, mod_params * width)

    def forward(self, time: torch.Tensor, *, batch_size: int, hidden_dtype: torch.dtype):
        embedded = self.emb(time, hidden_dtype)
        return self.linear(self.silu(embedded)), embedded


class _GELUProj(nn.Module):
    def __init__(self, width: int, inner: int, bias: bool):
        super().__init__()
        self.proj = QuantizedLinear(width, inner, bias=bias)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(self.proj(x))


class _FeedForward(nn.Module):
    def __init__(self, width: int, bias: bool):
        super().__init__()
        self.net = nn.ModuleList([
            _GELUProj(width, width * 4, bias),
            nn.Dropout(0.0),
            QuantizedLinear(width * 4, width, bias=bias),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.net:
            x = block(x)
        return x


class _LTXAttention(nn.Module):
    """LTX attention projection ABI, using TheNoise's SDPA and split-half RoPE."""

    def __init__(self, query_dim: int, cross_dim: int, heads: int, head_dim: int,
                 gated: bool):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.qk_norm = QKNorm(inner, eps=1e-6)
        self.to_q = QuantizedLinear(query_dim, inner, bias=True)
        self.to_k = QuantizedLinear(cross_dim, inner, bias=True)
        self.to_v = QuantizedLinear(cross_dim, inner, bias=True)
        self.to_out = nn.ModuleList([QuantizedLinear(inner, query_dim, bias=True), nn.Dropout(0.0)])
        self.to_gate_logits = QuantizedLinear(query_dim, heads, bias=True) if gated else None

    def forward(self, x: torch.Tensor, *, context: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
                q_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
                k_rope: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        if context is None:
            context = x
        logits = self.to_gate_logits(x) if self.to_gate_logits is not None else None
        q, k = self.qk_norm(self.to_q(x), self.to_k(context))
        v = self.to_v(context)
        b, q_tokens, _ = q.shape
        k_tokens = k.shape[1]
        q = q.reshape(b, q_tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, k_tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, k_tokens, self.heads, self.head_dim).transpose(1, 2)
        if q_rope is not None:
            q = apply_rope_split_half(q, *q_rope)
            k = apply_rope_split_half(k, *(k_rope if k_rope is not None else q_rope))
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
        x = attention(q, k, v, attn_params=AttentionParams(attention_mask=mask))
        if logits is not None:
            # Per-head sigmoid gates. Zero logits mean a unit gate, as in LTX.
            x = x.unflatten(-1, (self.heads, self.head_dim))
            x = x * (2.0 * logits.sigmoid()).unsqueeze(-1)
            x = x.flatten(-2)
        return self.to_out[1](self.to_out[0](x))


class _AVBlock(nn.Module):
    """One video/audio block: self, prompt, reciprocal cross and FFN attention."""

    def __init__(self, vdim: int, adim: int, vheads: int, aheads: int,
                 vhead_dim: int, ahead_dim: int, vctx: int, actx: int,
                 vgated: bool, agated: bool, vprompt_mod: bool,
                 aprompt_mod: bool, ff_bias: bool, audio_ff_bias: bool):
        super().__init__()
        self.video_cross_attn_adaln = vprompt_mod
        self.audio_cross_attn_adaln = aprompt_mod
        self.cross_attn_adaln = vprompt_mod or aprompt_mod
        self.norm1 = nn.RMSNorm(vdim, eps=1e-6, elementwise_affine=False)
        self.attn1 = _LTXAttention(vdim, vdim, vheads, vhead_dim, vgated)
        self.audio_norm1 = nn.RMSNorm(adim, eps=1e-6, elementwise_affine=False)
        self.audio_attn1 = _LTXAttention(adim, adim, aheads, ahead_dim, agated)
        self.norm2 = nn.RMSNorm(vdim, eps=1e-6, elementwise_affine=False)
        self.attn2 = _LTXAttention(vdim, vctx, vheads, vhead_dim, vgated)
        self.audio_norm2 = nn.RMSNorm(adim, eps=1e-6, elementwise_affine=False)
        self.audio_attn2 = _LTXAttention(adim, actx, aheads, ahead_dim, agated)
        self.audio_to_video_norm = nn.RMSNorm(vdim, eps=1e-6, elementwise_affine=False)
        # Cross attention uses the AUDIO heads/head_dim on BOTH sides.
        self.audio_to_video_attn = _LTXAttention(vdim, adim, aheads, ahead_dim, vgated)
        self.video_to_audio_norm = nn.RMSNorm(adim, eps=1e-6, elementwise_affine=False)
        self.video_to_audio_attn = _LTXAttention(adim, vdim, aheads, ahead_dim, agated)
        self.norm3 = nn.RMSNorm(vdim, eps=1e-6, elementwise_affine=False)
        self.ff = _FeedForward(vdim, ff_bias)
        self.audio_norm3 = nn.RMSNorm(adim, eps=1e-6, elementwise_affine=False)
        self.audio_ff = _FeedForward(adim, audio_ff_bias)
        self.scale_shift_table = nn.Parameter(torch.randn(9 if vprompt_mod else 6, vdim) / math.sqrt(vdim))
        self.audio_scale_shift_table = nn.Parameter(torch.randn(9 if aprompt_mod else 6, adim) / math.sqrt(adim))
        if self.cross_attn_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.randn(2, vdim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.randn(2, adim))
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, vdim))
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, adim))

    @staticmethod
    def _ada(table: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (table[None, None].to(emb) + emb.reshape(emb.shape[0], emb.shape[1], table.shape[0], -1)).unbind(2)

    def forward(self, v: torch.Tensor, a: torch.Tensor,
                vtext: torch.Tensor, atext: torch.Tensor,
                vtime: torch.Tensor, atime: torch.Tensor,
                v_cross: torch.Tensor, a_cross: torch.Tensor,
                v_gate: torch.Tensor, a_gate: torch.Tensor,
                v_prompt: torch.Tensor | None, a_prompt: torch.Tensor | None,
                v_rope: tuple[torch.Tensor, torch.Tensor],
                a_rope: tuple[torch.Tensor, torch.Tensor],
                v_ca_rope: tuple[torch.Tensor, torch.Tensor],
                a_ca_rope: tuple[torch.Tensor, torch.Tensor],
                vtext_mask: torch.Tensor | None,
                atext_mask: torch.Tensor | None,
                v_self_mask: torch.Tensor | None,
                *, isolate_modalities: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        s_v, c_v, g_v, s_vf, c_vf, g_vf, *text_v = self._ada(self.scale_shift_table, vtime)
        vn = self.norm1(v) * (1 + c_v) + s_v
        v = v + g_v * self.attn1(vn, mask=v_self_mask, q_rope=v_rope)

        s_a, c_a, g_a, s_af, c_af, g_af, *text_a = self._ada(self.audio_scale_shift_table, atime)
        an = self.audio_norm1(a) * (1 + c_a) + s_a
        a = a + g_a * self.audio_attn1(an, q_rope=a_rope)

        if self.cross_attn_adaln:
            if v_prompt is None:
                shift_vk, scale_vk = self.prompt_scale_shift_table[None, None].to(v).unbind(2)
            else:
                shift_vk, scale_vk = self._ada(self.prompt_scale_shift_table, v_prompt)
            if a_prompt is None:
                shift_ak, scale_ak = self.audio_prompt_scale_shift_table[None, None].to(a).unbind(2)
            else:
                shift_ak, scale_ak = self._ada(self.audio_prompt_scale_shift_table, a_prompt)

        vn = self.norm2(v)
        if self.video_cross_attn_adaln:
            shift_vq, scale_vq, gate_vq = text_v
            vn = vn * (1 + scale_vq) + shift_vq
        if self.cross_attn_adaln:
            vtext = vtext * (1 + scale_vk) + shift_vk
        vc = self.attn2(vn, context=vtext, mask=vtext_mask)
        if self.video_cross_attn_adaln:
            vc = vc * gate_vq
        v = v + vc

        an = self.audio_norm2(a)
        if self.audio_cross_attn_adaln:
            shift_aq, scale_aq, gate_aq = text_a
            an = an * (1 + scale_aq) + shift_aq
        if self.cross_attn_adaln:
            atext = atext * (1 + scale_ak) + shift_ak
        ac = self.audio_attn2(an, context=atext, mask=atext_mask)
        if self.audio_cross_attn_adaln:
            ac = ac * gate_aq
        a = a + ac

        if not isolate_modalities:
            vn = self.audio_to_video_norm(v)
            an = self.video_to_audio_norm(a)
            vsc, vsh, vcs, vch = self._ada(self.video_a2v_cross_attn_scale_shift_table[:4], v_cross)
            vg = self._ada(self.video_a2v_cross_attn_scale_shift_table[4:], v_gate)[0]
            asc, ash, acs, ach = self._ada(self.audio_a2v_cross_attn_scale_shift_table[:4], a_cross)
            ag = self._ada(self.audio_a2v_cross_attn_scale_shift_table[4:], a_gate)[0]
            a_to_v = self.audio_to_video_attn(
                vn * (1 + vsc) + vsh,
                context=an * (1 + asc) + ash,
                q_rope=v_ca_rope, k_rope=a_ca_rope,
            )
            # These use the *pre-cross-attention* normalized stream states.
            v_to_a = self.video_to_audio_attn(
                an * (1 + acs) + ach,
                context=vn * (1 + vcs) + vch,
                q_rope=a_ca_rope, k_rope=v_ca_rope,
            )
            v = v + vg * a_to_v
            a = a + ag * v_to_a

        vn = self.norm3(v) * (1 + c_vf) + s_vf
        v = v + g_vf * self.ff(vn)
        an = self.audio_norm3(a) * (1 + c_af) + s_af
        a = a + g_af * self.audio_ff(an)
        return v, a

@dataclass(frozen=True)
class LTXParams:
    """A transformer's geometry, as a checkpoint header states it.

    Past the widths, four halves of the module tree are optional and the file answers
    each: whether a stream's feed-forward carries biases (2.3 does, 2.5 does not),
    whether its attention is gated, whether prompt modulation is wired in, and whether
    the keyframe embedding exists (2.5 only). Reading them as tensors rather than
    assuming them is what lets a re-export of different width, or one shipped without
    its zero biases, load instead of failing a shape check.
    """

    in_channels: int
    out_channels: int
    dim: int
    num_layers: int
    cross_attention_dim: int
    audio_in_channels: int
    audio_out_channels: int
    audio_dim: int
    audio_cross_attention_dim: int
    ff_bias: bool
    audio_ff_bias: bool
    gated_attn: bool
    audio_gated_attn: bool
    cross_attn_mod: bool
    audio_cross_attn_mod: bool
    keyframes: bool

    @property
    def heads(self) -> int:
        return self.dim // VIDEO_HEAD_DIM

    @property
    def audio_heads(self) -> int:
        return self.audio_dim // AUDIO_HEAD_DIM


class LTX2VideoTransformer3DModel(nn.Module):
    """Native LTX distilled audiovisual DiT with original checkpoint weight names.

    The public `forward` arguments are the existing TheNoise LTX adapter's
    interface, *not* a Diffusers pipeline. VideoModel/MemoryManager remain
    entirely responsible for segment scheduling, residency and decoding.

    Only the distilled recipe is implemented: unit patches, so a latent voxel is one
    token; split-half RoPE; gelu-approximate feed-forward; RMS Q/K norms across heads
    and no affine block norms; biased attention projections; and prompt modulation from
    AdaLN tables rather than caption tokens as positions. The widths, the depth and the
    optional halves come from the checkpoint, not from here.
    """

    def __init__(
        self, *, in_channels: int, out_channels: int,
        num_attention_heads: int, attention_head_dim: int, cross_attention_dim: int,
        audio_in_channels: int, audio_out_channels: int,
        audio_num_attention_heads: int, audio_attention_head_dim: int,
        audio_cross_attention_dim: int, num_layers: int,
        ff_bias: bool, audio_ff_bias: bool, gated_attn: bool, audio_gated_attn: bool,
        cross_attn_mod: bool, audio_cross_attn_mod: bool,
        use_keyframes_abs_pos_embedding: bool, use_prompt_adaln_single: bool = True,
        vae_scale_factors: tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20, base_height: int = 2048, base_width: int = 2048,
        audio_scale_factor: int = 4, audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000, audio_hop_length: int = 160,
        rope_theta: float = 10000.0, rope_double_precision: bool = True,
        causal_offset: int = 1, timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            timestep_scale_multiplier=timestep_scale_multiplier,
            cross_attn_timestep_scale_multiplier=cross_attn_timestep_scale_multiplier,
            use_keyframes_abs_pos_embedding=use_keyframes_abs_pos_embedding,
            use_prompt_adaln_single=use_prompt_adaln_single,
        )
        vdim = num_attention_heads * attention_head_dim
        adim = audio_num_attention_heads * audio_attention_head_dim
        self.proj_in = QuantizedLinear(in_channels, vdim)
        self.audio_proj_in = QuantizedLinear(audio_in_channels, adim)
        if use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = nn.Parameter(torch.zeros(1, vdim))
        self.prompt_modulation = cross_attn_mod or audio_cross_attn_mod
        self.time_embed = _AdaLayerNormSingle(vdim, 9 if cross_attn_mod else 6)
        self.audio_time_embed = _AdaLayerNormSingle(adim, 9 if audio_cross_attn_mod else 6)
        self.av_cross_attn_video_scale_shift = _AdaLayerNormSingle(vdim, 4)
        self.av_cross_attn_audio_scale_shift = _AdaLayerNormSingle(adim, 4)
        self.av_cross_attn_video_a2v_gate = _AdaLayerNormSingle(vdim, 1)
        self.av_cross_attn_audio_v2a_gate = _AdaLayerNormSingle(adim, 1)
        self.scale_shift_table = nn.Parameter(torch.randn(2, vdim) / math.sqrt(vdim))
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, adim) / math.sqrt(adim))
        if self.prompt_modulation and use_prompt_adaln_single:
            self.prompt_adaln = _AdaLayerNormSingle(vdim, 2)
            self.audio_prompt_adaln = _AdaLayerNormSingle(adim, 2)
        base_rope = dict(base_num_frames=pos_embed_max_pos, theta=rope_theta,
                         double_precision=rope_double_precision,
                         causal_offset=causal_offset)
        self.rope = AVRope(vdim, num_attention_heads, modality="video",
                            base_height=base_height, base_width=base_width,
                            scale_factors=vae_scale_factors, **base_rope)
        self.audio_rope = AVRope(adim, audio_num_attention_heads, modality="audio",
                                  base_num_frames=audio_pos_embed_max_pos,
                                  scale_factors=(audio_scale_factor,),
                                  sampling_rate=audio_sampling_rate,
                                  hop_length=audio_hop_length,
                                  theta=rope_theta, double_precision=rope_double_precision,
                                  causal_offset=causal_offset)
        base_cross = max(pos_embed_max_pos, audio_pos_embed_max_pos)
        # Each one rotates a stream's own coordinates, so each carries that stream's
        # geometry and not just the width it turns.
        self.cross_attn_rope = AVRope(audio_cross_attention_dim, num_attention_heads,
                                      modality="video", base_num_frames=base_cross,
                                      base_height=base_height, base_width=base_width,
                                      scale_factors=vae_scale_factors,
                                      theta=rope_theta, double_precision=rope_double_precision,
                                      causal_offset=causal_offset)
        self.cross_attn_audio_rope = AVRope(audio_cross_attention_dim, audio_num_attention_heads,
                                            modality="audio", base_num_frames=base_cross,
                                            scale_factors=(audio_scale_factor,),
                                            sampling_rate=audio_sampling_rate,
                                            hop_length=audio_hop_length,
                                            theta=rope_theta, double_precision=rope_double_precision,
                                            causal_offset=causal_offset)
        self.transformer_blocks = nn.ModuleList([
            _AVBlock(vdim, adim, num_attention_heads, audio_num_attention_heads,
                     attention_head_dim, audio_attention_head_dim,
                     cross_attention_dim, audio_cross_attention_dim,
                     gated_attn, audio_gated_attn, cross_attn_mod, audio_cross_attn_mod,
                     ff_bias, audio_ff_bias)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(vdim, eps=1e-6, elementwise_affine=False)
        self.proj_out = QuantizedLinear(vdim, out_channels)
        self.audio_norm_out = nn.LayerNorm(adim, eps=1e-6, elementwise_affine=False)
        self.audio_proj_out = QuantizedLinear(adim, audio_out_channels)

    @property
    def dtype(self):
        return self.proj_in.weight.dtype

    @staticmethod
    def _embed(module: _AdaLayerNormSingle, time: torch.Tensor,
               batch: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        mod, features = module(time.flatten(), batch_size=batch, hidden_dtype=dtype)
        return mod.reshape(batch, -1, mod.shape[-1]), features.reshape(batch, -1, features.shape[-1])

    def forward(
        self, hidden_states: torch.Tensor, audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor, audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor, audio_timestep: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None, audio_sigma: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        num_frames: int | None = None, height: int | None = None,
        width: int | None = None, fps: float = 24.0,
        audio_num_frames: int | None = None,
        isolate_modalities: bool = False,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        use_cross_timestep: bool = False,
        attention_kwargs: dict | None = None,
        video_self_attention_mask: torch.Tensor | None = None,
        video_keyframes_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatio_temporal_guidance_blocks or perturbation_mask is not None or attention_kwargs:
            raise NotImplementedError("LTX distilled adapter does not support STG/perturbation/attention overrides")
        b = hidden_states.shape[0]
        audio_timestep = timestep if audio_timestep is None else audio_timestep
        audio_sigma = sigma if audio_sigma is None else audio_sigma
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = ((1 - encoder_attention_mask.to(hidden_states.dtype))
                                      * -10000.0).unsqueeze(1)
        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = ((1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype))
                                            * -10000.0).unsqueeze(1)
        if video_self_attention_mask is not None:
            video_self_attention_mask = (1 - video_self_attention_mask.to(hidden_states.dtype)) * -10000.0
        v_rope = self.rope.video(b, num_frames, height, width, fps, hidden_states.device)
        a_rope = self.audio_rope.audio(b, audio_num_frames, audio_hidden_states.device)
        # Cross-modality attention has only time to align on, so each stream hands the
        # other its time coordinate alone.
        v_ca_rope = self.cross_attn_rope.video_time(b, num_frames, height, width, fps,
                                                     hidden_states.device)
        a_ca_rope = self.cross_attn_audio_rope.audio(b, audio_num_frames,
                                                     audio_hidden_states.device)
        v = self.proj_in(hidden_states)
        a = self.audio_proj_in(audio_hidden_states)
        if self.config.use_keyframes_abs_pos_embedding and video_keyframes_mask is not None:
            v = v + (video_keyframes_mask > 0).to(v) * self.keyframes_abs_pos_embedding.to(v)
        vtime, embedded_v = self._embed(self.time_embed, timestep, b, v.dtype)
        atime, embedded_a = self._embed(self.audio_time_embed, audio_timestep, b, a.dtype)
        if self.prompt_modulation and self.config.use_prompt_adaln_single:
            if sigma is None or audio_sigma is None:
                raise ValueError("LTX prompt modulation needs sigma and audio_sigma")
            vprompt, _ = self._embed(self.prompt_adaln, sigma, b, v.dtype)
            aprompt, _ = self._embed(self.audio_prompt_adaln, audio_sigma, b, a.dtype)
        else:
            vprompt = aprompt = None
        ratio = (self.config.cross_attn_timestep_scale_multiplier /
                 self.config.timestep_scale_multiplier)
        v_cross_time = audio_sigma if use_cross_timestep else timestep
        a_cross_time = sigma if use_cross_timestep else audio_timestep
        if v_cross_time is None or a_cross_time is None:
            raise ValueError("LTX cross-modality modulation needs both time streams")
        v_cross, _ = self._embed(self.av_cross_attn_video_scale_shift, v_cross_time, b, v.dtype)
        a_cross, _ = self._embed(self.av_cross_attn_audio_scale_shift, a_cross_time, b, a.dtype)
        v_gate, _ = self._embed(self.av_cross_attn_video_a2v_gate, v_cross_time * ratio, b, v.dtype)
        a_gate, _ = self._embed(self.av_cross_attn_audio_v2a_gate, a_cross_time * ratio, b, a.dtype)
        for block in self.transformer_blocks:
            v, a = block(v, a, encoder_hidden_states, audio_encoder_hidden_states,
                         vtime, atime, v_cross, a_cross, v_gate, a_gate,
                         vprompt, aprompt, v_rope, a_rope, v_ca_rope, a_ca_rope,
                         encoder_attention_mask, audio_encoder_attention_mask,
                         video_self_attention_mask, isolate_modalities=isolate_modalities)
        vv = self.scale_shift_table[None, None].to(embedded_v) + embedded_v[:, :, None]
        ash = self.audio_scale_shift_table[None, None].to(embedded_a) + embedded_a[:, :, None]
        v = self.norm_out(v) * (1 + vv[:, :, 1]) + vv[:, :, 0]
        a = self.audio_norm_out(a) * (1 + ash[:, :, 1]) + ash[:, :, 0]
        return self.proj_out(v), self.audio_proj_out(a)


__all__ = ["LTX2VideoTransformer3DModel"]
