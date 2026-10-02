"""BailingMoeV2 — the Ling-2.0-mini MoE language core, ported from Ant Group's
``modeling_bailing_moe_v2.py`` onto this repo's primitives."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import AttentionParams, attention
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.rope import apply_rope_split_half

#: How the three position axes share the rotary frequencies. ``sum(sections)``
#: must be ``rotary_dim / 2``.
MROPE_SECTION = (8, 12, 12)

#: One ``(start, grid_h, grid_w)`` block marks a run of "image" tokens; ``grid_*``
#: are in MERGED units (``grid_h * grid_w`` tokens), ``start`` is its first token.
Block = tuple[int, int, int]


@dataclass
class BailingMoeV2Config:
    """Ling-2.0-mini as the released ``mllm/config.json`` describes it."""

    vocab_size: int = 157184
    hidden_size: int = 2048
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    partial_rotary_factor: float = 0.5
    rope_theta: float = 600000.0
    rms_norm_eps: float = 1e-6
    num_experts: int = 256
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    #: The first ``first_k_dense_replace`` layers are plain MLPs, the rest are MoE.
    first_k_dense_replace: int = 1

    @property
    def rotary_dim(self) -> int:
        """Rotated width; the head-dim tail passes through RoPE untouched."""
        return int(self.head_dim * self.partial_rotary_factor)


class BailingAttention(nn.Module):
    """GQA attention with per-head QK-RMSNorm and partial ``video_rope``: the
    split-half helper rotates the first ``rotary_dim`` channels, leaving the rest."""

    def __init__(self, config: BailingMoeV2Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        # ``query_key_value``/``dense`` are the checkpoint's own names.
        self.query_key_value = QuantizedLinear(
            config.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False
        )
        self.dense = QuantizedLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        # Plain RMSNorms, matching the checkpoint's ``q_norm``/``k_norm``.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], freqs) -> torch.Tensor:
        heads = self.num_heads + 2 * self.num_kv_heads
        qkv = self.query_key_value(x).unflatten(-1, (heads, self.head_dim)).transpose(1, 2)
        query, key, value = qkv.split((self.num_heads, self.num_kv_heads, self.num_kv_heads), dim=1)
        # Norm before rotating, as the weights were produced.
        query, key = self.q_norm(query), self.k_norm(key)
        cos, sin = freqs
        query = apply_rope_split_half(query, cos, sin)
        key = apply_rope_split_half(key, cos, sin)
        out = attention([query, key, value], attn_params=AttentionParams(attention_mask))
        return self.dense(out)


class MLP(nn.Module):
    """SwiGLU feed-forward (``down(silu(gate(x)) * up(x))``), bias-free."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = QuantizedLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = QuantizedLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = QuantizedLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertBank(nn.Module):
    """Every expert of one projection in a single ``[experts, out, in]`` weight; a
    quantized export keeps it low-bit and runs one row-sliced view per expert."""

    def __init__(self, num_experts: int, out_features: int, in_features: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.out_features = out_features
        self.in_features = in_features
        # Bound like ``nn.Linear`` (kaiming-derived); the checkpoint overwrites
        # every row.
        bound = in_features**-0.5
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        nn.init.uniform_(self.weight, -bound, bound)
        self._quantized = False

    def load_quantized(self, qt: torch.Tensor) -> None:
        """Switch this bank to a low-bit weight of any layout (and stay low-bit)."""
        del self.weight  # free the full-precision bank
        self.register_buffer("weight", qt)
        self._quantized = True

    def expert_weight(self, expert: int) -> torch.Tensor:
        """One expert's ``[out, in]`` weight, still low-bit when the bank is. The view
        is a slice of the buffers, never an index into the 3-D ``QuantizedTensor``."""
        weight = self.weight
        if not self._quantized:
            return weight[expert]
        params = weight.params
        scale = params.scale
        return weight._copy_with(
            qdata=weight._qdata[expert],
            params=dataclasses.replace(
                params,
                scale=scale[expert] if scale.dim() else scale,
                orig_shape=(self.out_features, self.in_features),
            ),
            clone_params=False,
        )


class Gate(nn.Module):
    """Sigmoid router with expert bias and group-limited top-k (fp32 math). The bias
    shifts only the *selection* score, never the returned sigmoid weight."""

    def __init__(self, config: BailingMoeV2Config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.proj = QuantizedLinear(config.hidden_size, config.num_experts, bias=False)
        self.expert_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(-1, x.shape[-1])
        # Routing math is fp32; the projection runs in the weight's own dtype.
        scores = torch.sigmoid(self.proj(x).float())
        selection = scores + self.expert_bias.to(scores.dtype)

        # Pick the ``topk_group`` strongest groups (sum of their two best experts),
        # then choose ``top_k`` experts among those groups only.
        grouped = selection.view(-1, self.n_group, self.num_experts // self.n_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        groups = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, groups, True)
        masked = grouped.masked_fill(~group_mask.unsqueeze(-1), float("-inf")).view(x.shape[0], -1)
        topk_idx = torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

        topk_weight = torch.gather(scores, dim=1, index=topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor


def expert_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    gate_up: ExpertBank,
    down: ExpertBank,
) -> torch.Tensor:
    """Route ``x`` through the fused banks: sort by expert so each runs one GEMM on
    its own tokens, then scatter the sum back over each token's experts."""

    num_tokens, top_k = topk_idx.shape
    flat_idx = topk_idx.reshape(-1)
    order = torch.argsort(flat_idx)
    counts = torch.bincount(flat_idx, minlength=gate_up.num_experts).tolist()

    sorted_x = x[order // top_k]
    weight = topk_weight.reshape(-1)[order].unsqueeze(-1)
    sorted_out = torch.empty_like(sorted_x)
    gate_width = gate_up.out_features // 2

    start = 0
    for expert, n in enumerate(counts):
        if n == 0:
            continue
        chunk = sorted_x[start : start + n]
        gated = F.linear(chunk, gate_up.expert_weight(expert))
        sorted_out[start : start + n] = (
            F.linear(F.silu(gated[..., :gate_width]) * gated[..., gate_width:], down.expert_weight(expert))
            * weight[start : start + n]
        )
        start += n

    routed = torch.empty_like(sorted_out)
    routed[order] = sorted_out
    return routed.view(num_tokens, top_k, -1).sum(dim=1)


class Experts(nn.Module):
    """The two fused expert banks of one MoE layer (``gate_up_proj``/``down_proj``)."""

    def __init__(self, config: BailingMoeV2Config) -> None:
        super().__init__()
        self.gate_up_proj = ExpertBank(
            config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size
        )
        self.down_proj = ExpertBank(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )


class SparseMoeBlock(nn.Module):
    """Top-k experts + one shared expert, routed by a text and an image router;
    ``image_mask`` picks which answer to use."""

    def __init__(self, config: BailingMoeV2Config) -> None:
        super().__init__()
        self.gate = Gate(config)
        self.image_gate = Gate(config)
        self.experts = Experts(config)
        self.shared_experts = MLP(config.hidden_size, config.moe_intermediate_size)

    def forward(self, x: torch.Tensor, image_mask: Optional[torch.Tensor]) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        text_idx, text_weight = self.gate(flat)
        if image_mask is None:
            topk_idx, topk_weight = text_idx, text_weight
        else:
            image_idx, image_weight = self.image_gate(flat)
            pick = image_mask.reshape(-1, 1)
            topk_idx = torch.where(pick, image_idx, text_idx)
            topk_weight = torch.where(pick, image_weight, text_weight)
        routed = expert_dispatch(
            flat, topk_idx, topk_weight, self.experts.gate_up_proj, self.experts.down_proj
        )
        return routed.view_as(x) + self.shared_experts(x)


class DecoderLayer(nn.Module):
    """Pre-norm transformer block: dense MLP for the first layer(s), MoE after."""

    def __init__(self, config: BailingMoeV2Config, layer_id: int) -> None:
        super().__init__()
        self.attention = BailingAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp: nn.Module
        if layer_id < config.first_k_dense_replace:
            self.mlp = MLP(config.hidden_size, config.intermediate_size)
        else:
            self.mlp = SparseMoeBlock(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs,
        image_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attention(self.input_layernorm(x), attention_mask, freqs)
        hidden = self.post_attention_layernorm(x)
        if isinstance(self.mlp, SparseMoeBlock):
            return x + self.mlp(hidden, image_mask)
        return x + self.mlp(hidden)


def video_rope(
    seq_len: int,
    blocks: Sequence[Block],
    rotary_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``video_rope`` cos/sin for a text sequence carrying image blocks: the text
    counter steps over each block; image tokens share temporal/height positions
    along width."""

    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    for start, grid_h, grid_w in blocks:
        pos[start + grid_h * grid_w :] -= grid_h * grid_w - 1

    height, width = pos.clone(), pos.clone()
    shift = 0
    for start, grid_h, grid_w in blocks:
        end = start + grid_h * grid_w
        base = start - shift
        pos[start:end] = base
        height[start:end] = base + (
            torch.arange(grid_h, dtype=torch.float32, device=device) - (grid_h - 1) // 2
        ).repeat_interleave(grid_w)
        width[start:end] = base + (
            torch.arange(grid_w, dtype=torch.float32, device=device) - (grid_w - 1) // 2
        ).repeat(grid_h)
        shift += grid_h * grid_w - 1

    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    spatial = sum(MROPE_SECTION[1:])
    angles = pos[:, None] * inv_freq
    angles[:, 0:spatial:2] = height[:, None] * inv_freq[0:spatial:2]
    angles[:, 1:spatial:2] = width[:, None] * inv_freq[1:spatial:2]

    emb = torch.cat((angles, angles), dim=-1)[:, None, None, :]
    return emb.cos(), emb.sin()


class BailingMoeV2(nn.Module):
    """The BailingMoeV2 stack: embeddings, ``num_hidden_layers`` blocks, final norm."""

    def __init__(self, config: Optional[BailingMoeV2Config] = None) -> None:
        super().__init__()
        self.config = config or BailingMoeV2Config()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(self.config, layer_id) for layer_id in range(self.config.num_hidden_layers)
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        blocks: Sequence[Block] = (),
        attention_mask: Optional[torch.Tensor] = None,
        capture_pre_layers: Sequence[int] = (),
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the stack over ``inputs_embeds`` ``[B, L, hidden]``, capturing each
        ``capture_pre_layers`` input state and always appending the post-final-norm one."""
        batch, seq_len, _ = inputs_embeds.shape
        # The band is built in fp32 and cast down with the activations; an fp32
        # cos/sin would promote q and k out of bf16 while v stayed put.
        cos, sin = video_rope(
            seq_len, blocks, self.config.rotary_dim, self.config.rope_theta, inputs_embeds.device
        )
        freqs = (cos.to(inputs_embeds.dtype), sin.to(inputs_embeds.dtype))

        image_mask = None
        if blocks:
            image_mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=inputs_embeds.device)
            for start, grid_h, grid_w in blocks:
                image_mask[:, start : start + grid_h * grid_w] = True

        # Causal, plus the padding the caller marks out; the diagonal is included.
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=inputs_embeds.device).tril()
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :].bool()
        else:
            mask = mask[None, None]

        captured: list[torch.Tensor] = []
        x = inputs_embeds
        for layer_id, layer in enumerate(self.layers):
            if layer_id in capture_pre_layers:
                captured.append(x)
            x = layer(x, mask, freqs, image_mask)
        x = self.norm(x)
        captured.append(x)
        return x, captured


__all__ = [
    "BailingAttention",
    "BailingMoeV2",
    "BailingMoeV2Config",
    "Block",
    "DecoderLayer",
    "ExpertBank",
    "Experts",
    "Gate",
    "MLP",
    "MROPE_SECTION",
    "SparseMoeBlock",
    "expert_dispatch",
    "video_rope",
]
