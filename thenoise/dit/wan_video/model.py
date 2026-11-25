"""Wan 2.2 transformer (TI2V-5B, and each expert of T2V-A14B) on TheNoise attention."""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Union

import torch
from comfy_kitchen.tensor import QuantizedTensor
from torch import Tensor, nn
from torch.nn import functional as F

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import attention
from thenoise.utils.dynamo import mark_token_axis
from thenoise.utils.positions import grid_positions
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rope import RopeCache, apply_rope, matrix_rope
from thenoise.utils.sol_attention import SolAttention, sol_attention
from thenoise.utils.safetensors import (
    Header,
    Signature,
    as_header,
    require_divisible,
)
from thenoise.utils.timestep import timestep_embedding

#: Wan's released transformers are 128-wide-headed with a 1e-6 norm epsilon. Neither is
#: a shape in the file, so the family supplies them.
HEAD_DIM = 128
NORM_EPS = 1e-6

#: Latent channels per variant, as they appear in the patch embedding.
CHANNELS = {"ti2v_5b": 48, "t2v_14b": 16, "i2v_14b": 36}

#: Every released Wan 2.2 transformer, whatever its variant. The patch embedding's shape
#: is what says which one, so the names only have to rule out everything else.
SIGNATURE = Signature(
    "Wan 2.2 transformer",
    keys=(
        "patch_embedding.weight",
        "blocks.0.self_attn.q.weight",
        "blocks.0.ffn.0.weight",
        "head.head.weight",
        "time_embedding.0.weight",
        "text_embedding.0.weight",
    ),
)


@dataclass(frozen=True)
class Wan22Params:
    """The transformer geometry, as read out of the checkpoint header."""

    in_channels: int
    out_channels: int
    dim: int
    num_layers: int
    ffn_dim: int
    freq_dim: int
    text_dim: int
    patch_size: tuple[int, int, int]
    head_dim: int = HEAD_DIM
    eps: float = NORM_EPS

    @property
    def heads(self) -> int:
        return self.dim // self.head_dim

    @property
    def tokens_per_latent(self) -> int:
        """Patch cells one latent voxel is unpatchified into."""
        return math.prod(self.patch_size)


def detect_params(checkpoint: Union[str, Header]) -> Wan22Params:
    """Read the transformer's geometry out of the checkpoint header.

    5B and the two 14B experts share one key layout, so the variant is a shape: the
    patch embedding says the latent channels (48 for TI2V-5B, 16 for the T2V experts),
    its width says ``dim``, and the block count says how deep it goes. Reading it here
    rather than from a per-model config is what lets one adapter take any of them.
    """
    header = as_header(checkpoint)
    header.require(SIGNATURE)

    patch = header.shape("patch_embedding.weight")
    if len(patch) != 5:
        raise ValueError(
            f"{header.path}: a Wan transformer's 'patch_embedding.weight' is 5D, found "
            f"{len(patch)}D")
    dim, in_channels, pt, ph, pw = patch
    if in_channels not in CHANNELS.values():
        raise ValueError(
            f"{header.path}: patch embedding takes {in_channels} latent channels, which is "
            f"no Wan variant (expected one of {sorted(CHANNELS.values())})"
        )
    ffn_in = header.shape("blocks.0.ffn.0.weight")
    if ffn_in[1] != dim:
        raise ValueError(
            f"{header.path}: FFN takes {ffn_in[1]} but the transformer is {dim} wide")
    head = header.shape("head.head.weight")
    if head[0] % (pt * ph * pw):
        raise ValueError(f"{header.path}: head output {head[0]} is not a whole number of "
                         f"patch cells ({pt}x{ph}x{pw})")
    require_divisible(dim, HEAD_DIM, "transformer width", "head dim")
    return Wan22Params(
        in_channels=in_channels,
        out_channels=head[0] // (pt * ph * pw),
        dim=dim,
        num_layers=header.block_count("blocks."),
        ffn_dim=ffn_in[0],
        freq_dim=header.shape("time_embedding.0.weight")[1],
        text_dim=header.shape("text_embedding.0.weight")[1],
        patch_size=(pt, ph, pw),
    )


class WanNativeRoPE(nn.Module):
    """3D RoPE frequencies for the patch grid, built once per latent shape."""

    def __init__(self, head_dim: int, patch_size):
        super().__init__()
        self.patch_size = tuple(patch_size)
        hw = 2 * (head_dim // 6)
        self._frequencies = RopeCache(matrix_rope([head_dim - 2 * hw, hw, hw], 10000.0))
        self._shape_device = None

    def _apply(self, fn):
        self._frequencies.clear()
        self._shape_device = None
        return super()._apply(fn)

    def forward(self, latents: Tensor) -> Tensor:
        _, _, t, h, w = latents.shape
        pt, ph, pw = self.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError("Wan native DiT needs latent geometry divisible by patch size")
        t, h, w = t // pt, h // ph, w // pw
        key = (t, h, w, latents.device)
        if key != self._shape_device:
            grid = grid_positions([t, h, w], device=latents.device)
            self._frequencies.store("grid", grid.reshape(1, -1, 3))
            self._shape_device = key
        return self._frequencies["grid"]


class WanNativeAttention(nn.Module):
    """Wan QK normalization and attention."""

    def __init__(self, dim: int, heads: int, eps: float):
        super().__init__()
        self.heads = heads
        self.to_q = QuantizedLinear(dim, dim)
        self.to_k = QuantizedLinear(dim, dim)
        self.to_v = QuantizedLinear(dim, dim)
        self.qk_norm = QKNorm(dim, eps=eps)
        self.to_out = QuantizedLinear(dim, dim)

    def forward(self, x: Tensor, context: Tensor | None, rope: Tensor | None,
                sol: SolAttention | None = None) -> Tensor:
        source = x if context is None else context
        q, k = self.qk_norm(self.to_q(x), self.to_k(source))
        if sol is not None and context is None:
            # Routing is offered to self-attention only, and asked for twice so a caller
            # that hands a plan to the text branch is refused rather than surprised. The
            # kernel reads tokens in row order, so the head-major transpose is what drops
            # out here; the rope broadcast moves to the head axis instead, measured
            # bit-identical to transposing the operands.
            q, k, v = (t.unflatten(-1, (self.heads, -1))
                       for t in (q, k, self.to_v(source)))
            if rope is not None:
                q, k = apply_rope(q, k, rope, head_axis=2)
            return self.to_out(sol_attention(q, k, v, sol).reshape(q.shape[0], q.shape[1], -1))
        q = q.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k = k.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        v = self.to_v(source).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, k, rope)
        out = attention([q, k, v])  # TheNoise: [B, N, H * D]
        return self.to_out(out)


class WanNativeBlock(nn.Module):
    def __init__(self, params: Wan22Params):
        super().__init__()
        self.norm1 = nn.LayerNorm(params.dim, eps=params.eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(params.dim, eps=params.eps, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(params.dim, eps=params.eps, elementwise_affine=False)
        self.attn1 = WanNativeAttention(params.dim, params.heads, params.eps)
        self.attn2 = WanNativeAttention(params.dim, params.heads, params.eps)
        self.ffn_in = QuantizedLinear(params.dim, params.ffn_dim)
        self.ffn_out = QuantizedLinear(params.ffn_dim, params.dim)
        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, params.dim))

    # fullgraph=False for the routed variant: the kernel is an island a compiled block does
    # not trace, so the step's graph is cut around the attention call. A dense step still
    # compiles whole, because the branch is taken outside the compiled region.
    @torch.compile(fullgraph=False)
    def forward(self, x: Tensor, context: Tensor, temb: Tensor, rope: Tensor,
                sol: SolAttention | None = None) -> Tensor:
        # I2V supplies [B, N, 6, D]; T2V supplies [B, 6, D].
        if temb.ndim == 4:
            mod = (self.scale_shift_table.unsqueeze(0) + temb.float()).unbind(dim=2)
        else:
            mod = (self.scale_shift_table + temb.float()).chunk(6, dim=1)
        shift, scale, gate, shift_ff, scale_ff, gate_ff = mod

        y = (self.norm1(x.float()) * (1 + scale) + shift).to(x.dtype)
        x = (x.float() + self.attn1(y, None, rope, sol) * gate).to(x.dtype)
        x = x + self.attn2(self.norm2(x.float()).to(x.dtype), context, None)
        y = (self.norm3(x.float()) * (1 + scale_ff) + shift_ff).to(x.dtype)
        y = self.ffn_out(F.gelu(self.ffn_in(y), approximate="tanh"))
        return (x.float() + y.float() * gate_ff).to(x.dtype)


class WanNativeCondition(nn.Module):
    """Timestep conditioning; the timestep path stays in FP32 (see below).

    The text branch is not in ``forward``: it depends on nothing but the prompt, so the
    adapter runs it once per request as ``WanNativeDiT.fuse_text``.
    """

    def __init__(self, params: Wan22Params):
        super().__init__()
        self.freq_dim = params.freq_dim
        self.time_linear_1 = QuantizedLinear(params.freq_dim, params.dim)
        self.time_linear_2 = QuantizedLinear(params.dim, params.dim)
        self.time_proj = QuantizedLinear(params.dim, params.dim * 6)
        self.text_linear_1 = QuantizedLinear(params.text_dim, params.dim)
        self.text_linear_2 = QuantizedLinear(params.dim, params.dim)

    def fuse_text(self, context: Tensor) -> Tensor:
        """Prompt features -> the ``[B, L, dim]`` context the blocks attend to."""
        return self.text_linear_2(F.gelu(self.text_linear_1(context),
                                         approximate="tanh"))

    def forward(self, timestep: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        # The adapter hands over a timestep already in Wan's [0, 1000] range.
        ts = timestep_embedding(timestep, self.freq_dim, time_factor=1.0)
        ts = ts.to(dtype=self.time_linear_1.weight.dtype)
        temb = self.time_linear_2(F.silu(self.time_linear_1(ts))).to(dtype=dtype)
        mod = self.time_proj(F.silu(temb))
        return temb, mod.unflatten(-1, (6, -1))


class WanNativeDiT(nn.Module):
    """Wan 2.2 transformer, built to the geometry a checkpoint says it has."""

    def __init__(self, params: Wan22Params):
        super().__init__()
        self.params = params
        self.rope = WanNativeRoPE(params.head_dim, params.patch_size)
        self.patch_embedding = nn.Conv3d(params.in_channels, params.dim,
                                         kernel_size=params.patch_size,
                                         stride=params.patch_size)
        self.condition_embedder = WanNativeCondition(params)
        self.blocks = nn.ModuleList([WanNativeBlock(params) for _ in range(params.num_layers)])
        self.norm_out = nn.LayerNorm(params.dim, eps=params.eps, elementwise_affine=False)
        self.proj_out = QuantizedLinear(params.dim, params.out_channels * params.tokens_per_latent)
        self.scale_shift_table = nn.Parameter(torch.empty(1, 2, params.dim))
        self.eval()

    def apply_precision(self) -> "WanNativeDiT":
        """Run at one trunk dtype, modulation in FP32.

        A file can hold one model in two precisions and a Linear split across both
        cannot run, so the largest weight — a block's FFN matrix — sets the trunk
        ``dtype`` reports. The timestep MLP and the shift/scale tables drive every
        residual branch, where rounding moves the whole trajectory rather than the
        last bits of one activation, so they are FP32 whatever the trunk was read
        at. A quantized weight already decides its own precision.
        """
        trunk = max((weight for block in self.blocks for weight in block.parameters()),
                    key=lambda weight: weight.numel() * weight.element_size())
        if not isinstance(trunk, QuantizedTensor):
            self.to(trunk.dtype)  # a no-op per tensor already at the trunk dtype
        islands = [(self, "patch_embedding"), (self, "scale_shift_table"),
                   (self.condition_embedder, "time_linear_1"),
                   (self.condition_embedder, "time_linear_2")]
        islands += [(block, name) for block in self.blocks
                    for name in ("norm1", "norm2", "norm3", "scale_shift_table")]
        for owner, name in islands:
            value = getattr(owner, name)
            if isinstance(value, nn.Parameter):
                value.data = value.data.float()
            elif not isinstance(getattr(value, "weight", None), QuantizedTensor):
                value.float()
        return self

    @property
    def dtype(self) -> torch.dtype:
        return self.proj_out.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embedding.weight.device

    def cache_context(self, *_args, **_kwargs):
        # No cross-step caching: Wan denoising changes every latent token.
        return nullcontext()

    def fuse_text(self, context: Tensor) -> Tensor:
        """Raw prompt features -> what ``forward`` takes as ``encoder_hidden_states``.

        Once per request, with these weights resident: the trunk never projects the
        prompt itself.
        """
        return self.condition_embedder.fuse_text(context)

    def head(self, x: Tensor, temb: Tensor) -> Tensor:
        # I2V carries a per-token timestep, so its modulation is [B, N, D].
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).unbind(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        x = (self.norm_out(x.float()) * (1 + scale) + shift).to(x.dtype)
        return self.proj_out(x)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        sol: SolAttention | None = None,
    ) -> Tensor:
        b, c, t, h, w = hidden_states.shape
        params = self.params
        pt, ph, pw = params.patch_size
        if c != params.in_channels or t % pt or h % ph or w % pw:
            raise ValueError("invalid Wan latent geometry")

        rope = self.rope(hidden_states)
        x = self.patch_embedding(hidden_states.float()).to(hidden_states.dtype)
        x = x.flatten(2).transpose(1, 2).contiguous()
        temb, modulation = self.condition_embedder(timestep, x.dtype)
        if timestep.ndim == 2 and timestep.shape != (b, x.shape[1]):
            raise ValueError("I2V token timesteps must match patchified latent tokens")
        mark_token_axis(x, encoder_hidden_states, rope)
        if modulation.ndim == 4:
            mark_token_axis(modulation)
        for block in self.blocks:
            x = block(x, encoder_hidden_states, modulation, rope, sol)
        x = self.head(x, temb)
        x = x.reshape(b, t // pt, h // ph, w // pw, pt, ph, pw, -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return x
