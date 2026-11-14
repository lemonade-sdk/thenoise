"""Wan 2.2 transformer (TI2V-5B, and each expert of T2V-A14B) on TheNoise attention."""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from comfy_kitchen.tensor import QuantizedTensor
from torch import Tensor, nn
from torch.nn import functional as F

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import attention
from thenoise.utils.positions import grid_positions
from thenoise.utils.rope import RopeCache, apply_rope, matrix_rope
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key
from thenoise.utils.timestep import timestep_embedding

#: Wan's released transformers are 128-wide-headed with a 1e-6 norm epsilon. Neither is
#: a shape in the file, so the family supplies them.
HEAD_DIM = 128
NORM_EPS = 1e-6

#: Latent channels per variant, as they appear in the patch embedding.
CHANNELS = {"ti2v_5b": 48, "t2v_14b": 16}


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


def _block_count(shapes: dict, prefix: str = "blocks.") -> int:
    count = 0
    while any(key.startswith(f"{prefix}{count}.") for key in shapes):
        count += 1
    if not count:
        raise ValueError(f"no '{prefix}0.*' keys in the checkpoint")
    return count


def detect_params(dit_path: str) -> Wan22Params:
    """Read the transformer's geometry out of the checkpoint header.

    5B and the two 14B experts share one key layout, so the variant is a shape: the
    patch embedding says the latent channels (48 for TI2V-5B, 16 for the T2V experts),
    its width says ``dim``, and the block count says how deep it goes. Reading it here
    rather than from a per-model config is what lets one adapter take any of them.
    """
    with MemoryEfficientSafeOpen(dit_path) as f:
        shapes = {unwrap_key(key): tuple(f.header[key]["shape"]) for key in f.keys()}

    patch = shapes.get("patch_embedding.weight")
    if patch is None or len(patch) != 5:
        raise ValueError(f"{dit_path}: expected a Wan transformer with a 5D "
                         "'patch_embedding.weight', found none")
    dim, in_channels, pt, ph, pw = patch
    if in_channels not in CHANNELS.values():
        raise ValueError(
            f"{dit_path}: patch embedding takes {in_channels} latent channels, which is "
            f"no Wan variant (expected one of {sorted(CHANNELS.values())})"
        )
    required = ("blocks.0.ffn.0.weight", "head.head.weight",
                "time_embedding.0.weight", "text_embedding.0.weight")
    if absent := [key for key in required if key not in shapes]:
        raise ValueError(f"{dit_path}: has a Wan patch embedding but is missing {absent}")
    ffn_in = shapes["blocks.0.ffn.0.weight"]
    if ffn_in[1] != dim:
        raise ValueError(f"{dit_path}: FFN takes {ffn_in[1]} but the transformer is {dim} wide")
    head = shapes["head.head.weight"]
    if head[0] % (pt * ph * pw):
        raise ValueError(f"{dit_path}: head output {head[0]} is not a whole number of "
                         f"patch cells ({pt}x{ph}x{pw})")
    if dim % HEAD_DIM:
        raise ValueError(f"{dit_path}: {dim} wide transformer cannot split into "
                         f"{HEAD_DIM}-wide heads")
    return Wan22Params(
        in_channels=in_channels,
        out_channels=head[0] // (pt * ph * pw),
        dim=dim,
        num_layers=_block_count(shapes),
        ffn_dim=ffn_in[0],
        freq_dim=shapes["time_embedding.0.weight"][1],
        text_dim=shapes["text_embedding.0.weight"][1],
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
    """Wan QK normalization and attention via TheNoise's shared SDPA."""

    def __init__(self, dim: int, heads: int, eps: float):
        super().__init__()
        self.heads = heads
        self.to_q = QuantizedLinear(dim, dim)
        self.to_k = QuantizedLinear(dim, dim)
        self.to_v = QuantizedLinear(dim, dim)
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)
        self.to_out = QuantizedLinear(dim, dim)

    def forward(self, x: Tensor, context: Tensor | None, rope: Tensor | None) -> Tensor:
        source = x if context is None else context
        q = self.norm_q(self.to_q(x)).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k = self.norm_k(self.to_k(source)).unflatten(-1, (self.heads, -1)).transpose(1, 2)
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

    @torch.compile(fullgraph=False)
    def forward(self, x: Tensor, context: Tensor, temb: Tensor, rope: Tensor) -> Tensor:
        # I2V supplies [B, N, 6, D]; T2V supplies [B, 6, D].
        if temb.ndim == 4:
            mod = (self.scale_shift_table.unsqueeze(0) + temb.float()).unbind(dim=2)
        else:
            mod = (self.scale_shift_table + temb.float()).chunk(6, dim=1)
        shift, scale, gate, shift_ff, scale_ff, gate_ff = mod

        y = (self.norm1(x.float()) * (1 + scale) + shift).to(x.dtype)
        x = (x.float() + self.attn1(y, None, rope) * gate).to(x.dtype)
        x = x + self.attn2(self.norm2(x.float()).to(x.dtype), context, None)
        y = (self.norm3(x.float()) * (1 + scale_ff) + shift_ff).to(x.dtype)
        y = self.ffn_out(F.gelu(self.ffn_in(y), approximate="tanh"))
        return (x.float() + y.float() * gate_ff).to(x.dtype)


class WanNativeCondition(nn.Module):
    """Timestep and text conditioning; the timestep path stays in FP32 (see below)."""

    def __init__(self, params: Wan22Params):
        super().__init__()
        self.freq_dim = params.freq_dim
        self.time_linear_1 = QuantizedLinear(params.freq_dim, params.dim)
        self.time_linear_2 = QuantizedLinear(params.dim, params.dim)
        self.time_proj = QuantizedLinear(params.dim, params.dim * 6)
        self.text_linear_1 = QuantizedLinear(params.text_dim, params.dim)
        self.text_linear_2 = QuantizedLinear(params.dim, params.dim)

    def forward(self, timestep: Tensor, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # The adapter hands over a timestep already in Wan's [0, 1000] range.
        ts = timestep_embedding(timestep, self.freq_dim, time_factor=1.0)
        ts = ts.to(dtype=self.time_linear_1.weight.dtype)
        temb = self.time_linear_2(F.silu(self.time_linear_1(ts))).to(dtype=context.dtype)
        mod = self.time_proj(F.silu(temb))
        mod = mod.unflatten(-1, (6, -1))
        ctx = self.text_linear_2(F.gelu(self.text_linear_1(context), approximate="tanh"))
        return temb, mod, ctx


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

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
    ) -> Tensor:
        b, c, t, h, w = hidden_states.shape
        params = self.params
        pt, ph, pw = params.patch_size
        if c != params.in_channels or t % pt or h % ph or w % pw:
            raise ValueError("invalid Wan latent geometry")

        rope = self.rope(hidden_states)
        x = self.patch_embedding(hidden_states.float()).to(hidden_states.dtype)
        x = x.flatten(2).transpose(1, 2).contiguous()
        temb, modulation, ctx = self.condition_embedder(timestep, encoder_hidden_states)
        if timestep.ndim == 2 and timestep.shape != (b, x.shape[1]):
            raise ValueError("I2V token timesteps must match patchified latent tokens")
        for block in self.blocks:
            x = block(x, ctx, modulation, rope)
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).unbind(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        x = (self.norm_out(x.float()) * (1 + scale) + shift).to(x.dtype)
        x = self.proj_out(x)
        x = x.reshape(b, t // pt, h // ph, w // pw, pt, ph, pw, -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return x
