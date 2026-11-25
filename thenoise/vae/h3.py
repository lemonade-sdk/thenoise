"""H3 codecs — a 3D causal CNN encoder over a ViT pixel decoder.

Ported from ComfyUI's ``comfy/ldm/minimax/vae.py`` with its module tree — and with it
the checkpoint key names — verbatim; only the ops differ. Two checkpoints share this
tree: the video one decodes a clip through ``patch_size_t = 4`` and 3D rope, the image
one drops the time axis the reference carries and decodes a frame at a time, which is
why its ``post_quant_conv`` is 2D while ``quant_conv`` stays 3D.

The latent is 24 channels at 16 pixels per cell, normalized by the constants below — the
image checkpoint ships none, so they are their one home; the video checkpoint ships its
own and :func:`load_h3_video_vae` reads them off the file. Encoding an image returns a
posterior *sample* on :data:`ENCODE_SEED`, never the mean: the decoder was trained on
samples. Encoding a clip returns the mean, as the reference does.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.attention import eager_attention
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.rope import apply_rope_split_half, split_half_rope_axes
from thenoise.utils.loader import load_codec
from thenoise.utils.safetensors import Header, Signature
from thenoise.utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

COMPUTE_DTYPE = torch.float16

#: Pixels are handed to the codec in [-1, 1] and normalized with ImageNet statistics.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

#: Per-channel latent statistics of the released checkpoint.
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
]

#: Seed of the posterior sample encode takes. Restarting the same stream keeps a
#: reference's latent from moving with the generation's own seed.
ENCODE_SEED = 42

#: Encoder geometry (``ch_mult`` stages, with the spatial and temporal stride of each).
CH = 128
CH_MULT = (1, 2, 2, 4, 4, 8)
NUM_RES_BLOCKS = 2
SPACE_DOWN = (2, 2, 2, 2, 1, 1)
TIME_DOWN = (1, 2, 2, 1, 1, 1)
Z_CHANNELS = 24

#: What a clip's frames compress to along the time axis, and therefore what one latent
#: frame stands for on the way back out.
TIME_RATIO = math.prod(TIME_DOWN)

#: Decoder geometry. ``HEAD_DIM`` is what the rope is a ratio of, so the head count
#: follows from the width the checkpoint states. The two checkpoints disagree on how
#: much of the head the rope turns and over how many axes: the image decoder knows
#: ``(h, w)``, the video one ``(t, h, w)``.
HEAD_DIM = 64
ROPE_DIM_RATIO = 0.5
VIDEO_ROPE_DIM_RATIO = 0.75
ROPE_THETA = 100.0
NUM_REGISTER_TOKENS = 4
NORM_EPS = 1e-5

#: Pixels per tile and the cross-fade between neighbours. The decoder's rope spans
#: whatever grid it is handed, so a wider grid is a scale it never saw.
TILE_SIZE = 256
TILE_BLEND = 64

#: Frames the video codec encodes per clip, and the tokens it throws away at the end of
#: each. Both feed :class:`TemporalPlan`, and both are stated by the checkpoint.
CLIP_LENGTH = 17
TOKEN_DROP = 3


# --------------------------------------------------------------------------- encoder


class CausalConv3d(nn.Conv3d):
    """Reflect-spatial, causal-temporal 3D convolution.

    ``padding`` is the (temporal, height, width) radius. At ``T = 1`` everything padded
    in front is zero, so the frame is convolved with the kernel's last temporal tap
    instead of building the padded video; a longer input is padded with the zeros a
    causal conv wants and sees every tap.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x, pre_norm=None, spatial_pad=None, residual=None):
        pad_t, pad_h, pad_w = self.causal_padding
        if spatial_pad is None:
            spatial_pad = (pad_w, pad_w, pad_h, pad_h)
        if pre_norm is not None:
            x = F.silu(pre_norm(x))
        if any(spatial_pad):
            x = F.pad(x, (*spatial_pad, 0, 0), mode="reflect")
        weight = self.weight
        if pad_t:
            if x.shape[2] == 1:
                weight = self.weight[:, :, -1:]
            else:
                x = F.pad(x, (0, 0, 0, 0, 2 * pad_t, 0))
        out = F.conv3d(x, weight, self.bias, self.stride)
        return out if residual is None else out.add_(residual)


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    """GroupNorm whose statistics come from one frame at a time.

    Time rides in the batch dimension while they are taken, so no frame learns its
    scale from its neighbours'. One frame needs no isolating, and the still path takes
    the plain norm straight through.
    """

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] == 1:
            return super().forward(x)
        batch, channels, frames, height, width = x.shape
        y = super().forward(x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, 1,
                                                             height, width))
        return y.view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


def group_norm_3d(num_channels: int) -> nn.GroupNorm:
    """The codec's norm: 32 groups, per-frame statistics."""
    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    """Stride convolution, spatially asymmetric-padded so the odd edge is kept.

    The temporal stride needs no such care: the causal padding in front is what rounds a
    clip's last frame up, whichever stride it is read at.
    """

    def __init__(self, in_channels, out_channels, space_stride=2, time_stride=1):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_channels, out_channels, kernel_size=3, padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
        )

    def forward(self, x):
        if self.space_stride == 2:
            return self.conv(x, spatial_pad=(0, 1, 0, 1))
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(x, pre_norm=self.norm1)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return self.conv2(h, pre_norm=self.norm2, residual=x)


class EncoderFCN3D(nn.Module):
    """The H3 video encoder: one ``ResnetBlock3D`` chain per ``ch_mult`` stage."""

    def __init__(self, ch=CH, ch_mult=CH_MULT, num_res_blocks=NUM_RES_BLOCKS,
                 space_down=SPACE_DOWN, time_down=TIME_DOWN,
                 in_channels=3, z_channels=Z_CHANNELS):
        super().__init__()
        if not len(ch_mult) == len(space_down) == len(time_down):
            raise ValueError(
                f"ch_mult {ch_mult}, space_down {space_down} and time_down {time_down} "
                "must describe the same number of stages"
            )
        block_mid = [ch * mult for mult in ch_mult]
        block_in = [block_mid[0]] + block_mid[:-1]

        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for i_level, mult in enumerate(ch_mult):
            level = nn.Module()
            level.block = nn.ModuleList(
                ResnetBlock3D(block_in[i_level] if i == 0 else block_mid[i_level], block_mid[i_level])
                for i in range(num_res_blocks)
            )
            if space_down[i_level] * time_down[i_level] > 1:
                level.downsample = Downsample3D(block_mid[i_level], mult * ch,
                                                space_stride=space_down[i_level],
                                                time_stride=time_down[i_level])
            self.down.append(level)

        self.norm_out = group_norm_3d(block_mid[-1])
        self.conv_out = CausalConv3d(block_mid[-1], 2 * z_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level, level in enumerate(self.down):
            for block in level.block:
                h = block(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        return self.conv_out(h, pre_norm=self.norm_out)


# --------------------------------------------------------------------------- decoder


def token_ids(dims: Tuple[int, ...], device, dtype) -> torch.Tensor:
    """Cell centres of the latent grid, each axis normalized to ``[-1, 1]``.

    ``(t, h, w)`` for a clip, ``(h, w)`` for a frame or for the image decoder's rope.
    """
    axes = [(torch.arange(0.5, dim, dtype=dtype, device=device) / dim) * 2.0 - 1.0 for dim in dims]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, len(dims) - 1)


class RotaryEmbedding(nn.Module):
    """Split-half rope over ``axes`` latent axes, over ``dim`` of the head.

    Coordinates arrive in ``[-1, 1]``, so a whole axis spans one turn of the lowest
    frequency, and the angles are duplicated to fill ``dim`` because
    :func:`~thenoise.utils.rope.apply_rope_split_half` pairs index ``i`` with
    ``i + dim/2``.
    """

    def __init__(self, dim: int, rotary_base: float = ROPE_THETA, axes: int = 2):
        super().__init__()
        self.axes = axes
        self.angle_scale = 2.0 * math.pi
        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[S, axes]`` -> the ``(cos, sin)`` pair ``[1, 1, S, dim // 2]``."""
        return split_half_rope_axes(self.inv_freq, self.angle_scale)(ids)


class Attention(nn.Module):
    """Self-attention with per-head QK norms; q, k and v interleave per head in qkv."""

    def __init__(self, heads: int, dim_head: int, bias: bool = True, eps: float = NORM_EPS):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.eps = eps
        self.to_qkv = QuantizedLinear(inner, inner * 3, bias=bias)
        self.to_out = QuantizedLinear(inner, inner, bias=bias)

    def forward(self, x, rope, pre_norm, residual, residual_scale):
        batch, seq, _ = x.shape
        qkv = self.to_qkv(pre_norm(x)).view(batch, seq, -1, 3 * self.dim_head)
        # Head-major, because that is the axis the QK norm and the rope read.
        query, key, value = (t.transpose(1, 2) for t in torch.chunk(qkv, 3, dim=-1))

        cos, sin = rope
        query = apply_rope_split_half(self.head_norm(query), cos, sin).to(x)
        key = apply_rope_split_half(self.head_norm(key), cos, sin).to(x)

        out = eager_attention([query, key, value])
        out = torch.nan_to_num(out)
        return torch.addcmul(residual, self.to_out(out), residual_scale)

    def head_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head RMS norm (the reference's non-affine ``norm_q``/``norm_k``), in fp32."""
        return F.rms_norm(x.float(), (self.dim_head,), eps=self.eps)


class FeedForward(nn.Module):
    """Gated SiLU feed-forward, its residual scaled by ``residual_scale``."""

    def __init__(self, dim: int, mult: int = 4, bias: bool = True):
        super().__init__()
        inner = dim * mult
        self.w1 = QuantizedLinear(dim, inner * 2, bias=bias)
        self.w2 = QuantizedLinear(inner, dim, bias=bias)

    def forward(self, x, pre_norm, residual, residual_scale):
        gate, up = self.w1(pre_norm(x)).chunk(2, dim=-1)
        return torch.addcmul(residual, self.w2(F.silu(gate) * up), residual_scale)


class TransformerBlock(nn.Module):
    def __init__(self, heads: int, dim_head: int, bias: bool = True, eps: float = NORM_EPS):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = Attention(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff = FeedForward(dim=dim, bias=bias)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def forward(self, x, rope):
        x = self.attn(x, rope, self.norm1, residual=x, residual_scale=self.scale1)
        return self.ff(x, self.norm2, residual=x, residual_scale=self.scale2)


class ViTDecoder(nn.Module):
    """The H3 pixel decoder: a ViT over patchified latent cells.

    A token is one latent cell and ``proj_out`` answers the pixels held inside it,
    ``patch_size`` per spatial axis and ``patch_size_t`` in time — 4 for the video
    checkpoint, 1 for the image one, which is then taken and returned a frame at a
    time. The sequence ends in the learnable register tokens plus one spare row
    (``mask_token``) that inference never reads, parked at the grid origin so the rope
    reads them as background.
    """

    def __init__(self, patch_size: int, dim: int, num_layers: int, patch_size_t: int = 1,
                 in_channels: int = Z_CHANNELS, out_channels: int = 3,
                 dim_head: int = HEAD_DIM, rope_dim_ratio: float = ROPE_DIM_RATIO,
                 rope_axes: int = 2, num_register_tokens: int = NUM_REGISTER_TOKENS,
                 eps: float = NORM_EPS):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.pos_embed = RotaryEmbedding(int(dim_head * rope_dim_ratio), axes=rope_axes)
        self.x_embedder = QuantizedLinear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        # Unused at inference; kept so the checkpoint loads without leftover keys.
        self.register_buffer("mask_token", torch.empty(1, 1, dim))

        heads = dim // dim_head
        if heads * dim_head != dim:
            raise ValueError(
                f"a decoder width of {dim} is not a whole number of {dim_head}-dim heads"
            )
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(heads=heads, dim_head=dim_head, eps=eps) for _ in range(num_layers)
        )

        self.norm_out = nn.LayerNorm(dim, eps=eps)
        self.proj_out = QuantizedLinear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, z, h, w]`` or ``[B, z, t, h, w]`` latent -> pixels of the same rank."""
        still = x.ndim == 4
        if still:
            x = x.unsqueeze(2)
        batch, _, frames, height, width = x.shape

        h = self.x_embedder(x.flatten(2).transpose(1, 2))
        num_patches = h.shape[1]

        registers = self.register_tokens.expand(batch, -1, -1)
        h = torch.cat([h, registers, torch.zeros_like(h[:, 0:1, :])], dim=1)

        axes = self.pos_embed.axes
        ids = token_ids((frames, height, width)[-axes:], x.device, torch.float32)
        suffix = torch.zeros(1 + self.num_register_tokens, axes, device=ids.device, dtype=ids.dtype)
        rope = self.pos_embed(torch.cat([ids, suffix], dim=0))

        for block in self.transformer_blocks:
            h = block(h, rope)

        out = self.proj_out(self.norm_out(h))[:, :num_patches]
        p = self.patch_size
        pixels = out.view(batch, frames, height, width, self.out_channels,
                          self.patch_size_t, p, p)
        pixels = pixels.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            batch, self.out_channels, frames * self.patch_size_t,
            height * p, width * p)
        return pixels.squeeze(2) if still else pixels


# --------------------------------------------------------------------------- tiling


def split_tiles(length: int, tile_size: int, blend: int, cell_px: int):
    """Tile ``(starts, lengths, blends)`` covering ``length`` pixels.

    Neighbours overlap by at least ``blend`` and by as much more as it takes to land on
    ``length`` exactly, a latent cell at a time. A frame no larger than a tile is its
    own single tile.
    """
    if blend >= tile_size:
        raise ValueError(f"a tile blend of {blend} leaves nothing to decode of a {tile_size} tile")
    if tile_size >= length:
        return [0], [length], []

    count = math.ceil(length / tile_size)
    overlaps = [blend] * (count - 1)
    while tile_size * count - sum(overlaps) < length:
        count += 1
        overlaps = [blend] * (count - 1)
    for i in range((tile_size * count - sum(overlaps) - length) // cell_px):
        overlaps[i % (count - 1)] += cell_px

    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile_size - overlap)
    return starts, [tile_size] * count, overlaps


def blend_tiles(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    """Cross-fade the leading ``extent`` slices of ``b`` against the trailing ones of ``a``."""
    extent = min(a.shape[dim], b.shape[dim], extent)
    if extent <= 0:
        return b
    ramp = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * b.ndim
    shape[dim] = extent
    ramp = ramp.view(shape)
    tail = [slice(None)] * a.ndim
    tail[dim] = slice(-extent, None)
    head = [slice(None)] * b.ndim
    head[dim] = slice(0, extent)
    blended = a[tuple(tail)] * (1.0 - ramp / extent) + b[tuple(head)] * (ramp / extent)
    if extent < b.shape[dim]:
        rest = [slice(None)] * b.ndim
        rest[dim] = slice(extent, None)
        return torch.cat([blended, b[tuple(rest)]], dim=dim)
    return blended


def tiled_moments(x: torch.Tensor, encode: Callable[[torch.Tensor], torch.Tensor],
                  cell_px: int, tile_size: int, blend: int) -> torch.Tensor:
    """Encode ``x``'s spatial extent as overlapping tiles, cross-fading the seams.

    ``encode(tile)`` answers the latent cells of one tile. Neighbours are blended in
    latent space, where a cell is ``cell_px`` pixels wide.
    """
    p = cell_px
    y_at, y_len, y_blend = split_tiles(x.shape[-2], tile_size, blend, p)
    x_at, x_len, x_blend = split_tiles(x.shape[-1], tile_size, blend, p)

    rows = [[encode(x[..., i:i + li, j:j + lj]) for j, lj in zip(x_at, x_len)]
            for i, li in zip(y_at, y_len)]
    y_blend = [b // p for b in y_blend]
    x_blend = [b // p for b in x_blend]

    out_rows = []
    for i, row in enumerate(rows):
        out = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_tiles(rows[i - 1][j], tile, y_blend[i - 1], dim=-2)
            if j > 0:
                tile = blend_tiles(row[j - 1], tile, x_blend[j - 1], dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., :-y_blend[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, :-x_blend[j]]
            out.append(tile)
        out_rows.append(torch.cat(out, dim=-1))
    return torch.cat(out_rows, dim=-2)


def tiled_canvas(z: torch.Tensor, decode: Callable[[torch.Tensor], torch.Tensor],
                 cell_px: int, tile_size: int, blend: int) -> torch.Tensor:
    """Decode ``z``'s spatial extent as overlapping tiles, cross-fading the seams.

    ``decode(tile)`` answers the pixels of one tile, ``cell_px`` of them to a latent
    cell. Each tile is a grid of its own, which keeps every tile's rope span inside the
    range the decoder was trained on, and only one row of tiles is alive at a time:
    a tile can only blend against the row above it and the tile to its left.
    """
    height, width = z.shape[-2] * cell_px, z.shape[-1] * cell_px
    y_at, y_len, y_blend = split_tiles(height, tile_size, blend, cell_px)
    x_at, x_len, x_blend = split_tiles(width, tile_size, blend, cell_px)

    canvas: Optional[torch.Tensor] = None
    strip: Optional[torch.Tensor] = None
    out_y = 0
    for i, (y, yl) in enumerate(zip(y_at, y_len)):
        row = z[..., y // cell_px:y // cell_px + yl // cell_px, :]
        new_strip = None
        left_tail = None
        out_x = 0
        tile_height = 0
        for j, x in enumerate(x_at):
            tile = decode(row[..., x // cell_px:x // cell_px + x_len[j] // cell_px])
            if i > 0:
                tile = blend_tiles(strip[..., :, x:x + x_len[j]], tile, y_blend[i - 1], dim=-2)
            if j > 0:
                tile = blend_tiles(left_tail, tile, x_blend[j - 1], dim=-1)
            last_x = j == len(x_at) - 1
            left_tail = None if last_x else tile[..., :, -x_blend[j]:].clone()
            if not last_x:
                tile = tile[..., :, :-x_blend[j]]
            last_y = i == len(y_at) - 1
            if not last_y:
                if new_strip is None:
                    new_strip = torch.empty((*tile.shape[:-2], y_blend[i], width),
                                            dtype=tile.dtype, device=tile.device)
                new_strip[..., :, out_x:out_x + tile.shape[-1]] = tile[..., -y_blend[i]:, :]
                tile = tile[..., :-y_blend[i], :]
            if canvas is None:
                canvas = torch.empty((*tile.shape[:-2], height, width),
                                     dtype=tile.dtype, device=tile.device)
            canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
            tile_height = tile.shape[-2]
            out_x += tile.shape[-1]
        strip = new_strip
        out_y += tile_height
    return canvas


# --------------------------------------------------------------------------- temporal plan


@dataclass(frozen=True)
class TemporalPlan:
    """How the video codec walks a latent's frames on the way out.

    The encoder is causal with a temporal stride of ``ratio_t``, so a decoded clip opens
    with ``pre_padding`` frames standing in for context it does not have, and they are
    dropped. Consecutive clips therefore re-read ``overlap_tokens`` latent frames and
    cross-fade the ``frame_overlap`` pixels that buys, so no frame is ever decoded cold;
    the same overlap is what costs the sequence ``token_drop`` tokens at the end.

    :attr:`frames` is the closed form of that walk: the first latent of every group of
    ``chunk_tokens`` carries the frame the padding took from it and the rest carry
    ``ratio_t`` each, which is the same schedule the DiT lays its own time axis out by.
    """

    latent_t: int
    ratio_t: int
    chunk_tokens: int
    pre_padding: int
    overlap_tokens: int
    frame_overlap: int
    pad_tokens: int
    clips: int
    frames: int

    @classmethod
    def of(cls, latent_t: int, clip_length: int = CLIP_LENGTH, token_drop: int = TOKEN_DROP,
           ratio_t: int = TIME_RATIO) -> "TemporalPlan":
        pre_padding = (-clip_length) % ratio_t
        chunk = -(-clip_length // ratio_t)
        overlap = (-token_drop) % chunk
        padded = latent_t + token_drop
        pad = (-padded) % chunk
        clips = (padded + pad) // chunk - int(token_drop > 0)
        if clips < 1:
            # Too few tokens for one clip: pad a whole clip's worth more, because a clip
            # decoded cold is worse than a padded one.
            pad += chunk
            clips = 1
        head = clip_length - (chunk - 1) * ratio_t
        whole, rest = divmod(latent_t, chunk)
        return cls(
            latent_t=latent_t, ratio_t=ratio_t, chunk_tokens=chunk, pre_padding=pre_padding,
            overlap_tokens=overlap, frame_overlap=max(overlap * ratio_t - pre_padding, 0),
            pad_tokens=pad, clips=clips,
            frames=whole * clip_length + (0 if rest == 0 else head + ratio_t * (rest - 1)),
        )

    @classmethod
    def of_frames(cls, pixel_frames: int, clip_length: int = CLIP_LENGTH,
                  token_drop: int = TOKEN_DROP,
                  ratio_t: int = TIME_RATIO) -> "TemporalPlan":
        """The plan whose clip runs exactly ``pixel_frames`` — :meth:`of` upside down.

        Only the counts the walk can reach have a plan: a clip opens with the one frame the
        padding leaves and then advances ``ratio_t`` at a time. A count off that ladder is
        an error naming the ladder rather than a rounding, because which reachable length a
        request becomes is a policy the caller owns.
        """
        chunk = -(-clip_length // ratio_t)
        whole, rest = divmod(pixel_frames, clip_length)
        if rest and rest % ratio_t != 1:
            raise ValueError(
                f"this codec runs {clip_length} frames per clip, opening with 1 and then "
                f"{ratio_t} per latent frame: a clip is {clip_length}k + "
                f"{', '.join(str(1 + ratio_t * i) for i in range(chunk))} frames long, "
                f"not {pixel_frames}"
            )
        return cls.of(whole * chunk + (0 if rest == 0 else rest // ratio_t + 1),
                      clip_length=clip_length, token_drop=token_drop, ratio_t=ratio_t)


def snap_frames(wanted: int, *, upward: bool, clip_length: int = CLIP_LENGTH,
                token_drop: int = TOKEN_DROP, ratio_t: int = TIME_RATIO) -> int:
    """The clip length this codec runs, nearest the one that was asked for.

    The decoder takes any latent frame count, but the release only ever runs the shortest
    clip it trained on and whole clips more: :meth:`TemporalPlan.of_frames` reaches
    ``clip_length`` further per step. An explicit frame count goes upward, because a
    request that names a length wants it kept; a duration takes the nearest, ties away
    from zero, because it names a length that may be rounded.
    """
    shortest = TemporalPlan.of(2, clip_length, token_drop, ratio_t=ratio_t).frames
    offset = wanted - shortest
    steps = (-(-offset // clip_length) if upward
             else math.floor(offset / clip_length + 0.5))
    return shortest + max(steps, 0) * clip_length


# --------------------------------------------------------------------------- video codec


class AutoencoderKLH3Video(nn.Module):
    """H3 video codec: canonical ``[B, 24, t, h, w]`` latents, frames in and out of
    the range ``[-1, 1]``.

    A decode is chunked in time and tiled in space, and hands frames out as they finish:
    the whole clip never exists at once, which on a 124-frame clip is the difference
    between a working set and an allocator error.
    """

    compute_dtype = COMPUTE_DTYPE

    z_dim = Z_CHANNELS
    spatial_compression = math.prod(SPACE_DOWN)
    temporal_compression = math.prod(TIME_DOWN)
    pixel_channels = 3

    def __init__(self, ch: int = CH, ch_mult=CH_MULT, space_down=SPACE_DOWN,
                 time_down=TIME_DOWN, dim: int = 2048, num_layers: int = 36,
                 clip_length: int = CLIP_LENGTH, token_drop: int = TOKEN_DROP,
                 latents_mean=LATENTS_MEAN, latents_std=LATENTS_STD,
                 tile_size: int = TILE_SIZE, tile_blend: int = TILE_BLEND):
        """The release's geometry by default, as :func:`load_h3_video_vae` checks it; the
        arguments exist so the codec can be exercised at a fraction of the width.

        The decoder un-patchifies exactly what the encoder's strides compressed, so the
        two patch sizes are read off ``space_down``/``time_down`` rather than stated
        again: a file disagreeing with them is not this codec.
        """
        super().__init__()
        self.patch_size = math.prod(space_down)
        self.patch_size_t = math.prod(time_down)
        self.clip_length = clip_length
        self.token_drop = token_drop
        self.tile_size = tile_size
        self.tile_blend = tile_blend

        self.encoder = EncoderFCN3D(ch=ch, ch_mult=ch_mult, space_down=space_down,
                                    time_down=time_down)
        self.quant_conv = nn.Conv3d(2 * Z_CHANNELS, 2 * Z_CHANNELS, 1)
        self.post_quant_conv = nn.Conv3d(Z_CHANNELS, Z_CHANNELS, 1)
        self.decoder = ViTDecoder(self.patch_size, dim=dim, num_layers=num_layers,
                                  patch_size_t=self.patch_size_t,
                                  rope_dim_ratio=VIDEO_ROPE_DIM_RATIO, rope_axes=3)

        self.register_buffer("latents_mean", torch.tensor(list(latents_mean)).view(1, -1, 1, 1, 1),
                             persistent=True)
        self.register_buffer("latents_std", torch.tensor(list(latents_std)).view(1, -1, 1, 1, 1),
                             persistent=True)
        self.register_buffer("pixel_mean", torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(PIXEL_STD).view(1, 3, 1, 1, 1), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.decoder.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    def decode_output_shape(self, latent_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """The pixel shape a latent of this shape decodes to."""
        batch, _, frames, height, width = latent_shape
        return (batch, self.pixel_channels,
                TemporalPlan.of(frames, self.clip_length, self.token_drop,
                             ratio_t=self.patch_size_t).frames,
                height * self.patch_size, width * self.patch_size)

    def _encode_moments(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(x))

    def _tiled_moments(self, x: torch.Tensor) -> torch.Tensor:
        return tiled_moments(x, self._encode_moments, self.patch_size,
                             self.tile_size, self.tile_blend)

    def _decode_pixels(self, z: torch.Tensor) -> torch.Tensor:
        """Un-normalized latent -> raw decoder pixels of the whole clip."""
        return self.decoder(self.post_quant_conv(z))

    def _tiled_decode(self, z: torch.Tensor, frames: slice = slice(None)) -> torch.Tensor:
        """One clip's latent -> pixels, tiled spatially, keeping only ``frames``.

        The frames are dropped per tile rather than per clip: a tile's pixels are the
        largest thing this decode holds.
        """
        return tiled_canvas(z, lambda tile: self._decode_pixels(tile)[:, :, frames],
                            self.patch_size, self.tile_size, self.tile_blend)

    def decode_to_pixel_frames(self, latents: torch.Tensor) -> Iterator[torch.Tensor]:
        """Canonical latent -> clamped frames, handed out as they are decoded.

        Each clip is decoded with its predecessor's tail re-read and cross-faded in, and
        opens by dropping the padding frames a clip always stands in for.
        """
        z = latents.to(self.device, self.dtype)
        z = z * self.latents_std.to(z) + self.latents_mean.to(z)

        plan = TemporalPlan.of(z.shape[2], self.clip_length, self.token_drop,
                                 ratio_t=self.patch_size_t)

        if z.shape[2] == 1:
            # A lone latent decodes badly on its own: the decoder only ever saw a frame
            # inside a clip, so it is decoded as a clip of copies and the frame it stands
            # for is kept from past the warm-up.
            clip = z.expand(-1, -1, plan.chunk_tokens, -1, -1)
            frame = slice(plan.pre_padding, plan.pre_padding + 1)
            yield self._finalize(self._tiled_decode(clip, frames=frame))
            return

        if plan.pad_tokens:
            z = torch.cat([z, z[:, :, -1:].expand(-1, -1, plan.pad_tokens, -1, -1)], dim=2)

        left = plan.frames
        held: Optional[torch.Tensor] = None
        clip_frames = plan.chunk_tokens * plan.ratio_t

        for i in range(plan.clips):
            start = i * plan.chunk_tokens
            clip = z[:, :, start:start + plan.chunk_tokens + plan.overlap_tokens]
            decoded = self._tiled_decode(clip)
            pieces = [decoded[:, :, plan.pre_padding:clip_frames]]
            if plan.overlap_tokens:
                pieces.append(decoded[:, :, clip_frames + plan.pre_padding:])

            for j, piece in enumerate(pieces):
                if j == 0:
                    if held is not None:
                        piece = blend_tiles(held, piece, plan.frame_overlap, dim=2)
                        held = None
                    keep, left = _take(piece, left)
                    if keep is not None:
                        yield self._finalize(keep)
                else:
                    held = piece
            if held is not None:
                keep, left = _take(held, left)
                if keep is not None:
                    yield self._finalize(keep)

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Canonical latent -> the whole clip. Streams; it does not hold it."""
        return torch.cat(list(self.decode_to_pixel_frames(latents)), dim=2)

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Frames in ``[-1, 1]`` -> the normalized canonical latent.

        The posterior *mean*, where the image codec takes a sample: a clip is encoded
        once and a keyframe is only ever the first latent of one.
        """
        x = pixels.to(self.device, self.dtype)
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = ((x + 1.0) * 0.5 - self.pixel_mean.to(x)) / self.pixel_std.to(x)

        moments = self._encode(x).float()
        mean = moments[:, :self.z_dim]
        return (mean - self.latents_mean.to(mean)) / self.latents_std.to(mean)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel-normalized frames -> posterior moments, a clip at a time."""
        if x.shape[2] == 1:
            return self._tiled_moments(x)[:, :, -1:]

        clips = []
        for start in range(0, x.shape[2], self.clip_length):
            clip = x[:, :, start:start + self.clip_length]
            short = self.clip_length - clip.shape[2]
            if short > 0:
                clip = torch.cat([clip, clip[:, :, -1:].expand(-1, -1, short, -1, -1)], dim=2)
            clips.append(self._tiled_moments(clip))

        moments = torch.cat(clips, dim=2)
        return moments[:, :, :-self.token_drop] if self.token_drop else moments

    def _finalize(self, pixels: torch.Tensor) -> torch.Tensor:
        """Codec-space pixels -> the ``[-1, 1]`` the pipeline works in."""
        pixels = (pixels * self.pixel_std.to(pixels) + self.pixel_mean.to(pixels)) * 2.0 - 1.0
        return pixels.clamp(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_to_pixels(self.encode_pixels_to_latents(x))


def _take(piece: torch.Tensor, left: int) -> tuple[Optional[torch.Tensor], int]:
    """The first ``left`` frames of ``piece``, and what is left of the budget after it.

    The plan says how many frames a latent is owed; the last clip's tail is padding that
    has to be trimmed to that promise rather than streamed past it.
    """
    if left <= 0 or piece.shape[2] <= 0:
        return None, left
    keep = piece[:, :, :left]
    return keep, left - keep.shape[2]


# --------------------------------------------------------------------------- image codec


class AutoencoderKLH3Image(nn.Module):
    """H3-Image VAE: canonical ``[B, 24, H//16, W//16]`` latents, pixels in/out
    in ``[-1, 1]``.
    """

    compute_dtype = COMPUTE_DTYPE

    z_dim = Z_CHANNELS
    spatial_compression = math.prod(SPACE_DOWN)
    pixel_channels = 3

    def __init__(self, ch: int = CH, ch_mult=CH_MULT, dim: int = 1024, num_layers: int = 24,
                 patch_size: Optional[int] = None, tile_size: int = TILE_SIZE,
                 tile_blend: int = TILE_BLEND):
        """``ch``/``ch_mult`` are the encoder's stage widths, ``dim``/``num_layers`` the
        decoder's. The defaults are the release's, which :func:`load_h3_image_vae`
        checks; the arguments exist so the geometry can be exercised at a fraction of
        the width.
        """
        super().__init__()
        self.patch_size = self.spatial_compression if patch_size is None else patch_size
        self.tile_size = tile_size
        self.tile_blend = tile_blend

        self.encoder = EncoderFCN3D(ch=ch, ch_mult=ch_mult)
        self.quant_conv = nn.Conv3d(2 * Z_CHANNELS, 2 * Z_CHANNELS, 1)
        self.post_quant_conv = nn.Conv2d(Z_CHANNELS, Z_CHANNELS, 1)
        self.decoder = ViTDecoder(self.patch_size, dim=dim, num_layers=num_layers)

        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("pixel_mean", torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(PIXEL_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.decoder.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    def _encode_moments(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel-normalized frames -> the ``2 * z`` posterior moments of the same grid."""
        return self.quant_conv(self.encoder(x.unsqueeze(2)))

    def _decode_pixels(self, z: torch.Tensor) -> torch.Tensor:
        """Un-normalized latent -> raw decoder pixels."""
        return self.decoder(self.post_quant_conv(z))

    def _tiled_moments(self, x: torch.Tensor) -> torch.Tensor:
        return tiled_moments(x, self._encode_moments, self.patch_size,
                             self.tile_size, self.tile_blend)

    def _tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        return tiled_canvas(z, self._decode_pixels, self.patch_size,
                            self.tile_size, self.tile_blend)

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` in [-1, 1] -> the normalized canonical latent.

        A posterior sample on :data:`ENCODE_SEED`, never the mean.
        """
        x = pixels.to(self.device, self.dtype)
        x = ((x + 1.0) * 0.5 - self.pixel_mean.to(x)) / self.pixel_std.to(x)

        moments = self._tiled_moments(x).float()
        mean, logvar = moments.chunk(2, dim=1)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype,
                            generator=torch.Generator(device=mean.device)
                            .manual_seed(ENCODE_SEED))
        latents = torch.addcmul(mean, torch.exp(0.5 * logvar.clamp(-30.0, 20.0)), noise)

        z = latents.squeeze(2)
        return (z - self.latents_mean.to(z)) / self.latents_std.to(z)

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized canonical latent -> pixels clamped to [-1, 1]."""
        z = latents.to(self.device, self.dtype)
        z = z * self.latents_std.to(z) + self.latents_mean.to(z)
        pixels = self._tiled_decode(z)
        pixels = (pixels * self.pixel_std.to(pixels) + self.pixel_mean.to(pixels)) * 2.0 - 1.0
        return pixels.clamp(-1.0, 1.0)


#: The pair no other codec here has: the latent embedding on the decoder side, the
#: pixel conv on the encoder's. Their widths are the geometry below.
H3_CODEC_SIGNATURE = Signature("H3-Image codec",
                               keys=("decoder.x_embedder.weight", "encoder.conv_in.weight"))


def _geometry(header: Header) -> dict:
    """The widths and the decoder depth the header states.

    The encoder's stage *strides* are not in the header, so they stay the release's
    (:data:`SPACE_DOWN`, :data:`TIME_DOWN`).
    """
    embed = header.shapes.get("decoder.x_embedder.weight")
    conv_in = header.shapes.get("encoder.conv_in.weight")
    if embed is None or embed[1] != Z_CHANNELS:
        raise ValueError(
            f"{header.path}: 'decoder.x_embedder.weight' taking {Z_CHANNELS} latent "
            "channels is missing (not a H3-Image VAE?)"
        )
    if conv_in is None or conv_in[1:] != (3, 3, 3, 3):
        raise ValueError(
            f"{header.path}: 'encoder.conv_in.weight' taking 3 pixel channels is "
            "missing (not a H3-Image VAE?)"
        )
    return {"ch": conv_in[0], "dim": embed[0],
            "num_layers": header.block_count("decoder.transformer_blocks.")}


def load_h3_image_vae(
    vae_path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> AutoencoderKLH3Image:
    """Load the H3-Image VAE from ``vae_path``.

    The codec runs in its own :attr:`AutoencoderKLH3Image.compute_dtype`;
    ``dtype`` is an override, not a request for FP32.
    """

    def construct(header: Header) -> AutoencoderKLH3Image:
        geometry = _geometry(header)
        logger.info("H3-Image VAE: encoder %d channels, %d-wide %d-layer decoder",
                    geometry["ch"], geometry["dim"], geometry["num_layers"])
        return AutoencoderKLH3Image(**geometry)

    return load_codec(vae_path, device,
                      what="H3-Image VAE",
                      signature=H3_CODEC_SIGNATURE,
                      construct=construct,
                      dtype=dtype)


#: What the video checkpoint calls its own geometry inside ``__metadata__``.
VIDEO_GEOMETRY_SECTION = "minimax_h3_video_vae"

#: The video codec's own claim. The Sol decoder is a different module tree —
#: ``decoder.blocks.N``, no ``x_embedder`` — so it fails this one on its face rather
#: than half-loading.
H3_VIDEO_SIGNATURE = Signature("H3 video codec",
                               keys=("decoder.x_embedder.weight",
                                     "decoder.transformer_blocks.0.attn.to_qkv.weight",
                                     "encoder.conv_in.weight"))


def _video_geometry(header: Header) -> dict:
    """The widths, the depth and the clip policy, as the file states them.

    The released checkpoint carries a :data:`VIDEO_GEOMETRY_SECTION` metadata section with
    the encoder's stage widths and strides, the decoder's rope and the clip length, and
    ships its latent statistics as tensors. A file with no such section — a re-export, or
    a test fixture — is taken to be the release's geometry, which is what the module
    constants say it is.
    """
    conv_in = header.shapes.get("encoder.conv_in.weight")
    embed = header.shapes.get("decoder.x_embedder.weight")
    if conv_in is None or conv_in[1:] != (3, 3, 3, 3):
        raise ValueError(
            f"{header.path}: 'encoder.conv_in.weight' taking 3 pixel channels is "
            "missing (not an H3 video VAE?)"
        )
    if embed is None or embed[1] != Z_CHANNELS:
        raise ValueError(
            f"{header.path}: 'decoder.x_embedder.weight' taking {Z_CHANNELS} latent "
            "channels is missing (not an H3 video VAE?)"
        )

    config = header.metadata_json(VIDEO_GEOMETRY_SECTION)
    meta = config.get("source_config") or {}
    geometry = {
        "ch": conv_in[0],
        "ch_mult": tuple(meta.get("ch_mult", CH_MULT)),
        "space_down": tuple(meta.get("space_down", SPACE_DOWN)),
        "time_down": tuple(meta.get("time_down", TIME_DOWN)),
        "dim": embed[0],
        "num_layers": header.block_count("decoder.transformer_blocks."),
        "clip_length": config.get("vae_clip_length", CLIP_LENGTH),
        "token_drop": config.get("vae_token_drop", TOKEN_DROP),
    }

    # The decoder answers the pixels one latent cell holds, so its output width is the
    # encoder's own compression. A file that disagrees pairs an encoder with a decoder
    # this module cannot build for it.
    out = header.shapes.get("decoder.proj_out.weight")
    pixels = (AutoencoderKLH3Video.pixel_channels * math.prod(geometry["time_down"])
              * math.prod(geometry["space_down"]) ** 2)
    if out is None or out[0] != pixels:
        raise ValueError(
            f"{header.path}: 'decoder.proj_out.weight' answering {pixels} pixels per "
            "latent cell is missing (not an H3 video VAE?)"
        )
    return geometry


def _video_state(state_dict: dict, model: torch.nn.Module) -> dict:
    """Reshape the shipped latent statistics to the codec's broadcast-ready layout."""
    for name in ("latents_mean", "latents_std"):
        if name in state_dict:
            state_dict[name] = state_dict[name].view(1, -1, 1, 1, 1)
    return state_dict


def load_h3_video_vae(
    vae_path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> AutoencoderKLH3Video:
    """Load the H3 video VAE from ``vae_path``.

    The codec runs in its own :attr:`AutoencoderKLH3Video.compute_dtype`; ``dtype`` is an
    override, not a request for FP32.
    """

    def construct(header: Header) -> AutoencoderKLH3Video:
        geometry = _video_geometry(header)
        logger.info("H3 video VAE: encoder %d channels, %d-wide %d-layer decoder, "
                    "%d-frame clips dropping %d tokens",
                    geometry["ch"], geometry["dim"], geometry["num_layers"],
                    geometry["clip_length"], geometry["token_drop"])
        return AutoencoderKLH3Video(**geometry)

    return load_codec(vae_path, device,
                      what="H3 video VAE",
                      signature=H3_VIDEO_SIGNATURE,
                      construct=construct,
                      state_map=_video_state,
                      dtype=dtype)


__all__ = [
    "AutoencoderKLH3Image",
    "AutoencoderKLH3Video",
    "TemporalPlan",
    "snap_frames",
    "load_h3_image_vae",
    "load_h3_video_vae",
    "CLIP_LENGTH",
    "ENCODE_SEED",
    "LATENTS_MEAN",
    "LATENTS_STD",
    "PIXEL_MEAN",
    "PIXEL_STD",
    "TIME_RATIO",
    "TOKEN_DROP",
]
