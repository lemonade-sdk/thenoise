"""H3-Image codec — a 3D causal CNN encoder over a single-frame 2D ViT decoder.

Ported from ComfyUI's ``comfy/ldm/minimax/vae.py``. The module tree, and with it the
checkpoint key names, is verbatim; only the ops change (``comfy.ops`` -> ``torch.nn``,
ComfyUI's fused norm/rope/attention helpers -> their plain equivalents here). Spatial
tiling and temporal chunking are not ported: a whole image goes through each half.

The codec is a video codec used as a still one, and both halves lose nothing by it:

  * **encode** — the H3 video encoder on one frame. A causal 3D conv sees a lone frame
    through its last temporal tap only, because the zeros padded in front of it
    contribute nothing, so every conv here runs on ``weight[:, :, -1:]``: same answer,
    a third of the arithmetic, no padded activations. The per-frame GroupNorm the video
    path needs is what a plain GroupNorm already computes at ``T = 1``.
  * **decode** — the image checkpoint's own ViT head: 24 layers, 16 heads, rope over
    half the head dim and over the two spatial axes only. The reference spells it
    ``ViT3DDecoder(patch_size_t=1, rope_dims=2)`` and carries a length-1 time axis
    through it; that axis is dropped here, which is why ``post_quant_conv`` is 2D while
    ``quant_conv`` stays 3D.

Latent: 24 channels at 16 pixels per cell, normalized by the release's own per-channel
mean and std — the checkpoint ships none, so the constants below are their one home.

Encoding returns a posterior **sample** on a fixed seed, not the mean. That is the
release's recipe: the decoder was trained on sampled latents and the mean decodes soft.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.utils.attention import eager_attention
from thenoise.utils.rope import apply_rope_split_half
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, load_safetensors
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

#: Seed of the posterior sample encode conditions the decoder on. The H3 keyframe recipe
#: restarts the same stream on purpose, so a reference's latent does not move with the
#: generation's own seed.
ENCODE_SEED = 42

#: Encoder geometry (``ch_mult`` stages, with the spatial and temporal stride of each).
CH = 128
CH_MULT = (1, 2, 2, 4, 4, 8)
NUM_RES_BLOCKS = 2
SPACE_DOWN = (2, 2, 2, 2, 1, 1)
TIME_DOWN = (1, 2, 2, 1, 1, 1)
Z_CHANNELS = 24

#: Decoder geometry. ``HEAD_DIM`` is what the rope is a ratio of, so the head count
#: follows from the width the checkpoint states.
HEAD_DIM = 64
ROPE_DIM_RATIO = 0.5
ROPE_THETA = 100.0
NUM_REGISTER_TOKENS = 4
NORM_EPS = 1e-5


# --------------------------------------------------------------------------- encoder


class CausalConv3d(nn.Conv3d):
    """Reflect-spatial, causal-temporal 3D convolution over a single frame.

    ``padding`` is the (temporal, height, width) radius; the temporal one is causal, so
    it pads in front only — and at ``T = 1`` everything padded is zero. The frame is
    therefore convolved with the kernel's last temporal tap and no temporal padding,
    which is what :meth:`forward` does instead of building the padded video.
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
        out = F.conv3d(x, self.weight[:, :, -1:], self.bias, self.stride)
        return out if residual is None else out.add_(residual)


def group_norm_3d(num_channels: int) -> nn.GroupNorm:
    """GroupNorm over a single frame — the video path's per-frame norm, at ``T = 1``."""
    return nn.GroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    """Stride convolution, spatially asymmetric-padded so the odd edge is kept."""

    def __init__(self, in_channels, out_channels, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_channels, out_channels, kernel_size=3, padding=(1, 0, 0),
            stride=(1, space_stride, space_stride),
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
                level.downsample = Downsample3D(block_mid[i_level], mult * ch, space_stride=space_down[i_level])
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


def token_ids(dims: Tuple[int, int], device, dtype) -> torch.Tensor:
    """Cell centres of the latent grid, each axis normalized to ``[-1, 1]``."""
    axes = [(torch.arange(0.5, dim, dtype=dtype, device=device) / dim) * 2.0 - 1.0 for dim in dims]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, 1)


class RotaryEmbedding2D(nn.Module):
    """Split-half rope over the ``(h, w)`` latent axes, over ``dim`` of the head dim.

    Coordinates arrive in ``[-1, 1]``, so a whole axis spans exactly one turn of the
    lowest frequency. The angles are duplicated to fill ``dim`` because
    :func:`~thenoise.utils.rope.apply_rope_split_half` pairs index ``i`` with
    ``i + dim/2``.
    """

    def __init__(self, dim: int, rotary_base: float = ROPE_THETA, n_dim: int = 2):
        super().__init__()
        self.angle_scale = 2.0 * math.pi
        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[S, n_dim]`` -> the ``(cos, sin)`` pair ``[S, 1, 1, dim]``."""
        angles = (self.angle_scale * ids[:, :, None] * self.inv_freq.to(ids.device)).flatten(1)
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos()[:, None, None, :], angles.sin()[:, None, None, :]


class Attention(nn.Module):
    """Self-attention with per-head QK norms; q, k and v interleave per head in qkv."""

    def __init__(self, heads: int, dim_head: int, bias: bool = True, eps: float = NORM_EPS):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.eps = eps
        self.to_qkv = nn.Linear(inner, inner * 3, bias=bias)
        self.to_out = nn.Linear(inner, inner, bias=bias)

    def forward(self, x, rope, pre_norm, residual, residual_scale):
        batch, seq, _ = x.shape
        qkv = self.to_qkv(pre_norm(x)).view(batch, seq, -1, 3 * self.dim_head)
        # [B, S, heads, 3*dh] -> head-major q, k, v: the head dim is the last axis the
        # QK norm and the rope both read.
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
        self.w1 = nn.Linear(dim, inner * 2, bias=bias)
        self.w2 = nn.Linear(inner, dim, bias=bias)

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


class ViTImageDecoder(nn.Module):
    """The image checkpoint's decoder: a 2D ViT that turns latent cells into pixels.

    ``patch_size`` pixels per latent cell come out of ``proj_out`` as one flattened
    patch, and the sequence ends in the learnable register tokens plus one spare row
    (``mask_token``) that inference never reads.
    """

    def __init__(self, patch_size: int, dim: int, num_layers: int,
                 in_channels: int = Z_CHANNELS, out_channels: int = 3,
                 dim_head: int = HEAD_DIM, rope_dim_ratio: float = ROPE_DIM_RATIO,
                 num_register_tokens: int = NUM_REGISTER_TOKENS, eps: float = NORM_EPS):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.pos_embed = RotaryEmbedding2D(int(dim_head * rope_dim_ratio))
        self.x_embedder = nn.Linear(in_channels, dim)
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
        self.proj_out = nn.Linear(dim, out_channels * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, z, H, W]`` latent -> ``[B, 3, H * patch, W * patch]`` pixels."""
        batch, _, height, width = x.shape
        h = self.x_embedder(x.flatten(2).transpose(1, 2))
        num_patches = h.shape[1]

        registers = self.register_tokens.expand(batch, -1, -1)
        h = torch.cat([h, registers, torch.zeros_like(h[:, 0:1, :])], dim=1)

        ids = token_ids((height, width), x.device, torch.float32)
        suffix = torch.zeros(1 + self.num_register_tokens, 2, device=ids.device, dtype=ids.dtype)
        rope = tuple(f.to(x.dtype) for f in self.pos_embed(torch.cat([ids, suffix], dim=0)))

        for block in self.transformer_blocks:
            h = block(h, rope)

        out = self.proj_out(self.norm_out(h))[:, :num_patches]
        patches = out.view(batch, height, width, self.out_channels, self.patch_size, self.patch_size)
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, self.out_channels, height * self.patch_size, width * self.patch_size
        )


# --------------------------------------------------------------------------- the codec


class AutoencoderKLH3Image(nn.Module):
    """H3-Image VAE: canonical ``[B, 24, H//16, W//16]`` latents, pixels in/out
    in ``[-1, 1]``.
    """

    compute_dtype = COMPUTE_DTYPE

    z_dim = Z_CHANNELS
    spatial_compression = math.prod(SPACE_DOWN)
    pixel_channels = 3

    def __init__(self, ch: int = CH, ch_mult=CH_MULT, dim: int = 1024, num_layers: int = 24,
                 patch_size: Optional[int] = None):
        """``ch``/``ch_mult`` are the encoder's stage widths, ``dim``/``num_layers`` the
        decoder's width and depth.

        The defaults are the release's, which the strict load in
        :func:`load_h3_image_vae` checks; the arguments exist so the geometry can
        be exercised at a fraction of the width. ``patch_size`` is fixed by
        :data:`SPACE_DOWN`, whose product is the compression the latent format promises.
        """
        super().__init__()
        self.patch_size = self.spatial_compression if patch_size is None else patch_size

        self.encoder = EncoderFCN3D(ch=ch, ch_mult=ch_mult)
        self.quant_conv = nn.Conv3d(2 * Z_CHANNELS, 2 * Z_CHANNELS, 1)
        self.post_quant_conv = nn.Conv2d(Z_CHANNELS, Z_CHANNELS, 1)
        self.decoder = ViTImageDecoder(self.patch_size, dim=dim, num_layers=num_layers)

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

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` in [-1, 1] -> the normalized canonical latent.

        A posterior sample on :data:`ENCODE_SEED`, never the mean: the decoder was
        trained on samples, and the mean decodes soft.
        """
        x = pixels.to(self.device, self.dtype)
        x = ((x + 1.0) * 0.5 - self.pixel_mean.to(x)) / self.pixel_std.to(x)

        moments = self.quant_conv(self.encoder(x.unsqueeze(2))).float()
        mean, logvar = moments.chunk(2, dim=1)
        noise = torch.randn(
            mean.shape, generator=torch.Generator().manual_seed(ENCODE_SEED)
        ).to(mean)
        latents = torch.addcmul(mean, torch.exp(0.5 * logvar.clamp(-30.0, 20.0)), noise)

        z = latents.squeeze(2)
        return (z - self.latents_mean.to(z)) / self.latents_std.to(z)

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized canonical latent -> pixels clamped to [-1, 1]."""
        z = latents.to(self.device, self.dtype)
        z = z * self.latents_std.to(z) + self.latents_mean.to(z)
        pixels = self.decoder(self.post_quant_conv(z))
        pixels = (pixels * self.pixel_std.to(pixels) + self.pixel_mean.to(pixels)) * 2.0 - 1.0
        return pixels.clamp(-1.0, 1.0)


def _geometry(vae_path: str) -> dict:
    """The widths and the decoder depth ``vae_path`` was written with, out of its header.

    Reading the knobs off the file lets a release with a different ViT head load
    unchanged, and turns a wrong checkpoint into a clear message instead of a shape error
    halfway through the load. The encoder's stage *strides* are not in the header, so
    they stay the release's (:data:`SPACE_DOWN`, :data:`TIME_DOWN`) — whose product is
    the compression the latent format promises — and the strict load right after checks
    everything else against them.
    """
    with MemoryEfficientSafeOpen(vae_path) as f:
        header = f.header
        embed = header.get("decoder.x_embedder.weight")
        conv_in = header.get("encoder.conv_in.weight")
        if embed is None or embed["shape"][1] != Z_CHANNELS:
            raise ValueError(
                f"'decoder.x_embedder.weight' taking {Z_CHANNELS} latent channels is "
                f"missing from {vae_path} (not a H3-Image VAE?)"
            )
        if conv_in is None or conv_in["shape"][1:] != [3, 3, 3, 3]:
            raise ValueError(
                f"'encoder.conv_in.weight' taking 3 pixel channels is missing from "
                f"{vae_path} (not a H3-Image VAE?)"
            )
        depth = 1 + max(
            int(key.split(".")[2]) for key in header
            if key.startswith("decoder.transformer_blocks.")
        )
        return {"ch": conv_in["shape"][0], "dim": embed["shape"][0], "num_layers": depth}


def load_h3_image_vae(
    vae_path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> AutoencoderKLH3Image:
    """Load the H3-Image VAE from ``vae_path``.

    The codec runs in its own :attr:`AutoencoderKLH3Image.compute_dtype`;
    ``dtype`` is an override, not a request for FP32.
    """
    device = torch.device(device)
    if dtype is None:
        dtype = COMPUTE_DTYPE
    logger.info("Loading H3-Image VAE from %s", vae_path)

    geometry = _geometry(vae_path)
    state_dict = load_safetensors(vae_path, device=device)
    vae = AutoencoderKLH3Image(**geometry)
    info = vae.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(
        "Loaded H3-Image VAE (encoder %d, %d-wide %d-layer decoder): %s",
        geometry["ch"], geometry["dim"], geometry["num_layers"], info,
    )

    vae.to(device, dtype)
    return vae


__all__ = [
    "AutoencoderKLH3Image",
    "load_h3_image_vae",
    "ENCODE_SEED",
    "LATENTS_MEAN",
    "LATENTS_STD",
    "PIXEL_MEAN",
    "PIXEL_STD",
]
