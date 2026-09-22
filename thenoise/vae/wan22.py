# Wan 2.2 VAE (AutoencoderKL) — single-frame (2D) port of the official video
# VAE (via ComfyUI's ``comfy/ldm/wan/vae2_2.py``). The video-only machinery is
# dropped: causal 3D convs become plain Conv2d (the loader keeps the last time
# slice of each 5D weight), the per-chunk ``time_conv`` layers and ``feat_cache``
# streaming are removed, and the parameter-free ``AvgDown3D``/``DupUp3D``
# shortcuts are folded to their exact one-frame behaviour (see the classes).
# Net effect: [B, C, H, W] in [-1, 1] <-> [B, z_dim, H/16, W/16] latents
# (2x2 patchify + 8x spatial compression), matching the reference's 4D path.
#
# Two shipped checkpoints share this layout, and ``load_wan22_vae`` infers which
# one it is from the file:
#
#   * Wan 2.2 VAE    — 48ch latent, RGB, 2x2 patchify, temporal kernel 3
#   * Qwen-Image 2.1 — 64ch latent, RGBA, no patchify, temporal kernel 1
#
# The pixel channel count is the one thing that is not architecture: Qwen-Image
# 2.1's VAE takes and returns 4 channels, so ``pixel_channels`` is part of the
# interface and the encode/decode boundary pads an opaque alpha onto RGB input
# (and keeps the alpha on output).
#
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from thenoise.utils.safetensors import load_safetensors

from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``[B, C, H*p, W*p]`` -> ``[B, C*p*p, H, W]`` (row-major p x p blocks)."""
    if patch_size == 1:
        return x
    return rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Inverse of :func:`patchify`."""
    if patch_size == 1:
        return x
    return rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)


# Elements per strip of rows (16.7M -> a 32MB bf16 buffer). Stripping bounds
# cuDNN's conv workspace: a single full-size conv activation at 4K costs GBs, and
# the library picks its algorithm from the tensor it is handed.
STRIP_ELEMS = 2**24


def strip_apply(fn, x: torch.Tensor, scale: int = 1, halo: int = 1, out: torch.Tensor | None = None):
    """Apply a spatial ``fn`` to ``x`` in horizontal strips of rows.

    ``scale`` is the height multiplier of ``fn`` (2 for a 2x upsample), ``halo``
    is the number of neighbouring rows fed in on each side — one per 3x3 conv in
    ``fn``, so the kept rows are bit-identical to the unstripped call. The strip
    count is chosen from the element count, so small activations (and every
    encoder pass at normal resolutions) take the plain ``fn(x)`` path.

    ``out`` is a preallocated result to ADD into (used to fold an upsample
    shortcut into a buffer that already exists instead of allocating a second
    full-size one).
    """
    n = -(-x.numel() * scale * scale // STRIP_ELEMS)
    if n <= 1 and out is None:
        return fn(x)
    add = out is not None
    size = x.shape[-2]
    step = -(-size // n)
    for a in range(0, size, step):
        b = min(size, a + step)
        lo = max(0, a - halo)
        y = fn(x.narrow(-2, lo, min(size, b + halo) - lo)).narrow(-2, (a - lo) * scale, (b - a) * scale)
        if out is None:
            out = y.new_empty(*y.shape[:-2], size * scale, y.shape[-1])
        dst = out.narrow(-2, a * scale, (b - a) * scale)
        if add:
            dst.add_(y)
        else:
            dst.copy_(y)
    return out


class Wan22RMSNorm(nn.Module):
    """RMS norm over the channel dim: ``x * rsqrt(mean(x^2)) * sqrt(C) * gamma``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma


class Wan22ResidualBlock(nn.Module):
    """v2_2 residual block; the fixed 7-slot ``residual`` and 1x1-conv
    ``shortcut`` match the checkpoint's key layout."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            Wan22RMSNorm(in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            Wan22RMSNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The whole block in strips: its intermediates never exist at full size
        # (``halo=2`` covers the two 3x3 convs, so the result is exact).
        return strip_apply(lambda s: self.residual(s).add_(self.shortcut(s)), x, halo=2)


class Wan22AttentionBlock(nn.Module):
    """Single-head spatial attention with 1x1 QKV convs (``norm``/``to_qkv``/``proj``)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = Wan22RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.size()
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.view(b, c, -1).transpose(1, 2)  # (b, h*w, c)
        k = k.view(b, c, -1).transpose(1, 2)
        v = v.view(b, c, -1).transpose(1, 2)
        # Manual single-head attention: ROCm's fused SDPA backends produce
        # broken-pixel artifacts in VAE decoders (see thenoise/vae/qwen_image.py).
        attn = (q @ k.transpose(-2, -1)) / c**0.5  # 1/sqrt(head_dim), single head
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return self.proj(x) + identity


class Wan22Resample(nn.Module):
    """2x spatial up/down sampling.

    The v2_2 ``time_conv`` layers are dropped: the reference runs them only on its
    streaming (``feat_cache``) path, never on the single-image one, so they carry
    no information in the 2D fold (the loader drops their weights).
    """

    def __init__(self, dim: int, upsample: bool) -> None:
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        else:
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            # nearest-exact + 3x3 conv: one halo row keeps the strips exact
            return strip_apply(self.resample, x, scale=2)
        return self.resample(x)


class Wan22AvgDown(nn.Module):
    """One-frame fold of the v2_2 ``AvgDown3D`` shortcut: the mean over each
    channel group, zero-padded time slot at the front of each group."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int = 1, factor_s: int = 2) -> None:
        super().__init__()
        assert in_channels * factor_t * factor_s * factor_s % out_channels == 0
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.group_size = in_channels * factor_t * factor_s * factor_s // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        # (b, c, r, s, h/fs, w/fs) -> (b, c, fs*fs, h/fs, w/fs); exact for fs=1 too
        block = x.view(b, c, h // fs, fs, w // fs, fs).permute(0, 1, 3, 5, 2, 4)
        block = block.reshape(b, c, fs * fs, h // fs, w // fs)
        if ft > 1:  # the reference zero-pads the time axis at the front
            block = torch.cat([block.new_zeros(b, c, ft - 1, fs * fs, h // fs, w // fs), block.unsqueeze(2)], dim=2)
        return block.view(b, self.out_channels, self.group_size, h // fs, w // fs).mean(dim=2)


class Wan22DupUp(nn.Module):
    """One-frame fold of the v2_2 ``DupUp3D`` shortcut: ``repeat_interleave``
    over the (t, r, s) channel cells, keeping only the last temporal cell."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int = 1, factor_s: int = 2) -> None:
        super().__init__()
        assert out_channels * factor_t * factor_s * factor_s % in_channels == 0
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.repeats = out_channels * factor_t * factor_s * factor_s // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(b, self.out_channels, ft, fs, fs, h, w)
        x = x[:, :, ft - 1, :, :, :, :]  # the image path keeps the last temporal frame
        x = x.permute(0, 1, 4, 2, 5, 3)
        return x.reshape(b, self.out_channels, h * fs, w * fs)


class Wan22DownBlock(nn.Module):
    """v2_2 ``Down_ResidualBlock``: residual stack + resample, plus the
    parameter-free ``avg_shortcut`` (no checkpoint keys of its own)."""

    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temperal_downsample: bool, downsample: bool) -> None:
        super().__init__()
        self.avg_shortcut = Wan22AvgDown(
            in_dim, out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if downsample else 1,
        )
        layers: list[nn.Module] = []
        in_c = in_dim
        for _ in range(num_res_blocks):
            layers.append(Wan22ResidualBlock(in_c, out_dim))
            in_c = out_dim
        if downsample:
            layers.append(Wan22Resample(out_dim, upsample=False))
        self.downsamples = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.avg_shortcut(x)
        for layer in self.downsamples:
            x = layer(x)
        return x + h


class Wan22UpBlock(nn.Module):
    """v2_2 ``Up_ResidualBlock``: residual stack + resample, plus the
    parameter-free ``avg_shortcut`` (no checkpoint keys of its own)."""

    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temperal_upsample: bool, upsample: bool) -> None:
        super().__init__()
        self.avg_shortcut = (
            Wan22DupUp(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2)
            if upsample
            else None
        )
        layers: list[nn.Module] = []
        in_c = in_dim
        for _ in range(num_res_blocks + 1):
            layers.append(Wan22ResidualBlock(in_c, out_dim))
            in_c = out_dim
        if upsample:
            layers.append(Wan22Resample(out_dim, upsample=True))
        self.upsamples = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.upsamples:
            h = layer(h)
        if self.avg_shortcut is not None:
            # add the shortcut straight into the block's own buffer rather than
            # through a second full-size temporary
            h = strip_apply(self.avg_shortcut, x, scale=self.avg_shortcut.factor_s, halo=0, out=h)
        return h


def _middle(dim: int) -> nn.Sequential:
    """resnet -> attention -> resnet, shared verbatim by encoder and decoder."""
    return nn.Sequential(
        Wan22ResidualBlock(dim, dim),
        Wan22AttentionBlock(dim),
        Wan22ResidualBlock(dim, dim),
    )


def _head(dim: int, out_dim: int) -> nn.Sequential:
    """RMS -> SiLU -> 3x3 conv, shared verbatim by encoder and decoder."""
    return nn.Sequential(Wan22RMSNorm(dim), nn.SiLU(), nn.Conv2d(dim, out_dim, 3, padding=1))


class Wan22Encoder(nn.Module):
    """2D encoder (v2_2 ``Encoder3d`` without the video cache path)."""

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        temperal_downsample: list[bool],
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [1, *dim_mult]]
        self.conv1 = nn.Conv2d(in_channels, dims[0], 3, padding=1)
        self.downsamples = nn.Sequential(
            *[
                Wan22DownBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    temperal_downsample[i] if i < len(temperal_downsample) else False,
                    downsample=i != len(dim_mult) - 1,
                )
                for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )
        self.middle = _middle(dims[-1])
        self.head = _head(dims[-1], z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class Wan22Decoder(nn.Module):
    """2D decoder (v2_2 ``Decoder3d`` without the video cache path); the
    ``temperal_upsample`` flags are the reverse of the encoder's."""

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: list[int],
        num_res_blocks: int,
        out_channels: int,
        temperal_upsample: list[bool],
    ) -> None:
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]
        self.conv1 = nn.Conv2d(z_dim, dims[0], 3, padding=1)
        self.middle = _middle(dims[0])
        self.upsamples = nn.Sequential(
            *[
                Wan22UpBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    temperal_upsample[i] if i < len(temperal_upsample) else False,
                    upsample=i != len(dim_mult) - 1,
                )
                for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )
        self.head = _head(dims[-1], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.upsamples:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Latent normalisation (latents are stored as ``(mu - mean) / std`` and the
# inverse is applied before decoding). Keyed by ``z_dim``, which is what the
# loader has to go on when it picks them for a checkpoint.
# ---------------------------------------------------------------------------

# Official Wan 2.2 (48ch), from the release's ``Wan2_2_VAE`` wrapper.
WAN22_LATENTS_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]

WAN22_LATENTS_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]

# Qwen-Image 2.1 (64ch), from the same source ComfyUI ships as its latent format.
QWEN_IMAGE21_LATENTS_MEAN = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
    1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
    -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
    0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
]

QWEN_IMAGE21_LATENTS_STD = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
    3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
    3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
    3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
    3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
]

#: z_dim -> (mean, std). A new checkpoint of this family is one more entry.
LATENT_STATS: dict[int, tuple[list[float], list[float]]] = {
    48: (WAN22_LATENTS_MEAN, WAN22_LATENTS_STD),
    64: (QWEN_IMAGE21_LATENTS_MEAN, QWEN_IMAGE21_LATENTS_STD),
}

#: Every variant of this family compresses 16x (patchify x 2 per downsample stage).
PIXEL_COMPRESSION = 16


class AutoencoderKLWan22(nn.Module):
    """Wan 2.2 family VAE for still images.

    Defaults match the released Wan 2.2 weights (dim=160, dec_dim=256, z_dim=48,
    2x2 patchify, RGB); the Qwen-Image 2.1 VAE is the same module with
    ``z_dim=64``, five ``dim_mult`` stages, ``patch_size=1`` and
    ``image_channels=4`` (RGBA). The per-stage temporal flags only pick the
    channel grouping of the parameter-free shortcuts.

    ``image_channels`` is the VAE's own pixel width and part of the interface:
    the encode/decode boundary pads an opaque alpha onto RGB input when the VAE
    wants four channels, and decoding hands back ``image_channels`` channels.
    """

    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 48,
        dim_mult: list[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: list[bool] = (False, True, True),
        image_channels: int = 3,
        patch_size: int = 2,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
        pad_channel_value: float = 1.0,
    ) -> None:
        super().__init__()
        dim_mult = list(dim_mult)
        if (latents_mean is None) != (latents_std is None):
            raise ValueError("pass both latents_mean and latents_std, or neither")
        if latents_mean is None:
            if z_dim not in LATENT_STATS:
                raise ValueError(
                    f"no latent statistics for z_dim={z_dim}; pass latents_mean/latents_std "
                    f"(known: {sorted(LATENT_STATS)})"
                )
            latents_mean, latents_std = LATENT_STATS[z_dim]
        assert len(latents_mean) == z_dim and len(latents_std) == z_dim

        self.dim_mult = dim_mult
        self.z_dim = z_dim
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.pad_channel_value = pad_channel_value
        # Hoisted buffers so .to(device) moves them with the module.
        self.register_buffer("_latents_mean", torch.tensor(latents_mean).view(1, z_dim, 1, 1), persistent=False)
        self.register_buffer("_latents_inv_std", (1.0 / torch.tensor(latents_std)).view(1, z_dim, 1, 1), persistent=False)

        self.encoder = Wan22Encoder(
            dim, z_dim * 2, dim_mult, num_res_blocks,
            image_channels * patch_size * patch_size, temperal_downsample,
        )
        self.conv1 = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = Wan22Decoder(
            dec_dim, z_dim, dim_mult, num_res_blocks,
            image_channels * patch_size * patch_size, temperal_downsample[::-1],
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def pixel_channels(self) -> int:
        """Pixel channels the VAE consumes/emits (3 = RGB, 4 = RGBA)."""
        return self.image_channels

    @property
    def spatial_compression(self) -> int:
        """Spatial compression: patchify x 2^(stages-1), e.g. 2 x 8 = 16."""
        return self.patch_size * 2 ** (len(self.dim_mult) - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pixels in [-1, 1] (the VAE's own channel count) -> unnormalised means."""
        x = patchify(x, self.patch_size)
        return self.conv1(self.encoder(x)).chunk(2, dim=1)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latents -> unclamped pixels in ~[-1, 1] (``spatial_compression`` up)."""
        x = self.decoder(self.conv2(z))
        return unpatchify(x, self.patch_size)

    def _match_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Fit a pixel tensor to the VAE's channel count.

        RGB into an RGBA VAE gets an opaque alpha (``pad_channel_value``); more
        channels than the VAE wants are truncated to RGB. Anything else is a
        caller error rather than a silent guess.
        """
        want = self.pixel_channels
        got = x.shape[1]
        if got == want:
            return x
        if got == 3 and want == 4:
            return F.pad(x, (0, 0, 0, 0, 0, 1), value=self.pad_channel_value)
        if got > 3 and want == 3:
            return x[:, :3]
        raise ValueError(f"expected {want} pixel channels for this VAE, got {got}")

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels in [-1, 1] (3 or ``pixel_channels``) -> normalised latents."""
        latents = self.encode(self._match_pixels(pixels.to(self.device, self.dtype)))
        mean = self._latents_mean.to(latents.device, latents.dtype)
        inv_std = self._latents_inv_std.to(latents.device, latents.dtype)
        return (latents - mean) * inv_std

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalised latents -> pixels clamped to [-1, 1] (``pixel_channels``)."""
        latents = latents.to(self.device, self.dtype)
        mean = self._latents_mean.to(latents.device, latents.dtype)
        inv_std = self._latents_inv_std.to(latents.device, latents.dtype)
        return self.decode(latents / inv_std + mean).clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- the loader


def _infer_arch(state_dict: dict, vae_path: str) -> dict:
    """Recover the architecture knobs a Wan 2.2 family checkpoint was built with.

    Only the names and shapes are read. The two shipped variants differ in width,
    depth (three vs four downsample stages), latent width and pixel channels, and
    all of it is recoverable: stage widths from the residual blocks, the number of
    residual blocks by counting them, the pixel patch from the 16x compression the
    family always delivers (``patch_size x 2^stages``), the per-stage temporal
    flags from the video-only ``time_conv`` layers the image path never runs.
    """
    enc_conv = state_dict["encoder.conv1.weight"]
    if enc_conv.dim() != 5:
        raise ValueError(f"{vae_path}: expected the 5D video layout, got {tuple(enc_conv.shape)}")
    dim = enc_conv.shape[0]
    dec_dim = state_dict["decoder.head.0.gamma"].shape[0]
    z_dim = state_dict["conv2.weight"].shape[0]

    # Residual blocks per stage: the reference puts the same count in every stage.
    num_res_blocks = 0
    while f"encoder.downsamples.0.downsamples.{num_res_blocks}.residual.2.weight" in state_dict:
        num_res_blocks += 1
    if num_res_blocks == 0:
        raise ValueError(f"'encoder.downsamples.0.downsamples.0.residual.2.weight' not found in {vae_path}")

    # A stage's last residual block ends on that stage's output width.
    widths = []
    i = 0
    while f"encoder.downsamples.{i}.downsamples.{num_res_blocks - 1}.residual.2.weight" in state_dict:
        w = state_dict[f"encoder.downsamples.{i}.downsamples.{num_res_blocks - 1}.residual.2.weight"].shape[0]
        if w % dim:
            raise ValueError(f"stage width {w} is not a multiple of dim {dim} in {vae_path}")
        widths.append(w // dim)
        i += 1
    if not widths:
        raise ValueError(f"'encoder.downsamples.0.?' not found in {vae_path} (not a Wan 2.2 VAE?)")
    dim_mult = widths

    # The family compresses 16x, so whatever the stages do not cover is patchify.
    stages = len(dim_mult) - 1
    if PIXEL_COMPRESSION % 2**stages:
        raise ValueError(f"{vae_path}: {stages} downsample stages cannot make a 16x compression")
    patch_size = PIXEL_COMPRESSION // 2**stages

    in_channels = enc_conv.shape[1]
    out_channels = state_dict["decoder.head.2.weight"].shape[0]
    if in_channels != out_channels or in_channels % (patch_size * patch_size):
        raise ValueError(
            f"unexpected in/out channels {in_channels}/{out_channels} with patch_size={patch_size} in {vae_path}"
        )
    image_channels = in_channels // (patch_size * patch_size)

    # Per-stage temporal flags: recorded by each resample's video-only time_conv.
    enc_flags = [
        f"encoder.downsamples.{i}.downsamples.{num_res_blocks}.time_conv.weight" in state_dict
        for i in range(stages)
    ]
    dec_flags = [
        f"decoder.upsamples.{i}.upsamples.{num_res_blocks + 1}.time_conv.weight" in state_dict
        for i in range(stages)
    ]
    if dec_flags != enc_flags[::-1]:
        raise ValueError(f"inconsistent temporal flags in {vae_path}: encoder {enc_flags}, decoder {dec_flags}")

    if state_dict["encoder.head.0.gamma"].shape[0] != dim * dim_mult[-1]:
        raise ValueError(
            f"encoder width {state_dict['encoder.head.0.gamma'].shape[0]} != dim*mult[-1] {dim * dim_mult[-1]}"
        )
    if state_dict["decoder.conv1.weight"].shape[0] != dec_dim * dim_mult[-1]:
        raise ValueError(
            f"decoder width {state_dict['decoder.conv1.weight'].shape[0]} != dec_dim*mult[-1] {dec_dim * dim_mult[-1]}"
        )
    if tuple(state_dict["conv1.weight"].shape) != (z_dim * 2, z_dim * 2, 1, 1, 1):
        raise ValueError(f"unexpected conv1 shape {tuple(state_dict['conv1.weight'].shape)}")

    return {
        "dim": dim,
        "dec_dim": dec_dim,
        "z_dim": z_dim,
        "dim_mult": dim_mult,
        "num_res_blocks": num_res_blocks,
        "temperal_downsample": enc_flags,
        "image_channels": image_channels,
        "patch_size": patch_size,
    }


def load_wan22_vae(
    vae_path: str,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    latents_mean: list[float] | None = None,
    latents_std: list[float] | None = None,
) -> AutoencoderKLWan22:
    """Load a Wan 2.2 family VAE (ComfyUI ``vae2_2`` layout) for single-frame use.

    The architecture is inferred from the checkpoint (see :func:`_infer_arch`) and
    the latent statistics are picked from its ``z_dim`` unless passed explicitly,
    so both the 48ch Wan 2.2 and the 64ch RGBA Qwen-Image 2.1 file load through
    here. The video weights are then collapsed to 2D: drop ``time_conv``, keep the
    last time slice of 5D convs (causal padding puts the single frame at the end),
    squeeze 4D gammas.
    """
    logger.info("Loading Wan 2.2 family VAE from %s", vae_path)
    state_dict = load_safetensors(vae_path, device=device)

    required = (
        "encoder.conv1.weight",
        "decoder.conv1.weight",
        "conv1.weight",
        "conv2.weight",
        "encoder.head.0.gamma",
        "encoder.head.2.weight",
        "decoder.middle.0.residual.0.gamma",
        "decoder.head.0.gamma",
        "decoder.head.2.weight",
    )
    for key in required:
        if key not in state_dict:
            raise ValueError(f"'{key}' not found in {vae_path} (not a Wan 2.2 VAE?)")

    arch = _infer_arch(state_dict, vae_path)

    vae = AutoencoderKLWan22(
        latents_mean=latents_mean, latents_std=latents_std, **arch
    )

    # Collapse the video layout to 2D (see docstring).
    state_dict = {k: v for k, v in state_dict.items() if ".time_conv." not in k}
    for key, val in state_dict.items():
        if val.dim() == 5:
            state_dict[key] = val[:, :, -1]
        elif key.endswith(".gamma") and val.dim() == 4:
            state_dict[key] = val.reshape(val.shape[0], 1, 1)

    vae.load_state_dict(state_dict, assign=True)
    logger.info(
        "Loaded VAE from %s (%s)",
        vae_path,
        " ".join(f"{k}={v}" for k, v in arch.items()),
    )
    vae.to(device, dtype)
    return vae.eval().requires_grad_(False)
