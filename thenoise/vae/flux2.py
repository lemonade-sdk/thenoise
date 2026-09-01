# Flux.2 VAE, ported from kohya-ss/musubi-tuner's
# ``flux2_models.AutoEncoder`` (itself a copy of the Black Forest Labs FLUX repo,
# Apache-2.0).
#
# Both the decoder (still-image decode) and the encoder (image -> latent for
# reference-latent editing) are kept: the ``encoder``/``decoder`` modules, the
# ``quant_conv``/``post_quant_conv`` 1x1s, and the BatchNorm ``bn`` whose running
# statistics normalize the packed latent.
#
# Flux.2 latents are *packed*: the AE packs a 32-channel latent 2x2 into 128 channels
# (spatial compression 16 in packed space), then normalizes via BatchNorm. The DiT
# denoises this normalized 128-channel packed latent, so the canonical latent here is
# ``[B, 128, H//16, W//16]``. Decode un-normalizes (``z*sqrt(var) + mean``), unpacks
# to ``[B, 32, H//8, W//8]``, runs the decoder, and clamps to [-1, 1]. Encode is the
# inverse: pixels -> encoder -> quant_conv -> 32ch@8x, then patchifies to 128ch@16x
# and normalizes via BatchNorm.

# Copyright 2023 The HuggingFace Team. Licensed under the Apache-2.0 License.
# Copyright 2024 Black Forest Labs. Flux is released under the Apache-2.0 License.
from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.utils.safetensors import load_safetensors
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _patchify(z: torch.Tensor) -> torch.Tensor:
    """32ch -> 128ch via 2x2 spatial packing (inverse of ``_unpatchify``)."""
    B, C, H, W = z.shape
    return (
        z.float()
        .reshape(B, C, H // 2, 2, W // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(B, C * 4, H // 2, W // 2)
    )


def _unpatchify(z: torch.Tensor) -> torch.Tensor:
    """128ch -> 32ch via 2x2 spatial unpacking."""
    B, C, H, W = z.shape
    return (
        z.float()
        .reshape(B, C // 4, 2, 2, H, W)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(B, C // 4, H * 2, W * 2)
    )


class _AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.view(b, 1, h * w, c).transpose(2, 3)
        k = k.view(b, 1, h * w, c).transpose(2, 3)
        v = v.view(b, 1, h * w, c).transpose(2, 3)
        h_ = F.scaled_dot_product_attention(q, k, v)
        return h_.transpose(2, 3).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(x))


class _ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class _Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class _Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # No asymmetric padding in torch conv, so pad manually then stride-2 conv.
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class _Encoder(nn.Module):
    """Flux.2 encoder: pixels -> 32ch raw latent (8x spatial down), then quant_conv.

    Exact mirror of ``_Decoder`` (ch 128, ch_mult [1,2,4,4], 2 res-blocks per level,
    mid attention only), with a ``quant_conv`` 1x1 reducing the 2*z_channels output
    to ``z_channels``. Key naming matches the official ``flux2-vae.safetensors``
    (``encoder.*``, including ``encoder.quant_conv.*``).
    """

    def __init__(
        self,
        ch: int = 128,
        in_channels: int = 3,
        ch_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        z_channels: int = 32,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.z_channels = z_channels

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = None
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(_ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn  # no attention in the down path (only mid), matches decoder
            if i_level != self.num_resolutions - 1:
                down.downsample = _Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)
        # quant_conv keeps the 2*z_channels (mean+logvar) output; the gaussian
        # mean is taken by the caller (``encode_pixels_to_latents``) in eval mode.
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, kernel_size=1)

    @property
    def dtype(self):
        return next(self.down.parameters()).dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)  # [B, 2*z_channels, H//8, W//8]
        return self.quant_conv(h)  # [B, 2*z_channels, H//8, W//8]


class _Decoder(nn.Module):
    """Flux.2 decoder: z -> pixels (32ch raw -> 3ch, 8x spatial up)."""

    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        z_channels: int = 32,
    ):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = z_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(_ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in)
            self.up.insert(0, up)  # prepend to keep consistent order

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    @property
    def dtype(self):
        return next(self.up.parameters()).dtype

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = h.to(upscale_dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        return self.conv_out(h)


class AutoencoderKLFlux2(nn.Module):
    """Flux.2 VAE (encoder + decoder), with musubi/Flux key naming.

    Accepts/produces the canonical packed latent ``[B, 128, H//16, W//16]``
    (normalized). ``decode_to_pixels`` returns pixels in [-1, 1];
    ``encode_pixels_to_latents`` takes pixels in [-1, 1].
    """

    z_dim = 128  # packed latent channels (the canonical Flux.2 latent)
    spatial_compression = 16  # pixel / packed-latent ratio
    bn_eps = 1e-4

    def __init__(self, channels: int = 128):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, eps=self.bn_eps, momentum=0.1, affine=False, track_running_stats=True)
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    @property
    def dtype(self):
        return next(self.decoder.parameters()).dtype

    @property
    def device(self):
        return next(self.decoder.parameters()).device

    def inv_normalize(self, z: torch.Tensor) -> torch.Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Raw packed latent -> normalized canonical latent (``(z - mean) / sqrt(var)``)."""
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return (z.float() - m) / s

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels (``[B, C, H, W]`` in [-1, 1]) -> canonical packed latent.

        Pixels -> encoder -> 2*z_channels (mean+logvar) -> gaussian mean ->
        32ch@8x raw latent -> patchify to 128ch@16x -> BatchNorm normalize.
        The inverse of ``decode_to_pixels`` (plus the eval-mode gaussian mean).
        The input is cast to the encoder's dtype (the model runs in bf16).
        """
        z = self.encoder(pixels.to(device=self.device, dtype=self.encoder.dtype))  # [B, 2*z_channels, H//8, W//8]
        z = z.chunk(2, dim=1)[0]  # gaussian mean (eval mode) -> [B, z_channels, ...]
        z = _patchify(z)  # [B, 32, H//8, W//8] -> [B, 128, H//16, W//16]
        return self.normalize(z)

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the canonical packed latent to pixels in [-1, 1]."""
        z = self.inv_normalize(latents.float())  # undo BN -> raw packed 128ch
        z = _unpatchify(z).to(self.decoder.dtype)  # [B, 32, H//8, W//8]
        image = self.decoder(z)
        return image.clamp(-1.0, 1.0)


def load_flux2_vae(
    vae_path: str,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> AutoencoderKLFlux2:
    """Load the Flux.2 VAE weights from ``vae_path`` (e.g. ae.safetensors).

    Keeps the encoder (+ its ``quant_conv``), decoder (+ ``post_quant_conv``) and
    BatchNorm keys; the official file's keys carry the matching ``encoder.`` /
    ``decoder.`` / ``bn.`` prefixes directly.
    """
    device = torch.device(device)
    logger.info("Loading Flux.2 VAE from %s", vae_path)
    state_dict = load_safetensors(vae_path, device=device, disable_mmap=disable_mmap)

    vae = AutoencoderKLFlux2()
    keep = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("encoder.") or k.startswith("decoder.") or k.startswith("bn.")
    }
    if not keep:
        raise ValueError(f"No 'encoder.*'/'decoder.*'/'bn.*' keys found in {vae_path} (not a Flux.2 VAE?)")
    info = vae.load_state_dict(keep, strict=True, assign=True)
    logger.info("Loaded Flux.2 VAE: %s", info)

    vae.to(device)
    if dtype is not None:
        vae.to(dtype)
    return vae.eval().requires_grad_(False)


__all__ = ["AutoencoderKLFlux2", "load_flux2_vae"]
