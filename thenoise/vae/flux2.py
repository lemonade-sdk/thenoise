# Flux.2 VAE decoder (decode-only), ported from kohya-ss/musubi-tuner's
# ``flux2_models.AutoEncoder`` (itself a copy of the Black Forest Labs FLUX repo,
# Apache-2.0).
#
# Only still-image DECODE is supported here (this engine generates images; it never
# encodes pixels), so only the decoder side of the Flux.2 AutoEncoder is kept: the
# ``decoder`` module (post_quant_conv + up-blocks) and the BatchNorm ``bn`` whose
# running statistics normalize the packed latent. The encoder/quant_conv are omitted.
#
# Flux.2 latents are *packed*: the AE packs a 32-channel latent 2x2 into 128 channels
# (spatial compression 16 in packed space), then normalizes via BatchNorm. The DiT
# denoises this normalized 128-channel packed latent, so the canonical latent here is
# ``[B, 128, H//16, W//16]``. Decode un-normalizes (``z*sqrt(var) + mean``), unpacks
# to ``[B, 32, H//8, W//8]``, runs the decoder, and clamps to [-1, 1].

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
    """Flux.2 VAE decoder (decode-only), with musubi/Flux key naming.

    Accepts the canonical packed latent ``[B, 128, H//16, W//16]`` (normalized) and
    returns pixels ``[B, 3, H, W]`` clamped to [-1, 1].
    """

    z_dim = 128  # packed latent channels (the canonical Flux.2 latent)
    spatial_compression = 16  # pixel / packed-latent ratio
    bn_eps = 1e-4

    def __init__(self, channels: int = 128):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, eps=self.bn_eps, momentum=0.1, affine=False, track_running_stats=True)
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

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the canonical packed latent to pixels in [-1, 1]."""
        z = self.inv_normalize(latents.float())  # undo BN -> raw packed 128ch
        z = _unpatchify(z).to(self.decoder.dtype)  # [B, 32, H//8, W//8]
        image = self.decoder(z)
        return image.clamp(-1.0, 1.0)


def load_flux2_vae(
    vae_path: str,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> AutoencoderKLFlux2:
    """Load the Flux.2 VAE decoder weights from ``vae_path`` (e.g. ae.safetensors).

    Only ``decoder.*`` and ``bn.*`` keys are kept; the encoder/quant_conv keys
    present in the file are ignored.
    """
    device = torch.device(device)
    logger.info("Loading Flux.2 VAE from %s", vae_path)
    state_dict = load_safetensors(vae_path, device=device, disable_mmap=disable_mmap)

    vae = AutoencoderKLFlux2()
    # Keep only the decoder side + the BatchNorm (whose running stats the decode
    # needs). The keys already carry the ``decoder.``/``bn.`` prefixes that match
    # this module's own submodule names, so load them directly.
    keep = {k: v for k, v in state_dict.items() if k.startswith("decoder.") or k.startswith("bn.")}
    if not keep:
        raise ValueError(f"No 'decoder.*'/'bn.*' keys found in {vae_path} (not a Flux.2 VAE?)")
    info = vae.load_state_dict(keep, strict=True, assign=True)
    logger.info("Loaded Flux.2 VAE: %s", info)

    vae.to(device)
    if dtype is not None:
        vae.to(dtype)
    return vae.eval().requires_grad_(False)


__all__ = ["AutoencoderKLFlux2", "load_flux2_vae"]
