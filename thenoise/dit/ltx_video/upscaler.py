"""LTX-2.3/2.5 distilled spatial x2 latent upscaler (PyTorch only).

The numeric architecture and original parameter names follow the Apache-2.0
Lightricks/LTX-2 source, v1.6.2:
  packages/ltx-core/src/ltx_core/model/upsampler/{model,res_block,pixel_shuffle}.py
See https://github.com/Lightricks/LTX-2/blob/v1.6.2/LICENSE

Unlike a general LTX upscaler, this implements precisely the shipped 3D,
spatial-only x2 module. Do not add generic resampling concepts unless another
checkpoint requires them. TheNoise's existing streaming loader and MemoryManager
own weights and staging; this file contains numerical layers only.
"""
from __future__ import annotations

import torch
from torch import nn


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class LTXSpatialUpscaler(nn.Module):
    """Native spatial-only LTX x2 module, shape [B,C,T,H,W] -> [B,C,T,2H,2W].

    Attribute names must match original LTX checkpoint tensor names exactly:
    `upsampler.0`, `initial_conv`, `res_blocks`, `post_upsample_res_blocks`,
    `final_conv`. Changes to the naming require a matching checkpoint conversion.
    """

    def __init__(self, in_channels: int = 128, mid_channels: int = 1024,
                 num_blocks_per_stage: int = 4):
        super().__init__()
        if mid_channels % 32:
            raise ValueError("LTX spatial upscaler mid_channels must be divisible by 32")
        self.initial_conv = nn.Conv3d(in_channels, mid_channels, 3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)
        self.initial_activation = nn.SiLU()
        self.res_blocks = nn.ModuleList(
            [_ResBlock(mid_channels) for _ in range(num_blocks_per_stage)]
        )
        # Original checkpoint: upsampler.0 is a *2D* convolution applied per frame.
        # nn.PixelShuffle is parameter-free, preserving original state_dict keys.
        self.upsampler = nn.Sequential(
            nn.Conv2d(mid_channels, 4 * mid_channels, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.post_upsample_res_blocks = nn.ModuleList(
            [_ResBlock(mid_channels) for _ in range(num_blocks_per_stage)]
        )
        self.final_conv = nn.Conv3d(mid_channels, in_channels, 3, padding=1)

    @property
    def dtype(self) -> torch.dtype:
        """Keep the existing LTX adapter's `upscaler.dtype` contract."""
        return self.initial_conv.weight.dtype

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5:
            raise ValueError("LTX spatial upscaler requires [B,C,T,H,W]")
        x = self.initial_activation(self.initial_norm(self.initial_conv(latent)))
        for block in self.res_blocks:
            x = block(x)
        b, c, t, h, w = x.shape
        # Each frame uses the same spatial weights; no temporal interpolation.
        x = self.upsampler(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        x = x.reshape(b, t, c, h * 2, w * 2).permute(0, 2, 1, 3, 4)
        for block in self.post_upsample_res_blocks:
            x = block(x)
        return self.final_conv(x)


__all__ = ["LTXSpatialUpscaler"]
