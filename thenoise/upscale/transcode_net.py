"""The latent-transcode upscale network — vendored subset for thenoise.

The raw network only: it takes the pipeline's latent plus a feature map from the
model's own VAE decoder and returns the latent at 2x the resolution.

Copied and trimmed from https://github.com/LoganBooker/SesquiLSR (MIT).

Original copyright/license notice follows.
"""

# Copyright (c) 2025 LoganBooker. Licensed under the MIT License.
# Source: https://github.com/LoganBooker/SesquiLSR
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm2D(nn.Module):
    """Channel-wise RMS norm with the statistic taken in fp32."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().square().mean(1, keepdim=True) + self.eps)
        return x * scale.to(x.dtype) * self.weight.to(x.dtype).view(1, -1, 1, 1)


class GatedSwiGLUStage(nn.Module):
    """Pre-norm gated stage: pointwise expand, depthwise mix, pointwise down."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = RMSNorm2D(channels)
        self.input = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.spatial = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            padding_mode="reflect",
            groups=channels,
            bias=False,
        )
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.input(F.mish(self.norm(x))).chunk(2, 1)
        value = self.spatial(F.mish(value) * 2.0 * torch.tanh(0.5 * gate))
        return x + self.output(value)


class LatentTranscodeNet(nn.Module):
    """Latent + VAE-decoder feature -> the latent at 2x the resolution.

    The detail path is ``out(body(head(feature)))``; it is added to a bicubic
    upsample of the latent and refined by ``skip``, so the bridge is a residual
    correction on a crude resize.

    ``feature_channels`` is the channel width of the VAE-decoder feature and
    ``latent_channels`` the pipeline's latent width.
    """

    #: Fixed by the two ``F.interpolate`` calls below: 2x or not at all.
    scale = 2

    def __init__(
        self,
        *,
        feature_channels: int = 1152,
        latent_channels: int = 64,
        width: int = 384,
        depth: int = 6
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.feature_channels = feature_channels

        self.head = nn.Conv2d(feature_channels, width, 1)
        self.body = nn.Sequential(*(GatedSwiGLUStage(width) for _ in range(depth)))
        self.out = nn.Conv2d(width, latent_channels, 3, padding=1, padding_mode="reflect")
        self.skip = nn.Conv2d(latent_channels, latent_channels, 3, padding=1, padding_mode="reflect")        

    def forward(self, feature: Tensor, latent: Tensor) -> Tensor:
        """``feature`` (2x the latent grid) + ``latent`` -> 2x latent."""
        size = (2 * latent.shape[-2], 2 * latent.shape[-1])
        if feature.shape[-2:] != size:
            raise ValueError(
                f"decoder-prefix feature must be exactly {self.scale}x the latent "
                f"grid, got {tuple(size)} for a {tuple(latent.shape[-2:])} latent"
            )

        hidden = self.body(self.head(feature))
        skip = F.interpolate(latent, size=size, mode="bicubic", align_corners=False)
        return self.skip(skip + self.out(hidden))


__all__ = ["LatentTranscodeNet", "GatedSwiGLUStage", "RMSNorm2D"]
