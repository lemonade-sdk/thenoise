"""The latent-transcode upscale network — vendored subset for thenoise.

Copied and trimmed from https://github.com/LoganBooker/SesquiLSR (MIT).

This is the raw network only. It takes the pipeline's latent plus a feature map
from the model's own VAE decoder and returns the latent at 2x the resolution.

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
    """Channel-wise RMS norm with the statistic taken in fp32.

    The mean square is computed on an upcast and the scale cast back, so a bf16
    feature does not lose its normalisation to an 8-bit mantissa — the same reason
    the engine's VAE latent arithmetic runs in fp32.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().square().mean(1, keepdim=True) + self.eps)
        return x * scale.to(x.dtype) * self.weight.to(x.dtype).view(1, -1, 1, 1)


class GatedSwiGLUStage(nn.Module):
    """One pre-norm gated stage: pointwise expand, depthwise mix, pointwise down.

    ``shortcut`` exists only where the stage changes width, which is exactly the
    first stage of the trunk — so most stages carry four parameter tensors and the
    entry stage five, and ``load_state_dict`` therefore pins the widths too.
    """

    def __init__(self, in_channels: int, out_channels: int, *, expansion: float = 1.0):
        super().__init__()
        hidden = max(1, round(out_channels * expansion))
        self.norm = RMSNorm2D(in_channels)
        self.input = nn.Conv2d(in_channels, 2 * hidden, 1, bias=False)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.output = nn.Conv2d(hidden, out_channels, 1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.input(F.mish(self.norm(x))).chunk(2, 1)
        return self.shortcut(x) + self.output(self.spatial(F.mish(value) * gate))


class LatentTranscodeNet(nn.Module):
    """Latent + VAE-decoder feature -> the latent at 2x the resolution.

    Two paths are summed on the way out:

      * ``out(body(head(feature) + trunk))`` — the detail. The decoder feature is
        projected to the trunk width and added to the trunk's own upsampled latent,
        so the bridge corrects the crude interpolation instead of inventing the
        image from scratch;
      * ``skip(interpolate(latent))`` — a 3x3 correction on the bicubic upsample,
        which is what makes a zeroed trunk degrade into a plain resize rather than
        into noise.

    ``feature_channels`` is the channel width of the VAE-decoder feature and
    ``latent_channels`` the pipeline's latent width; for Qwen-Image 2.1 they are
    the decoder's first upsample stage (``dec_dim * dim_mult[-1]``) and its
    ``z_dim``. ``depth``/``expansion``/``width`` are the trunk's.
    """

    #: Fixed by the two ``F.interpolate`` calls below: 2x or not at all.
    scale = 2

    def __init__(
        self,
        *,
        feature_channels: int = 1152,
        latent_channels: int = 64,
        width: int = 384,
        depth: int = 6,
        expansion: float = 1.0,
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.latent_channels = latent_channels

        # ``head``'s input width is the VAE side of the contract; everything else
        # is internal to the bridge.
        self.head = nn.Conv2d(feature_channels, width, 1)

        # The trunk that carries the *latent*, run at the source resolution and
        # upsampled afterwards: cheaper than doing it at 4x the pixel count.
        self.in_body = nn.Sequential(*(
            GatedSwiGLUStage(
                latent_channels if i == 0 else width, width, expansion=expansion
            )
            for i in range(depth)
        ))
        # The trunk that sees the fused feature, run at the target resolution.
        self.body = nn.Sequential(*(
            GatedSwiGLUStage(width, width, expansion=expansion) for _ in range(depth)
        ))
        self.out = nn.Conv2d(width, latent_channels, 3, padding=1)
        self.skip = nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False)

    def forward(self, feature: Tensor, latent: Tensor) -> Tensor:
        """``feature`` (2x the latent grid) + ``latent`` -> 2x latent."""
        if latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected a {self.latent_channels}-channel latent, got {latent.shape[1]}"
            )
        size = feature.shape[-2:]
        if size != (self.scale * latent.shape[-2], self.scale * latent.shape[-1]):
            raise ValueError(
                f"decoder-prefix feature must be exactly {self.scale}x the latent "
                f"grid, got {tuple(size)} for a {tuple(latent.shape[-2:])} latent"
            )

        trunk = F.interpolate(
            self.in_body(latent), size=size, mode="bicubic", align_corners=False
        )
        hidden = self.body(self.head(feature) + trunk)
        skip = F.interpolate(latent, size=size, mode="bicubic", align_corners=False)
        return self.skip(skip) + self.out(hidden)


__all__ = ["LatentTranscodeNet", "GatedSwiGLUStage", "RMSNorm2D"]
