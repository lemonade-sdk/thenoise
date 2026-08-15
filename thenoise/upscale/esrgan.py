"""Real-ESRGAN (RRDBNet x4) pixel super-resolution, vendored for thenoise.

ESRGAN works on *decoded pixels* (fp32, RGB) after the VAE decode and
complements the latent (Sesqui) upscaler: the latent path supplies a 2x
multiplier in ``refined`` mode, ESRGAN adds another 4x on top. It is fast, so it
is *not* pipeline-cached (only the decoded VAE output is cached).

Like the reference inference, the model operates on RGB in **[0, 1]** and emits
[0, 1]; the caller converts between the pipeline's [-1, 1] pixel range.

The architecture is adapted from the original Real-ESRGAN (BSD-3-Clause) and
loads the ComfyUI repackaged weights directly (``Comfy-Org/Real-ESRGAN_repackaged`` /
``RealESRGAN_x4plus.safetensors``). The repackage keeps ComfyUI's key naming
(``body.N.rdbN.convN.*``, ``conv_first``, ``conv_body``, ``conv_up1``,
``conv_up2``, ``conv_hr``, ``conv_last``) over the original RRDBNet structure,
so the state dict loads with no key remapping. The forward is faithful to the
original, including the crucial trunk residual skip ``conv_first(x) +
conv_body(body(conv_first(x)))`` and nearest-neighbour upsampling between the
``conv_up`` stages.

Weights are kept in fp32: bf16's ~7-bit mantissa degrades super-resolution
detail, and the model is small enough that fp32 is cheap. The upscale scale
(2 or 4) is auto-detected from ``conv_first``'s input channels via
``detect_esrgan_scale``: scale-2 models pixel-unshuffle the input by 2 (so
``conv_first`` takes 12 channels) and use two 2x upsample stages for a net 2x;
scale-4 models take the 3 RGB channels directly and use two 2x stages for 4x.

Original copyright/license notice follows.
"""

# Copyright (c) 2021 xinntao. Licensed under the BSD-3-Clause License.
# Source: https://github.com/xinntao/Real-ESRGAN
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualDenseBlock(nn.Module):
    """``ResidualDenseBlock_4C``: 5-conv dense residual block.

    Grows the channel count toward ``gc`` per layer via feature concat, then
    collapses back to ``nf``; 0.2 local residual.
    """

    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 dense blocks + 0.2 global residual."""

    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x: Tensor) -> Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def _pixel_unshuffle(x: Tensor, scale: int) -> Tensor:
    """Inverse of pixel-shuffle: reduce spatial size, enlarge channels."""
    b, c, hh, hw = x.shape
    out_channel = c * (scale**2)
    y = x.reshape(b, c, hh // scale, scale, hw // scale, scale)
    y = y.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, hh // scale, hw // scale)
    return y


class RRDBNet(nn.Module):
    """Real-ESRGAN generator (BasicSR RRDBNet, ComfyUI repackaged weights).

    Faithful to the original: the trunk output is added back to the
    ``conv_first`` feature map (``feat + conv_body(body(feat))``) before the
    nearest-neighbour upsample stages. Omitting that residual skip ruins the
    image (ghost/halo artifacts). Scale is 2 or 4: scale-2 models pixel-unshuffle
    the input by 2 (``conv_first`` takes 12 channels) and use two 2x upsample
    stages for a net 2x; scale-4 models take the 3 RGB channels and use two 2x
    stages for 4x. Weights: ``RealESRGAN_x2plus/x4plus.safetensors``
    (num_feat 64, num_block 23, num_grow_ch 32).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"unsupported ESRGAN scale: {scale} (use 2 or 4)")
        self.scale = scale
        # Scale-2 models pixel-unshuffle the input by 2 (BasicSR), so
        # ``conv_first`` takes 4x the input channels there; scale-4 models take
        # the 3 RGB channels directly.
        in_ch = num_in_ch * 4 if scale == 2 else num_in_ch
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # Both scales use two 2x upsample stages (net 2x / 4x).
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        feat = _pixel_unshuffle(x, 2) if self.scale == 2 else x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))  # trunk residual skip
        # One 2x nearest + conv stage per upsampling level.
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

    @torch.no_grad()
    def forward_tiled(
        self, img: Tensor, tile_size: int = 512, tile_pad: int = 48
    ) -> Tensor:
        """Tiled forward to bound peak activation memory on large inputs.

        Each tile is padded by ``tile_pad`` for context; only the central
        ``tile_size`` region of each tile's output is kept, so tiles stitch
        seamlessly (no overlap blending). RRDB has a large receptive field, so a
        generous pad (48) is used to keep seams below visual threshold; ``img``
        smaller than ``tile_size`` runs as a single exact tile.
        Returns ``[B, C, H*s, W*s]``.
        """
        scale = self.scale
        b, c, h, w = img.shape
        out_h, out_w = h * scale, w * scale
        out = torch.zeros((b, c, out_h, out_w), device=img.device, dtype=img.dtype)

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y0 = max(0, y - tile_pad)
                y1 = min(h, y + tile_size + tile_pad)
                x0 = max(0, x - tile_pad)
                x1 = min(w, x + tile_size + tile_pad)
                tile = img[:, :, y0:y1, x0:x1]
                ot = self(tile)

                # Central authoritative region (input [y, y+ts] x [x, x+ts]).
                oy0 = (y - y0) * scale
                oy1 = oy0 + (min(h, y + tile_size) - y) * scale
                ox0 = (x - x0) * scale
                ox1 = ox0 + (min(w, x + tile_size) - x) * scale
                out[:, :, y * scale : y * scale + (oy1 - oy0), x * scale : x * scale + (ox1 - ox0)] = (
                    ot[:, :, oy0:oy1, ox0:ox1]
                )
        return out


def detect_esrgan_scale(path: str) -> int:
    """Detect an ESRGAN model's upscale scale (2 or 4) from its safetensors header.

    Reads only the header (no weight tensors are loaded). The scale is
    determined by ``conv_first``'s input channels: scale-2 models pixel-unshuffle
    the input by 2, so ``conv_first`` takes 3*4 = 12 channels; scale-4 models
    take the 3 RGB channels directly.
    """
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        in_ch = f.get_slice("conv_first.weight").get_shape()[1]
    if in_ch == 12:
        return 2
    if in_ch == 3:
        return 4
    raise ValueError(
        f"could not detect ESRGAN scale from {path}: conv_first takes "
        f"{in_ch} input channels (expected 12 for 2x or 3 for 4x)"
    )


def load_esrgan(path: str, device: str = "cuda") -> tuple[RRDBNet, int]:
    """Load a Real-ESRGAN model from a safetensors (ComfyUI repackaged keys).

    The upscale scale (2 or 4) is auto-detected from the state dict. Returns
    ``(model, scale)``. Weights are kept in fp32 for super-resolution quality.
    """
    from safetensors.torch import load_file

    state_dict = load_file(path, device="cpu")
    scale = detect_esrgan_scale(path)
    model = RRDBNet(scale=scale)
    model.load_state_dict(state_dict)
    model.to(device=device).eval().requires_grad_(False)
    return model, scale


__all__ = ["RRDBNet", "load_esrgan", "detect_esrgan_scale"]
