"""Leaves the Wan codec family otherwise builds in more than one place.

One family, three homes: ``wan22.py`` (2D, the 2.2 releases), ``qwen_image.py`` (2D,
the 2.1 family, which is the same architecture at one frame) and ``wan_video.py`` (3D,
both). Only math that is *identical* in all three belongs here.

What stays per codec, on purpose:

* the family signature -- ``QwenImageResample`` halves channels on the way up where
  ``Wan22Resample`` keeps them, and the average/duplicate shortcut folds exist only for
  2.2, whose one-frame version lives in ``wan22.py``;
* the temporal causal machinery, which is the 3D codec's whole reason to exist;
* the norms, which are three near-copies that are not the same arithmetic: the 3D one
  normalizes in FP32 and multiplies out-of-place, the 2.2 one normalizes in the input
  dtype and multiplies in place. Merging them moves image pixels in the last bits.
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``[..., H*p, W*p]`` -> ``[..., H, W]``, with each ``p x p`` block folded into channels.

    Row-major within a block and channel-major across the fold, which is the order both
    2D codecs and the 3D one's patch embedding want; a leading time axis rides through
    untouched.
    """
    if patch_size == 1:
        return x
    if x.shape[-1] % patch_size or x.shape[-2] % patch_size:
        raise ValueError(
            f"patchify needs a size divisible by {patch_size},"
            f" got {tuple(x.shape[-2:])}")
    return rearrange(x, "b c ... (h q) (w r) -> b (c r q) ... h w",
                     q=patch_size, r=patch_size)


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Inverse of :func:`patchify`."""
    if patch_size == 1:
        return x
    return rearrange(x, "b (c r q) ... h w -> b c ... (h q) (w r)",
                     q=patch_size, r=patch_size)


def downsample2d(dim: int) -> nn.Sequential:
    """The reference's stride-2 3x3 conv, padded right and bottom only.

    The pad is a slot as much as an operation: ``qwen_image``'s strip walk indexes past
    it to pad for itself, but every codec's checkpoint counts it as a layer.
    """
    return nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)),
                         nn.Conv2d(dim, dim, 3, stride=(2, 2)))


def upsample2d(in_dim: int, out_dim: int) -> nn.Sequential:
    """``nearest-exact`` by two, then a smoothing 3x3. The 3D codec's version runs the
    sampler in FP32 and stays with it.
    """
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest-exact"),
                         nn.Conv2d(in_dim, out_dim, 3, padding=1))


class RmsNorm(nn.Module):
    """RMS norm over the channel axis of ``[B, C, H, W]``, with a per-channel ``gamma``.

    The scale folds into ``gamma`` rather than the activations: one elementwise pass over
    the feature map instead of two, and no second full-size temporary next to a decode
    that is already the codec's peak. The 3D codec normalizes in FP32 and keeps its own
    class for that.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1).mul_(self.scale * self.gamma)
