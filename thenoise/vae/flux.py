# Flux VAE decoder (AutoencoderKL), ported from diffusers / black-forest-labs.
#
# Only still-image DECODE is supported here (this engine generates images; it never
# encodes pixels), so only the decoder side of the Flux AutoencoderKL is kept. The
# encoder, quant_conv and post_quant_conv are omitted. Latent post-processing follows
# the Flux convention: z = (latent / scaling_factor) + shift_factor before decoding,
# then the pixels are clamped to [-1, 1].
#
# Copyright 2023 The HuggingFace Team. Licensed under the Apache-2.0 License.
# Copyright 2024 Black Forest Labs. The Flux VAE is released under the Apache-2.0 License.

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.utils.safetensors import load_safetensors
from thenoise.utils.setup_logging import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


class _ResnetBlock(nn.Module):
    """Resnet block with ComfyUI / Flux-VAE key naming (``norm1/conv1/norm2/conv2/nin_shortcut``)."""

    def __init__(self, in_channels, out_channels, norm_num_groups, act_fn="silu"):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_num_groups, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(norm_num_groups, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = nn.SiLU() if act_fn == "silu" else None
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        hidden = self.nonlinearity(self.norm1(x))
        hidden = self.conv1(hidden)
        hidden = self.nonlinearity(self.norm2(hidden))
        hidden = self.conv2(hidden)
        return hidden + self.nin_shortcut(x)


class _Attention(nn.Module):
    """Single-head spatial attention with QKV 1x1 convs (``norm/q/k/v/proj_out``)."""

    def __init__(self, channels, norm_num_groups):
        super().__init__()
        self.norm = nn.GroupNorm(norm_num_groups, channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        hidden = self.norm(x)
        # Spatial attention over the h*w positions with a single head:
        # [B, C, H, W] -> [B, 1, H*W, C]; SDPA over the H*W sequence.
        q = self.q(hidden).view(b, 1, c, -1).transpose(2, 3).contiguous()  # (b, 1, hw, c)
        k = self.k(hidden).view(b, 1, c, -1).transpose(2, 3).contiguous()
        v = self.v(hidden).view(b, 1, c, -1).transpose(2, 3).contiguous()
        hidden = F.scaled_dot_product_attention(q, k, v)
        hidden = hidden.transpose(2, 3).reshape(b, c, h, w).contiguous()
        return x + self.proj_out(hidden)


class _Upsample(nn.Module):
    """Nearest 2x upsampler followed by a 3x3 conv (``upsample.conv``)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class _MidBlock(nn.Module):
    def __init__(self, channels, norm_num_groups, act_fn, add_attention=True):
        super().__init__()
        self.block_1 = _ResnetBlock(channels, channels, norm_num_groups, act_fn)
        self.attn_1 = _Attention(channels, norm_num_groups) if add_attention else nn.Identity()
        self.block_2 = _ResnetBlock(channels, channels, norm_num_groups, act_fn)

    def forward(self, x):
        x = self.block_1(x)
        x = self.attn_1(x)
        return self.block_2(x)


class _UpBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, norm_num_groups, act_fn, add_upsampling):
        super().__init__()
        blocks = []
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            blocks.append(_ResnetBlock(res_in, out_channels, norm_num_groups, act_fn))
        self.block = nn.ModuleList(blocks)  # -> block.0 / block.1 / block.2
        self.upsample = _Upsample(out_channels) if add_upsampling else nn.Identity()

    def forward(self, x):
        for resnet in self.block:
            x = resnet(x)
        x = self.upsample(x)
        return x


class AutoencoderKLFlux(nn.Module):
    """Flux VAE decoder (decode-only), with ComfyUI / Flux-VAE key naming.

    The decoder up blocks are applied in the order ``up.3 -> up.2 -> up.1 -> up.0``
    (512 -> 512 -> 256 -> 128), with nearest upsamplers on the first three and none
    on the last — 8x spatial compression. Accepts canonical 4D latents
    ``[B, C, H, W]``.
    """

    z_dim = 16
    scaling_factor = 0.3611
    shift_factor = 0.1159

    def __init__(self, in_channels=16, out_channels=3, block_out_channels=(128, 256, 512, 512),
                 layers_per_block=2, norm_num_groups=32, act_fn="silu", mid_block_add_attention=True):
        super().__init__()
        self.block_out_channels = list(block_out_channels)
        self.conv_in = nn.Conv2d(in_channels, self.block_out_channels[-1], kernel_size=3, padding=1)
        self.mid = _MidBlock(self.block_out_channels[-1], norm_num_groups, act_fn, mid_block_add_attention)

        # Decoder application order (up.3 -> up.2 -> up.1 -> up.0) with out dims
        # [512, 512, 256, 128]; the first three upsample, the last does not.
        outs = self.block_out_channels[::-1]                 # [512, 512, 256, 128]
        ins = [self.block_out_channels[-1]] + outs[:-1]      # [512, 512, 512, 256]
        flags = [True, True, True, False]
        indices = ["3", "2", "1", "0"]
        self.up = nn.ModuleDict(
            {
                idx: _UpBlock(
                    layers_per_block + 1, in_c, out_c, norm_num_groups, act_fn, add_upsampling=flag
                )
                for idx, in_c, out_c, flag in zip(indices, ins, outs, flags)
            }
        )

        self.norm_out = nn.GroupNorm(norm_num_groups, self.block_out_channels[0], eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(self.block_out_channels[0], out_channels, kernel_size=3, padding=1)

    @property
    def dtype(self):
        return self.conv_in.weight.dtype

    @property
    def device(self):
        return self.conv_in.weight.device

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        x = self.mid(x)
        for idx in ("3", "2", "1", "0"):
            x = self.up[idx](x)
        x = self.norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode canonical 2D latents ``[B, C, H, W]`` to pixels in [-1, 1].

        The VAE is 2D / single-frame: it accepts the canonical latent directly and
        returns ``[B, C, H, W]`` pixels (no frame axis is added).
        """
        z = (latents.float() / self.scaling_factor) + self.shift_factor
        image = self.decode(z.to(self.dtype))
        return image.clamp(-1.0, 1.0)


def load_flux_vae(
    vae_path: str,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> AutoencoderKLFlux:
    """Load the Flux VAE decoder weights from ``vae_path`` (e.g. flux-dev/ae.safetensors).

    Only ``decoder.*`` keys are needed; the encoder/quant_conv/post_quant_conv keys
    present in the file are ignored.
    """
    device = torch.device(device)
    logger.info("Loading Flux VAE from %s", vae_path)
    state_dict = load_safetensors(vae_path, device=device, disable_mmap=disable_mmap)

    vae = AutoencoderKLFlux()
    # Keep only the decoder side and strip the ``decoder.`` prefix so keys line up
    # with the model's bare ``conv_in.*`` / ``up.*`` names.
    decoder_sd = {
        k[len("decoder."):]: v for k, v in state_dict.items() if k.startswith("decoder.")
    }
    if not decoder_sd:
        raise ValueError(f"No 'decoder.*' keys found in {vae_path} (not a Flux/AutoencoderKL VAE?)")
    info = vae.load_state_dict(decoder_sd, strict=True, assign=True)
    logger.info("Loaded Flux VAE decoder: %s", info)

    vae.to(device)
    if dtype is not None:
        vae.to(dtype)
    return vae.eval().requires_grad_(False)
