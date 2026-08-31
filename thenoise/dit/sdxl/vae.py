# SDXL VAE — decoder-only inference.
#
# The classic CompVis/LDM ``AutoencoderKL`` decoder shared by Stable Diffusion
# XL. The checkpoint stores it under ``first_stage_model.`` (encoder /
# ``quant_conv`` / ``post_quant_conv`` / decoder). Only the decoder is needed
# for text-to-image; the splitter keeps ``post_quant_conv`` and the ``decoder``.
#
# Latent format: 4 channels, 8x spatial compression. The UNet operates on the
# scaled latent space; the VAE decodes with ``latents / scaling_factor`` where
# ``scaling_factor = 0.13025`` (SDXL), so ``decode_to_pixels`` applies that
# before ``post_quant_conv -> decoder``.
#
# Decoder structure (LDM keys):
#   conv_in: 4 -> 512
#   mid:     block_1 (512), attn_1 (512), block_2 (512)
#   up.3:    512, 3 blocks + upsample ; up.2: 512, 3 blocks + upsample
#   up.1:    256, 3 blocks + upsample ; up.0: 128, 3 blocks (no upsample)
#   (the up blocks are applied in *reverse* index order: up.3 -> up.0)
#   norm_out (GroupNorm 128), conv_out: 128 -> 3

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

#: SDXL latent scaling factor (UNet latent space -> VAE raw space).
SCALING_FACTOR = 0.13025


class _VAEResnetBlock(nn.Module):
    """``conv1 - norm1 - act - conv2 - norm2 - act`` + optional shortcut.

    Keys: ``norm1``, ``conv1``, ``norm2``, ``conv2``, ``nin_shortcut``.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.nin_shortcut(x) + h


class _AttnBlock(nn.Module):
    """Single-head 1x1-conv attention. Keys: ``norm``, ``q``, ``k``, ``v``, ``proj_out``."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        B, C, H, W = h.shape
        q = self.q(h).view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        k = self.k(h).view(B, C, -1)  # [B, C, HW]
        v = self.v(h).view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        return self.proj_out(out) + x


class _Upsample(nn.Module):
    """Nearest upsample + conv. Key: ``conv``."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class _UpLevel(nn.Module):
    """One decoder up level: a ``block`` ModuleList of resnets + optional ``upsample``.

    Keys: ``block.0/1/2.*`` (resnets), ``upsample.conv`` (when upsampling).
    """

    def __init__(self, in_ch, out_ch, num_res_blocks, upsample):
        super().__init__()
        self.block = nn.ModuleList(
            [_VAEResnetBlock(in_ch, out_ch)]
            + [_VAEResnetBlock(out_ch, out_ch) for _ in range(num_res_blocks)]
        )
        if upsample:
            self.upsample = _Upsample(out_ch)
        else:
            self.upsample = None

    def forward(self, x):
        for b in self.block:
            x = b(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class _Decoder(nn.Module):
    def __init__(self, z_channels=4, base_dim=128, dim_mult=(1, 2, 4, 4), num_res_blocks=2):
        super().__init__()
        block_in = base_dim * dim_mult[-1]  # 512
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, padding=1)  # 4 -> 512

        # middle: block_1 (512), attn_1 (512), block_2 (512)
        self.mid = nn.Sequential(
            OrderedDict(
                [
                    ("block_1", _VAEResnetBlock(block_in, block_in)),
                    ("attn_1", _AttnBlock(block_in)),
                    ("block_2", _VAEResnetBlock(block_in, block_in)),
                ]
            )
        )

        # up blocks. Index 0 is the *final* (128ch) block applied last; index 3
        # is the first (512ch) applied right after ``mid``. Forward iterates in
        # reverse index order: up.3 (512) -> up.2 (512) -> up.1 (256) -> up.0.
        self.up = nn.ModuleList(
            [
                _UpLevel(256, 128, num_res_blocks, upsample=False),
                _UpLevel(512, 256, num_res_blocks, upsample=True),
                _UpLevel(512, 512, num_res_blocks, upsample=True),
                _UpLevel(block_in, 512, num_res_blocks, upsample=True),
            ]
        )

        self.norm_out = nn.GroupNorm(32, 128, eps=1e-6)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)  # 4 -> 512
        h = self.mid(h)
        for module in reversed(self.up):
            h = module(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class AutoencoderKLSdxl(nn.Module):
    """Decoder-only SDXL VAE. ``decode_to_pixels`` yields pixels in [-1, 1]."""

    def __init__(self, scaling_factor=SCALING_FACTOR):
        super().__init__()
        self.z_dim = 4
        self.compression = 8
        self.scaling_factor = scaling_factor
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        self.decoder = _Decoder()

    @property
    def dtype(self):
        return self.decoder.conv_in.weight.dtype

    @property
    def device(self):
        return self.decoder.conv_in.weight.device

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(self.dtype) / self.scaling_factor
        z = self.post_quant_conv(z)
        img = self.decoder(z)
        return img.clamp(-1.0, 1.0)


def build_sdxl_vae(vae_sd: dict, device="cpu"):
    """Build and load the SDXL VAE decoder from a bare ``decoder.*`` state dict.

    ``vae_sd`` holds ``post_quant_conv.*`` and ``decoder.*`` keys (the splitter
    strips ``first_stage_model.``).
    """
    vae = AutoencoderKLSdxl()
    info = vae.load_state_dict(vae_sd, strict=True, assign=True)
    if info.unexpected_keys or info.missing_keys:
        raise RuntimeError(
            f"SDXL VAE checkpoint did not match: "
            f"missing={info.missing_keys[:10]}, unexpected={info.unexpected_keys[:10]}"
        )
    vae.to(device)
    logger.info("Loaded SDXL VAE")
    return vae


def load_sdxl_vae(vae_path: str, device="cpu", disable_mmap=True):
    """Build and load the SDXL VAE decoder from a safetensors file."""
    from thenoise.utils.safetensors import load_safetensors

    sd = load_safetensors(vae_path, device=str(device), disable_mmap=disable_mmap)
    return build_sdxl_vae(sd, device)
