# Copied and modified from Diffusers (via Musubi-Tuner). Original copyright notice follows.

# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team. All rights reserved.
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
#
# We gratefully acknowledge the Wan Team for their outstanding contributions.
# QwenImageVAE is further fine-tuned from the Wan Video VAE to achieve improved performance.
# For more information about the Wan VAE, please refer to:
# - GitHub: https://github.com/Wan-Video/Wan2.1
# - arXiv: https://arxiv.org/abs/2503.20314

import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel

from thenoise.utils.safetensors import load_safetensors

from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# region diffusers-vae


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        if generator is not None and generator.device.type != self.parameters.device.type:
            rand_device = generator.device
        else:
            rand_device = self.parameters.device
        sample = torch.randn(self.mean.shape, generator=generator, device=rand_device, dtype=self.parameters.dtype).to(
            self.parameters.device
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


# endregion diffusers-vae


class QwenImageRMS_norm(nn.Module):
    r"""RMS normalization over the channel dim for a 2D image tensor ``[B, C, H, W]``.

    ``F.normalize(x, dim=1) * dim**0.5`` equals ``x * rsqrt(mean(x^2))``, i.e. the
    standard RMS norm. The per-channel ``gamma`` has shape ``(C, 1, 1)``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class QwenImageResample(nn.Module):
    r"""
    A 2D spatial resampling module (upsample or downsample) for single-frame images.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - anything else: Identity (no resampling).
    """

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


class QwenImageResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        non_linearity: str = "silu",
    ) -> None:
        assert non_linearity in ["silu"], "Only 'silu' non-linearity is supported currently."
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = nn.SiLU()  # get_activation(non_linearity)

        # layers
        self.norm1 = QwenImageRMS_norm(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.conv2(x)

        # Add residual connection
        return x + h


class QwenImageAttentionBlock(nn.Module):
    r"""
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, height, width = x.size()

        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # Manual single-head attention instead of ``F.scaled_dot_product_attention``.
        #
        # On ROCm 7.14+ the fused SDPA backends (flash and memory-efficient) produce
        # localized broken-pixel artifacts in this VAE decoder (the math backend and
        # this manual implementation are both clean).
        # using with sdpa_kernel([SDPBackend.MATH]):
        #                x = F.scaled_dot_product_attention(q, k, v)
        # would work as well but seems to introduce a delay (at least on gfx1150).
        scale = channels ** 0.5  # SDPA default scale = 1/sqrt(head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size, channels, height, width)

        # output projection
        x = self.proj(x)

        return x + identity


class QwenImageMidBlock(nn.Module):
    """
    Middle block for QwenImageVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(self, dim: int, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [QwenImageResidualBlock(dim, dim, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First residual block
        x = self.resnets[0](x)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x = resnet(x)

        return x


class QwenImageEncoder2d(nn.Module):
    r"""
    A 2D encoder module (single-frame / still-image).

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        input_channels (int): Number of input channels.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        input_channels: int = 3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        assert non_linearity in ["silu"], "Only 'silu' non-linearity is supported currently."
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.nonlinearity = nn.SiLU()  # get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = nn.Conv2d(input_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock(in_dim, out_dim))
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                self.down_blocks.append(QwenImageResample(out_dim, mode="downsample2d"))
                scale /= 2.0

        # middle blocks
        self.mid_block = QwenImageMidBlock(out_dim, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = QwenImageRMS_norm(out_dim)
        self.conv_out = nn.Conv2d(out_dim, z_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            x = layer(x)

        ## middle
        x = self.mid_block(x)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
        return x


class QwenImageUpBlock(nn.Module):
    """
    A block that handles upsampling for the QwenImageVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or None)
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(current_dim, out_dim, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class QwenImageDecoder2d(nn.Module):
    r"""
    A 2D decoder module (single-frame / still-image).

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        output_channels (int): Number of output channels.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        output_channels: int = 3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        assert non_linearity in ["silu"], "Only 'silu' non-linearity is supported currently."
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.nonlinearity = nn.SiLU()  # get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv_in = nn.Conv2d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = QwenImageMidBlock(dims[0], non_linearity, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0:
                in_dim = in_dim // 2

            # Determine if we need upsampling
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample2d"

            # Create and add the upsampling block
            up_block = QwenImageUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            # Update scale for next iteration
            if upsample_mode is not None:
                scale *= 2.0

        # output blocks
        self.norm_out = QwenImageRMS_norm(out_dim)
        self.conv_out = nn.Conv2d(out_dim, output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## conv1
        x = self.conv_in(x)

        ## middle
        x = self.mid_block(x)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)
        return x


class AutoencoderKLQwenImage(nn.Module):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent
    representations into pixels.

    Only still-image (single-frame) inference is supported: video caching,
    tiling, slicing and spatial chunking have been removed. The encoder is kept
    for future image-to-image workflows.
    """

    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        latents_mean: List[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: List[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        # Hoisted buffers (built once; moved with the module via `.to(device)`).
        self.register_buffer(
            "_latents_mean",
            torch.tensor(latents_mean).view(1, self.z_dim, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_latents_std",
            (1.0 / torch.tensor(latents_std)).view(1, self.z_dim, 1, 1),
            persistent=False,
        )

        self.encoder = QwenImageEncoder2d(
            base_dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, input_channels
        )
        self.quant_conv = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, 1)

        self.decoder = QwenImageDecoder2d(
            base_dim, z_dim, dim_mult, num_res_blocks, attn_scales, input_channels
        )

    @property
    def dtype(self):
        return self.encoder.parameters().__next__().dtype

    @property
    def device(self):
        return self.encoder.parameters().__next__().device

    @property
    def compression(self) -> int:
        """Spatial compression factor (2^num_downsampling_stages), e.g. 8x for this VAE."""
        return 2 ** (len(self.encoder.dim_mult) - 1)

    def _encode(self, x: torch.Tensor):
        out = self.encoder(x)
        return self.quant_conv(out)

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of single-frame images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images, shape [B, C, H, W].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dictionary instead of a plain tuple.

        Returns:
            The latent representations of the encoded images.
        """
        h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return {"latent_dist": posterior}

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        out = self.decoder(self.post_quant_conv(z))
        out = torch.clamp(out, min=-1.0, max=1.0)
        if not return_dict:
            return (out,)
        return {"sample": out}

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        r"""
        Decode a batch of single-frame latents into pixels.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors, shape [B, C, H, W].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dictionary instead of a plain tuple.

        Returns:
            The decoded pixels in [-1, 1].
        """
        decoded = self._decode(z)["sample"]

        if not return_dict:
            return (decoded,)
        return {"sample": decoded}

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.dtype)
        latents_mean = self._latents_mean.to(latents.device, latents.dtype)
        latents_std = self._latents_std.to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        image = self.decode(latents, return_dict=False)[0]  # -1 to 1
        return image.clamp(-1.0, 1.0)

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel values to latents and apply normalization using mean/std.

        Args:
            pixels (torch.Tensor): Input pixels in [0, 1] range with shape [B, C, H, W].

        Returns:
            torch.Tensor: Normalized latents
        """
        pixels = pixels.to(self.dtype)

        # Encode to latent space
        posterior = self.encode(pixels, return_dict=False)[0]
        latents = posterior.mode()  # Use mode instead of sampling for deterministic results

        # Apply normalization using mean/std
        latents_mean = self._latents_mean.to(latents.device, latents.dtype)
        latents_std = self._latents_std.to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std

        return latents


# region utils

# This region is not included in the original implementation. Added for musubi-tuner/sd-scripts.


# Convert ComfyUI keys to standard keys if necessary
def convert_comfyui_state_dict(sd):
    if "conv1.bias" not in sd:
        return sd

    # Key mapping from ComfyUI VAE to official VAE, auto-generated by a script
    key_map = {
        "conv1": "quant_conv",
        "conv2": "post_quant_conv",
        "decoder.conv1": "decoder.conv_in",
        "decoder.head.0": "decoder.norm_out",
        "decoder.head.2": "decoder.conv_out",
        "decoder.middle.0.residual.0": "decoder.mid_block.resnets.0.norm1",
        "decoder.middle.0.residual.2": "decoder.mid_block.resnets.0.conv1",
        "decoder.middle.0.residual.3": "decoder.mid_block.resnets.0.norm2",
        "decoder.middle.0.residual.6": "decoder.mid_block.resnets.0.conv2",
        "decoder.middle.1.norm": "decoder.mid_block.attentions.0.norm",
        "decoder.middle.1.proj": "decoder.mid_block.attentions.0.proj",
        "decoder.middle.1.to_qkv": "decoder.mid_block.attentions.0.to_qkv",
        "decoder.middle.2.residual.0": "decoder.mid_block.resnets.1.norm1",
        "decoder.middle.2.residual.2": "decoder.mid_block.resnets.1.conv1",
        "decoder.middle.2.residual.3": "decoder.mid_block.resnets.1.norm2",
        "decoder.middle.2.residual.6": "decoder.mid_block.resnets.1.conv2",
        "decoder.upsamples.0.residual.0": "decoder.up_blocks.0.resnets.0.norm1",
        "decoder.upsamples.0.residual.2": "decoder.up_blocks.0.resnets.0.conv1",
        "decoder.upsamples.0.residual.3": "decoder.up_blocks.0.resnets.0.norm2",
        "decoder.upsamples.0.residual.6": "decoder.up_blocks.0.resnets.0.conv2",
        "decoder.upsamples.1.residual.0": "decoder.up_blocks.0.resnets.1.norm1",
        "decoder.upsamples.1.residual.2": "decoder.up_blocks.0.resnets.1.conv1",
        "decoder.upsamples.1.residual.3": "decoder.up_blocks.0.resnets.1.norm2",
        "decoder.upsamples.1.residual.6": "decoder.up_blocks.0.resnets.1.conv2",
        "decoder.upsamples.10.residual.0": "decoder.up_blocks.2.resnets.2.norm1",
        "decoder.upsamples.10.residual.2": "decoder.up_blocks.2.resnets.2.conv1",
        "decoder.upsamples.10.residual.3": "decoder.up_blocks.2.resnets.2.norm2",
        "decoder.upsamples.10.residual.6": "decoder.up_blocks.2.resnets.2.conv2",
        "decoder.upsamples.11.resample.1": "decoder.up_blocks.2.upsamplers.0.resample.1",
        "decoder.upsamples.12.residual.0": "decoder.up_blocks.3.resnets.0.norm1",
        "decoder.upsamples.12.residual.2": "decoder.up_blocks.3.resnets.0.conv1",
        "decoder.upsamples.12.residual.3": "decoder.up_blocks.3.resnets.0.norm2",
        "decoder.upsamples.12.residual.6": "decoder.up_blocks.3.resnets.0.conv2",
        "decoder.upsamples.13.residual.0": "decoder.up_blocks.3.resnets.1.norm1",
        "decoder.upsamples.13.residual.2": "decoder.up_blocks.3.resnets.1.conv1",
        "decoder.upsamples.13.residual.3": "decoder.up_blocks.3.resnets.1.norm2",
        "decoder.upsamples.13.residual.6": "decoder.up_blocks.3.resnets.1.conv2",
        "decoder.upsamples.14.residual.0": "decoder.up_blocks.3.resnets.2.norm1",
        "decoder.upsamples.14.residual.2": "decoder.up_blocks.3.resnets.2.conv1",
        "decoder.upsamples.14.residual.3": "decoder.up_blocks.3.resnets.2.norm2",
        "decoder.upsamples.14.residual.6": "decoder.up_blocks.3.resnets.2.conv2",
        "decoder.upsamples.2.residual.0": "decoder.up_blocks.0.resnets.2.norm1",
        "decoder.upsamples.2.residual.2": "decoder.up_blocks.0.resnets.2.conv1",
        "decoder.upsamples.2.residual.3": "decoder.up_blocks.0.resnets.2.norm2",
        "decoder.upsamples.2.residual.6": "decoder.up_blocks.0.resnets.2.conv2",
        "decoder.upsamples.3.resample.1": "decoder.up_blocks.0.upsamplers.0.resample.1",
        "decoder.upsamples.3.time_conv": "decoder.up_blocks.0.upsamplers.0.time_conv",
        "decoder.upsamples.4.residual.0": "decoder.up_blocks.1.resnets.0.norm1",
        "decoder.upsamples.4.residual.2": "decoder.up_blocks.1.resnets.0.conv1",
        "decoder.upsamples.4.residual.3": "decoder.up_blocks.1.resnets.0.norm2",
        "decoder.upsamples.4.residual.6": "decoder.up_blocks.1.resnets.0.conv2",
        "decoder.upsamples.4.shortcut": "decoder.up_blocks.1.resnets.0.conv_shortcut",
        "decoder.upsamples.5.residual.0": "decoder.up_blocks.1.resnets.1.norm1",
        "decoder.upsamples.5.residual.2": "decoder.up_blocks.1.resnets.1.conv1",
        "decoder.upsamples.5.residual.3": "decoder.up_blocks.1.resnets.1.norm2",
        "decoder.upsamples.5.residual.6": "decoder.up_blocks.1.resnets.1.conv2",
        "decoder.upsamples.6.residual.0": "decoder.up_blocks.1.resnets.2.norm1",
        "decoder.upsamples.6.residual.2": "decoder.up_blocks.1.resnets.2.conv1",
        "decoder.upsamples.6.residual.3": "decoder.up_blocks.1.resnets.2.norm2",
        "decoder.upsamples.6.residual.6": "decoder.up_blocks.1.resnets.2.conv2",
        "decoder.upsamples.7.resample.1": "decoder.up_blocks.1.upsamplers.0.resample.1",
        "decoder.upsamples.7.time_conv": "decoder.up_blocks.1.upsamplers.0.time_conv",
        "decoder.upsamples.8.residual.0": "decoder.up_blocks.2.resnets.0.norm1",
        "decoder.upsamples.8.residual.2": "decoder.up_blocks.2.resnets.0.conv1",
        "decoder.upsamples.8.residual.3": "decoder.up_blocks.2.resnets.0.norm2",
        "decoder.upsamples.8.residual.6": "decoder.up_blocks.2.resnets.0.conv2",
        "decoder.upsamples.9.residual.0": "decoder.up_blocks.2.resnets.1.norm1",
        "decoder.upsamples.9.residual.2": "decoder.up_blocks.2.resnets.1.conv1",
        "decoder.upsamples.9.residual.3": "decoder.up_blocks.2.resnets.1.norm2",
        "decoder.upsamples.9.residual.6": "decoder.up_blocks.2.resnets.1.conv2",
        "encoder.conv1": "encoder.conv_in",
        "encoder.downsamples.0.residual.0": "encoder.down_blocks.0.norm1",
        "encoder.downsamples.0.residual.2": "encoder.down_blocks.0.conv1",
        "encoder.downsamples.0.residual.3": "encoder.down_blocks.0.norm2",
        "encoder.downsamples.0.residual.6": "encoder.down_blocks.0.conv2",
        "encoder.downsamples.1.residual.0": "encoder.down_blocks.1.norm1",
        "encoder.downsamples.1.residual.2": "encoder.down_blocks.1.conv1",
        "encoder.downsamples.1.residual.3": "encoder.down_blocks.1.norm2",
        "encoder.downsamples.1.residual.6": "encoder.down_blocks.1.conv2",
        "encoder.downsamples.10.residual.0": "encoder.down_blocks.10.norm1",
        "encoder.downsamples.10.residual.2": "encoder.down_blocks.10.conv1",
        "encoder.downsamples.10.residual.3": "encoder.down_blocks.10.norm2",
        "encoder.downsamples.10.residual.6": "encoder.down_blocks.10.conv2",
        "encoder.downsamples.2.resample.1": "encoder.down_blocks.2.resample.1",
        "encoder.downsamples.3.residual.0": "encoder.down_blocks.3.norm1",
        "encoder.downsamples.3.residual.2": "encoder.down_blocks.3.conv1",
        "encoder.downsamples.3.residual.3": "encoder.down_blocks.3.norm2",
        "encoder.downsamples.3.residual.6": "encoder.down_blocks.3.conv2",
        "encoder.downsamples.3.shortcut": "encoder.down_blocks.3.conv_shortcut",
        "encoder.downsamples.4.residual.0": "encoder.down_blocks.4.norm1",
        "encoder.downsamples.4.residual.2": "encoder.down_blocks.4.conv1",
        "encoder.downsamples.4.residual.3": "encoder.down_blocks.4.norm2",
        "encoder.downsamples.4.residual.6": "encoder.down_blocks.4.conv2",
        "encoder.downsamples.5.resample.1": "encoder.down_blocks.5.resample.1",
        "encoder.downsamples.5.time_conv": "encoder.down_blocks.5.time_conv",
        "encoder.downsamples.6.residual.0": "encoder.down_blocks.6.norm1",
        "encoder.downsamples.6.residual.2": "encoder.down_blocks.6.conv1",
        "encoder.downsamples.6.residual.3": "encoder.down_blocks.6.norm2",
        "encoder.downsamples.6.residual.6": "encoder.down_blocks.6.conv2",
        "encoder.downsamples.6.shortcut": "encoder.down_blocks.6.conv_shortcut",
        "encoder.downsamples.7.residual.0": "encoder.down_blocks.7.norm1",
        "encoder.downsamples.7.residual.2": "encoder.down_blocks.7.conv1",
        "encoder.downsamples.7.residual.3": "encoder.down_blocks.7.norm2",
        "encoder.downsamples.7.residual.6": "encoder.down_blocks.7.conv2",
        "encoder.downsamples.8.resample.1": "encoder.down_blocks.8.resample.1",
        "encoder.downsamples.8.time_conv": "encoder.down_blocks.8.time_conv",
        "encoder.downsamples.9.residual.0": "encoder.down_blocks.9.norm1",
        "encoder.downsamples.9.residual.2": "encoder.down_blocks.9.conv1",
        "encoder.downsamples.9.residual.3": "encoder.down_blocks.9.norm2",
        "encoder.downsamples.9.residual.6": "encoder.down_blocks.9.conv2",
        "encoder.head.0": "encoder.norm_out",
        "encoder.head.2": "encoder.conv_out",
        "encoder.middle.0.residual.0": "encoder.mid_block.resnets.0.norm1",
        "encoder.middle.0.residual.2": "encoder.mid_block.resnets.0.conv1",
        "encoder.middle.0.residual.3": "encoder.mid_block.resnets.0.norm2",
        "encoder.middle.0.residual.6": "encoder.mid_block.resnets.0.conv2",
        "encoder.middle.1.norm": "encoder.mid_block.attentions.0.norm",
        "encoder.middle.1.proj": "encoder.mid_block.attentions.0.proj",
        "encoder.middle.1.to_qkv": "encoder.mid_block.attentions.0.to_qkv",
        "encoder.middle.2.residual.0": "encoder.mid_block.resnets.1.norm1",
        "encoder.middle.2.residual.2": "encoder.mid_block.resnets.1.conv1",
        "encoder.middle.2.residual.3": "encoder.mid_block.resnets.1.norm2",
        "encoder.middle.2.residual.6": "encoder.mid_block.resnets.1.conv2",
    }

    new_state_dict = {}
    for key in sd.keys():
        new_key = key
        key_without_suffix = key.rsplit(".", 1)[0]
        if key_without_suffix in key_map:
            new_key = key.replace(key_without_suffix, key_map[key_without_suffix])
        new_state_dict[new_key] = sd[key]

    logger.info("Converted ComfyUI AutoencoderKL state dict keys to official format")
    return new_state_dict


def load_qwen_vae(
    vae_path: str,
    input_channels: int = 3,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
) -> AutoencoderKLQwenImage:
    """Load the Qwen-Image VAE from a given path."""
    VAE_CONFIG_JSON = """
{
  "_class_name": "AutoencoderKLQwenImage",
  "_diffusers_version": "0.34.0.dev0",
  "attn_scales": [],
  "base_dim": 96,
  "dim_mult": [
    1,
    2,
    4,
    4
  ],
  "latents_mean": [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921
  ],
  "latents_std": [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916
  ],
  "num_res_blocks": 2,
  "z_dim": 16
}
"""
    logger.info("Initializing VAE")

    config = json.loads(VAE_CONFIG_JSON)
    vae = AutoencoderKLQwenImage(
        base_dim=config["base_dim"],
        z_dim=config["z_dim"],
        dim_mult=config["dim_mult"],
        num_res_blocks=config["num_res_blocks"],
        attn_scales=config["attn_scales"],
        latents_mean=config["latents_mean"],
        latents_std=config["latents_std"],
        input_channels=input_channels,
    )

    logger.info(f"Loading VAE from {vae_path}")
    state_dict = load_safetensors(vae_path, device=device, disable_mmap=disable_mmap)

    # Convert ComfyUI VAE keys to official VAE keys
    state_dict = convert_comfyui_state_dict(state_dict)

    # Collapse the 3D (video) weight layout to 2D for single-frame inference:
    #   - every 5D conv weight -> 2D by its LAST time slice (index 2 for k=3,
    #     index 0 for k=1). Causal padding pads the time axis by (2, 0), so the
    #     single frame lands at the end of the padded axis and only that kernel
    #     time slice contributes;
    #   - residual/norm_out RMS gammas (C, 1, 1, 1) -> (C, 1, 1);
    #   - drop the never-used `time_conv` layers entirely.
    state_dict = {k: v for k, v in state_dict.items() if ".time_conv." not in k}
    for key in state_dict:
        val = state_dict[key]
        if val.dim() == 5:
            state_dict[key] = val[:, :, -1]
        elif key.endswith(".gamma") and val.dim() == 4:
            state_dict[key] = val.reshape(val.shape[0], 1, 1)

    info = vae.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Loaded VAE: {info}")

    vae.to(device)
    return vae
