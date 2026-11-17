# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Lightricks Team and the HuggingFace Team.
# Native model port for TheNoise. Numerical reference:
# huggingface/diffusers v0.43.0, autoencoder_kl_ltx2.py (Apache-2.0).
"""LTX-2.3/2.5 video VAE, using native PyTorch and TheNoise's VAE contract.

This module implements the released distilled video VAE geometry, not the
Diffusers pipeline, loader, tiling framework, or config model. The original
parameter names are preserved for TheNoise's strict streamed Safetensors loader.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from thenoise.vae.qwen_image import DiagonalGaussianDistribution


class _PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.square().mean(dim=1, keepdim=True) + 1e-8)


class _Conv3d(nn.Module):
    """3D convolution with frame-edge temporal padding (LTX causal/noncausal)."""

    def __init__(self, cin: int, cout: int, *, kernel_size: int = 3,
                 stride: int | tuple[int, int, int] = 1,
                 spatial_padding_mode: str = "zeros"):
        super().__init__()
        stride = (stride,) * 3 if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size,) * 3
        self.conv = nn.Conv3d(cin, cout, kernel_size, stride=stride,
                              padding=(0, kernel_size // 2, kernel_size // 2),
                              padding_mode=spatial_padding_mode)

    def forward(self, x: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        n = self.kernel_size[0] - 1
        if causal:
            x = torch.cat((x[:, :, :1].repeat(1, 1, n, 1, 1), x), dim=2)
        else:
            r = n // 2
            x = torch.cat((x[:, :, :1].repeat(1, 1, r, 1, 1), x,
                           x[:, :, -1:].repeat(1, 1, r, 1, 1)), dim=2)
        return self.conv(x)


class _Resnet(nn.Module):
    """LTX pixel-RMS ResNet, with the released checkpoint's shortcut layout."""

    def __init__(self, cin: int, cout: int, *, spatial_padding_mode: str):
        super().__init__()
        self.norm1 = _PixelNorm()
        self.nonlinearity = nn.SiLU()
        self.conv1 = _Conv3d(cin, cout, spatial_padding_mode=spatial_padding_mode)
        self.norm2 = _PixelNorm()
        self.dropout = nn.Dropout(0.0)
        self.conv2 = _Conv3d(cout, cout, spatial_padding_mode=spatial_padding_mode)
        self.norm3 = nn.LayerNorm(cin, eps=1e-6) if cin != cout else None
        self.conv_shortcut = nn.Conv3d(cin, cout, kernel_size=1) if cin != cout else None

    def forward(self, x: torch.Tensor, temb=None, generator=None, causal=True) -> torch.Tensor:
        residual = x
        y = self.conv1(self.nonlinearity(self.norm1(x)), causal=causal)
        y = self.conv2(self.dropout(self.nonlinearity(self.norm2(y))), causal=causal)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(self.norm3(residual.movedim(1, -1)).movedim(-1, 1))
        return y + residual


class _Downsample(nn.Module):
    def __init__(self, cin: int, cout: int, stride: tuple[int, int, int],
                 *, spatial_padding_mode: str):
        super().__init__()
        self.stride = stride
        self.group_size = (cin * stride[0] * stride[1] * stride[2]) // cout
        self.conv = _Conv3d(cin, cout // (stride[0] * stride[1] * stride[2]),
                            spatial_padding_mode=spatial_padding_mode)

    def _pack(self, x: torch.Tensor) -> torch.Tensor:
        t, h, w = self.stride
        x = x.unflatten(4, (-1, w)).unflatten(3, (-1, h)).unflatten(2, (-1, t))
        return x.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)

    def forward(self, x: torch.Tensor, causal=True) -> torch.Tensor:
        x = torch.cat((x[:, :, :self.stride[0]-1], x), dim=2)
        residual = self._pack(x).unflatten(1, (-1, self.group_size)).mean(dim=2)
        return self._pack(self.conv(x, causal=causal)) + residual


class _Upsample(nn.Module):
    def __init__(self, channels: int, stride: tuple[int, int, int], *, residual: bool,
                 factor: int, spatial_padding_mode: str):
        super().__init__()
        self.stride = stride
        self.residual = residual
        self.upscale_factor = factor
        count = stride[0] * stride[1] * stride[2]
        self.conv = _Conv3d(channels, channels * count // factor,
                            spatial_padding_mode=spatial_padding_mode)

    def _unpack(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f, h, w = x.shape
        t, y, z = self.stride
        x = x.reshape(b, -1, t, y, z, f, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return x[:, :, t - 1:]

    def forward(self, x: torch.Tensor, causal=True) -> torch.Tensor:
        y = self._unpack(self.conv(x, causal=causal))
        if self.residual:
            b, _, f, h, w = x.shape
            t, sy, sx = self.stride
            r = x.reshape(b, -1, t, sy, sx, f, h, w)
            r = r.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
            r = r.repeat(1, t * sy * sx // self.upscale_factor, 1, 1, 1)[:, :, t - 1:]
            y = y + r
        return y


class _DownBlock(nn.Module):
    def __init__(self, cin: int, cout: int, num_layers: int, downsample_type: str,
                 downsample: bool, *, spatial_padding_mode: str):
        super().__init__()
        self.resnets = nn.ModuleList([_Resnet(cin, cin, spatial_padding_mode=spatial_padding_mode)
                                      for _ in range(num_layers)])
        strides = {"spatial": (1, 2, 2), "temporal": (2, 1, 1),
                   "spatiotemporal": (2, 2, 2)}
        if downsample:
            if downsample_type not in strides:
                raise ValueError(f"Unsupported LTX downsample type {downsample_type!r}")
            self.downsamplers = nn.ModuleList([_Downsample(
                cin, cout, strides[downsample_type], spatial_padding_mode=spatial_padding_mode)])
        else:
            self.downsamplers = None

    def forward(self, x: torch.Tensor, causal=True) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, causal=causal)
        if self.downsamplers is not None:
            for downsample in self.downsamplers:
                x = downsample(x, causal=causal)
        return x


class _MidBlock(nn.Module):
    def __init__(self, channels: int, num_layers: int, *, spatial_padding_mode: str):
        super().__init__()
        self.resnets = nn.ModuleList([_Resnet(channels, channels, spatial_padding_mode=spatial_padding_mode)
                                      for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, temb=None, generator=None, causal=True) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, causal=causal)
        return x


class _UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int, num_layers: int, scale: bool,
                 upsample_type: str, factor: int, residual: bool,
                 *, spatial_padding_mode: str):
        super().__init__()
        self.conv_in = (_Resnet(cin, cout, spatial_padding_mode=spatial_padding_mode)
                        if cin != cout else None)
        strides = {"spatial": (1, 2, 2), "temporal": (2, 1, 1),
                   "spatiotemporal": (2, 2, 2)}
        if scale:
            if upsample_type not in strides:
                raise ValueError(f"Unsupported LTX upsample type {upsample_type!r}")
            self.upsamplers = nn.ModuleList([_Upsample(
                cout * factor, strides[upsample_type], residual=residual,
                factor=factor, spatial_padding_mode=spatial_padding_mode)])
        else:
            self.upsamplers = None
        self.resnets = nn.ModuleList([_Resnet(cout, cout, spatial_padding_mode=spatial_padding_mode)
                                      for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, temb=None, generator=None, causal=True) -> torch.Tensor:
        if self.conv_in is not None:
            x = self.conv_in(x, causal=causal)
        if self.upsamplers is not None:
            for upsample in self.upsamplers:
                x = upsample(x, causal=causal)
        for resnet in self.resnets:
            x = resnet(x, causal=causal)
        return x


class _Encoder(nn.Module):
    def __init__(self, cin: int, latent: int, channels: tuple[int, ...],
                 layers: tuple[int, ...], scaling: tuple[bool, ...],
                 downsample_types: tuple[str, ...], patch_size: int,
                 *, spatial_padding_mode: str, causal: bool):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = 1
        self.is_causal = causal
        self.conv_in = _Conv3d(cin * patch_size**2, latent, spatial_padding_mode=spatial_padding_mode)
        current = latent
        self.down_blocks = nn.ModuleList()
        for i, next_channels in enumerate(channels):
            self.down_blocks.append(_DownBlock(current, next_channels, layers[i],
                                               downsample_types[i], scaling[i],
                                               spatial_padding_mode=spatial_padding_mode))
            current = next_channels
        self.mid_block = _MidBlock(current, layers[-1], spatial_padding_mode=spatial_padding_mode)
        self.norm_out = _PixelNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = _Conv3d(current, latent + 1, spatial_padding_mode=spatial_padding_mode)

    def forward(self, x: torch.Tensor, causal: bool | None = None) -> torch.Tensor:
        causal = self.is_causal if causal is None else causal
        b, c, t, h, w = x.shape
        p = self.patch_size
        if h % p or w % p:
            raise ValueError("LTX encoder spatial dimensions must be divisible by patch size")
        x = x.reshape(b, c, t, 1, h // p, p, w // p, p)
        x = x.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        x = self.conv_in(x, causal=causal)
        for block in self.down_blocks:
            x = block(x, causal=causal)
        x = self.mid_block(x, causal=causal)
        x = self.conv_out(self.conv_act(self.norm_out(x)), causal=causal)
        last = x[:, -1:].repeat(1, x.shape[1] - 2, 1, 1, 1)
        return torch.cat((x, last), dim=1)


class _Decoder(nn.Module):
    def __init__(self, latent: int, cout: int, channels: tuple[int, ...],
                 layers: tuple[int, ...], scaling: tuple[bool, ...],
                 upsample_types: tuple[str, ...], factors: tuple[int, ...],
                 residuals: tuple[bool, ...], patch_size: int,
                 *, spatial_padding_mode: str, causal: bool):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = 1
        self.is_causal = causal
        channels = tuple(reversed(channels))
        layers = tuple(reversed(layers))
        scaling = tuple(reversed(scaling))
        factors = tuple(reversed(factors))
        residuals = tuple(reversed(residuals))
        current = channels[0]
        self.conv_in = _Conv3d(latent, current, spatial_padding_mode=spatial_padding_mode)
        self.mid_block = _MidBlock(current, layers[0], spatial_padding_mode=spatial_padding_mode)
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels)):
            cin = current // factors[i]
            cout_block = channels[i] // factors[i]
            self.up_blocks.append(_UpBlock(cin, cout_block, layers[i + 1], scaling[i],
                                           upsample_types[i], factors[i], residuals[i],
                                           spatial_padding_mode=spatial_padding_mode))
            current = cout_block
        self.norm_out = _PixelNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = _Conv3d(current, cout * patch_size**2, spatial_padding_mode=spatial_padding_mode)

    def forward(self, x: torch.Tensor, temb=None, causal: bool | None = None) -> torch.Tensor:
        causal = self.is_causal if causal is None else causal
        x = self.conv_in(x, causal=causal)
        x = self.mid_block(x, causal=causal)
        for block in self.up_blocks:
            x = block(x, causal=causal)
        x = self.conv_out(self.conv_act(self.norm_out(x)), causal=causal)
        b, c, t, h, w = x.shape
        p = self.patch_size
        x = x.reshape(b, -1, 1, p, p, t, h, w)
        return x.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)


class LTXVideoVAE(nn.Module):
    """Original LTX video VAE parameter layout and TheNoise encode/decode ABI."""

    def __init__(self, *, in_channels: int = 3, out_channels: int = 3,
                 latent_channels: int = 128,
                 block_out_channels: tuple[int, ...] = (256, 512, 1024, 1024),
                 down_block_types: tuple[str, ...] = ("LTX2VideoDownBlock3D",) * 4,
                 decoder_block_out_channels: tuple[int, ...] = (256, 512, 512, 1024),
                 layers_per_block: tuple[int, ...] = (4, 6, 4, 2, 2),
                 decoder_layers_per_block: tuple[int, ...] = (4, 6, 4, 2, 2),
                 spatio_temporal_scaling: tuple[bool, ...] = (True,) * 4,
                 decoder_spatio_temporal_scaling: tuple[bool, ...] = (True,) * 4,
                 decoder_inject_noise: tuple[bool, ...] = (False,) * 5,
                 downsample_type: tuple[str, ...] = ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
                 upsample_type: tuple[str, ...] = ("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
                 upsample_residual: tuple[bool, ...] = (False,) * 4,
                 upsample_factor: tuple[int, ...] = (2, 2, 1, 2),
                 patch_size: int = 4, patch_size_t: int = 1,
                 resnet_norm_eps: float = 1e-6, encoder_causal: bool = True,
                 decoder_causal: bool = False,
                 encoder_spatial_padding_mode: str = "zeros",
                 decoder_spatial_padding_mode: str = "zeros",
                 spatial_compression_ratio: int = 32,
                 temporal_compression_ratio: int = 8,
                 timestep_conditioning: bool = False,
                 scaling_factor: float = 1.0,
                 **kwargs):
        super().__init__()
        if (kwargs or timestep_conditioning or any(decoder_inject_noise)
                or any(kind != "LTX2VideoDownBlock3D" for kind in down_block_types)
                or patch_size_t != 1 or resnet_norm_eps != 1e-6):
            raise ValueError("Native LTX video VAE supports the released distilled non-timestep configuration only")
        if len(layers_per_block) != len(block_out_channels) + 1:
            raise ValueError("Invalid LTX encoder layers")
        if len(decoder_layers_per_block) != len(decoder_block_out_channels) + 1:
            raise ValueError("Invalid LTX decoder layers")
        self.config = SimpleNamespace(scaling_factor=float(scaling_factor))
        self.encoder = _Encoder(in_channels, latent_channels, block_out_channels,
                                layers_per_block, spatio_temporal_scaling, downsample_type,
                                patch_size, spatial_padding_mode=encoder_spatial_padding_mode,
                                causal=encoder_causal)
        self.decoder = _Decoder(latent_channels, out_channels, decoder_block_out_channels,
                                decoder_layers_per_block, decoder_spatio_temporal_scaling,
                                upsample_type, upsample_factor, upsample_residual, patch_size,
                                spatial_padding_mode=decoder_spatial_padding_mode,
                                causal=decoder_causal)
        self.register_buffer("latents_mean", torch.zeros(latent_channels), persistent=True)
        self.register_buffer("latents_std", torch.ones(latent_channels), persistent=True)
        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.conv_in.conv.weight.dtype

    def encode(self, x: torch.Tensor, causal: bool | None = None,
               return_dict: bool = True):
        posterior = DiagonalGaussianDistribution(self.encoder(x, causal=causal))
        return SimpleNamespace(latent_dist=posterior) if return_dict else (posterior,)

    def decode(self, z: torch.Tensor, temb=None, causal: bool | None = None,
               return_dict: bool = True):
        frames = self.decoder(z, temb, causal=causal)
        return SimpleNamespace(sample=frames) if return_dict else (frames,)


__all__ = ["LTXVideoVAE"]
