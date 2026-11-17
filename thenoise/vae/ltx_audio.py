# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Lightricks and Hugging Face contributors.
"""LTX-2.x audio VAE numerical layers, without Diffusers or other pipelines.

Adapted from the Apache-2.0 Lightricks/LTX-2 audio VAE and the compatible
HuggingFace Diffusers LTX-2 audio VAE implementation. TheNoise owns checkpoint
loading, memory placement and audio postprocessing; this module only creates
and runs the original convolutional encoder/decoder structure.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class _CausalConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, causality_axis: str = "height"):
        super().__init__()
        kernel = (kernel_size, kernel_size)
        pad_h, pad_w = kernel[0] - 1, kernel[1] - 1
        if causality_axis == "none":
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif causality_axis in ("width", "width-compatibility"):
            self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif causality_axis == "height":
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise ValueError(f"Invalid audio VAE causal axis: {causality_axis}")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.padding))


class _PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1e-6)


def _norm(channels: int, norm_type: str) -> nn.Module:
    if norm_type == "pixel":
        return _PixelNorm()
    if norm_type == "group":
        return nn.GroupNorm(32, channels, eps=1e-6)
    raise ValueError(f"Invalid audio VAE norm: {norm_type}")


def _conv(in_ch: int, out_ch: int, k: int, axis: str | None) -> nn.Module:
    if axis is None:
        return nn.Conv2d(in_ch, out_ch, k, padding=k // 2)
    return _CausalConv2d(in_ch, out_ch, k, causality_axis=axis)


class _Attn(nn.Module):
    def __init__(self, channels: int, norm_type: str):
        super().__init__()
        self.norm = _norm(channels, norm_type)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, height, width = q.shape
        q = q.reshape(b, c, -1).transpose(1, 2).contiguous()
        k = k.reshape(b, c, -1)
        attention = (q @ k) * (c ** -0.5)
        attention = attention.softmax(dim=2)
        v = v.reshape(b, c, -1)
        h = (v @ attention.transpose(1, 2).contiguous()).reshape(b, c, height, width)
        return x + self.proj_out(h)


class _Resnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str,
                 causality_axis: str | None, dropout: float):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = _norm(in_channels, norm_type)
        self.non_linearity = nn.SiLU()
        self.conv1 = _conv(in_channels, out_channels, 3, causality_axis)
        self.norm2 = _norm(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _conv(out_channels, out_channels, 3, causality_axis)
        if in_channels != out_channels:
            self.nin_shortcut = _conv(in_channels, out_channels, 1, causality_axis)

    def forward(self, x: torch.Tensor, temb=None) -> torch.Tensor:
        h = self.conv1(self.non_linearity(self.norm1(x)))
        h = self.conv2(self.dropout(self.non_linearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class _Downsample(nn.Module):
    def __init__(self, channels: int, axis: str):
        super().__init__()
        self.causality_axis = axis
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causality_axis == "none":
            padding = (0, 1, 0, 1)
        elif self.causality_axis == "width":
            padding = (2, 0, 0, 1)
        elif self.causality_axis == "height":
            padding = (0, 1, 2, 0)
        elif self.causality_axis == "width-compatibility":
            padding = (1, 0, 0, 1)
        else:
            raise ValueError(f"Unsupported audio downsampling axis: {self.causality_axis}")
        return self.conv(F.pad(x, padding))


class _Upsample(nn.Module):
    def __init__(self, channels: int, axis: str | None):
        super().__init__()
        self.causality_axis = axis
        self.conv = _conv(channels, channels, 3, axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))
        if self.causality_axis == "height":
            return x[:, :, 1:, :]
        if self.causality_axis == "width":
            return x[:, :, :, 1:]
        return x


def _middle(channels: int, norm_type: str, axis: str | None,
            dropout: float, attention: bool) -> nn.Module:
    middle = nn.Module()
    middle.block_1 = _Resnet(channels, channels, norm_type, axis, dropout)
    middle.attn_1 = _Attn(channels, norm_type) if attention else nn.Identity()
    middle.block_2 = _Resnet(channels, channels, norm_type, axis, dropout)
    return middle


class _Encoder(nn.Module):
    def __init__(self, *, base_channels: int, output_channels: int,
                 ch_mult: tuple[int, ...], num_res_blocks: int,
                 attn_resolutions: tuple[int, ...] | None,
                 in_channels: int, resolution: int, latent_channels: int,
                 norm_type: str, causality_axis: str, dropout: float,
                 mid_block_add_attention: bool, double_z: bool):
        super().__init__()
        self.conv_in = _conv(in_channels, base_channels, 3, causality_axis)
        self.down = nn.ModuleList()
        channels, current_resolution = base_channels, resolution
        for level, multiplier in enumerate(ch_mult):
            stage = nn.Module()
            stage.block, stage.attn = nn.ModuleList(), nn.ModuleList()
            out_channels = base_channels * multiplier
            for _ in range(num_res_blocks):
                stage.block.append(_Resnet(channels, out_channels, norm_type, causality_axis, dropout))
                channels = out_channels
                if attn_resolutions and current_resolution in attn_resolutions:
                    stage.attn.append(_Attn(channels, norm_type))
            if level != len(ch_mult) - 1:
                stage.downsample = _Downsample(channels, causality_axis)
                current_resolution //= 2
            self.down.append(stage)
        self.mid = _middle(channels, norm_type, causality_axis, dropout, mid_block_add_attention)
        self.norm_out = _norm(channels, norm_type)
        self.non_linearity = nn.SiLU()
        self.conv_out = _conv(channels, latent_channels * (2 if double_z else 1), 3, causality_axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for i, stage in enumerate(self.down):
            for j, block in enumerate(stage.block):
                x = block(x)
                if stage.attn:
                    x = stage.attn[j](x)
            if i != len(self.down) - 1:
                x = stage.downsample(x)
        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))
        return self.conv_out(self.non_linearity(self.norm_out(x)))


class _Decoder(nn.Module):
    def __init__(self, *, base_channels: int, output_channels: int,
                 ch_mult: tuple[int, ...], num_res_blocks: int,
                 attn_resolutions: tuple[int, ...] | None,
                 in_channels: int, resolution: int, latent_channels: int,
                 norm_type: str, causality_axis: str, dropout: float,
                 mid_block_add_attention: bool, mel_bins: int):
        super().__init__()
        channels = base_channels * ch_mult[-1]
        self.conv_in = _conv(latent_channels, channels, 3, causality_axis)
        self.non_linearity = nn.SiLU()
        self.mid = _middle(channels, norm_type, causality_axis, dropout, mid_block_add_attention)
        self.up = nn.ModuleList()
        current_resolution = resolution // (2 ** (len(ch_mult) - 1))
        for level in reversed(range(len(ch_mult))):
            stage = nn.Module()
            stage.block, stage.attn = nn.ModuleList(), nn.ModuleList()
            out_channels = base_channels * ch_mult[level]
            for _ in range(num_res_blocks + 1):
                stage.block.append(_Resnet(channels, out_channels, norm_type, causality_axis, dropout))
                channels = out_channels
                if attn_resolutions and current_resolution in attn_resolutions:
                    stage.attn.append(_Attn(channels, norm_type))
            if level != 0:
                stage.upsample = _Upsample(channels, causality_axis)
                current_resolution *= 2
            self.up.insert(0, stage)
        self.norm_out = _norm(channels, norm_type)
        self.conv_out = _conv(channels, output_channels, 3, causality_axis)
        self.out_ch = output_channels
        self.mel_bins = mel_bins

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        target_time = max(sample.shape[2] * 4 - 3, 1)
        target_freq = self.mel_bins
        x = self.conv_in(sample)
        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))
        for level in reversed(range(len(self.up))):
            stage = self.up[level]
            for j, block in enumerate(stage.block):
                x = block(x)
                if stage.attn:
                    x = stage.attn[j](x)
            if level != 0:
                x = stage.upsample(x)
        x = self.conv_out(self.non_linearity(self.norm_out(x)))
        # Original decoder bounds output on the causal 4x audio grid; pad the
        # trailing edge, never reorder the temporal/frequency axes.
        x = x[:, :self.out_ch, :min(x.shape[2], target_time), :min(x.shape[3], target_freq)]
        time_padding = max(target_time - x.shape[2], 0)
        freq_padding = max(target_freq - x.shape[3], 0)
        if time_padding or freq_padding:
            x = F.pad(x, (0, freq_padding, 0, time_padding))
        return x[:, :self.out_ch, :target_time, :target_freq]


class LTXAudioVAE(nn.Module):
    """LTX audio encoder/decoder, preserving the original checkpoint key layout."""

    def __init__(self, *, base_channels: int = 128, output_channels: int = 2,
                 ch_mult: tuple[int, ...] = (1, 2, 4), num_res_blocks: int = 2,
                 attn_resolutions: tuple[int, ...] | None = None, in_channels: int = 2,
                 resolution: int = 256, latent_channels: int = 8,
                 norm_type: str = "pixel", causality_axis: str = "height",
                 dropout: float = 0.0, mid_block_add_attention: bool = False,
                 sample_rate: int = 16000, mel_hop_length: int = 160,
                 is_causal: bool = True, mel_bins: int = 64, double_z: bool = True):
        super().__init__()
        if causality_axis not in {"none", "width", "height", "width-compatibility"}:
            raise ValueError(f"unsupported audio VAE causal axis: {causality_axis}")
        cfg = dict(base_channels=base_channels, output_channels=output_channels,
                   ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                   attn_resolutions=attn_resolutions, in_channels=in_channels,
                   resolution=resolution, latent_channels=latent_channels,
                   norm_type=norm_type, causality_axis=causality_axis,
                   dropout=dropout, mid_block_add_attention=mid_block_add_attention)
        self.encoder = _Encoder(**cfg, double_z=double_z)
        self.decoder = _Decoder(**cfg, mel_bins=mel_bins)
        self.register_buffer("latents_mean", torch.zeros(base_channels))
        self.register_buffer("latents_std", torch.ones(base_channels))
        self.temporal_compression_ratio = 4

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.conv_in.conv.weight.dtype if hasattr(self.decoder.conv_in, "conv") else self.decoder.conv_in.weight.dtype

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        decoded = self.decoder(z)
        return SimpleNamespace(sample=decoded) if return_dict else (decoded,)


__all__ = ["LTXAudioVAE"]
