# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Lightricks and Hugging Face contributors.
"""LTX-2.3/2.5 native audiovisual vocoder and bandwidth extender.

Adapted to TheNoise from the Apache-2.0 LTX implementations:
Lightricks/LTX-2 v1.6.2: packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py
huggingface/diffusers: src/diffusers/pipelines/ltx2/vocoder.py

Only numerical layers live here; checkpoint loading, memory placement, audio
packaging and the generation controller remain TheNoise's responsibility.
The parameter and persistent buffer names intentionally match the original
LTX-2.x packed checkpoint. No Diffusers, LTX SDK, or ComfyUI runtime imports.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    delta_f = 4 * half_width
    half_size = kernel_size // 2
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (torch.arange(-half_size, half_size) + 0.5 if kernel_size % 2 == 0
            else torch.arange(kernel_size) - half_size)
    if cutoff == 0:
        return torch.zeros_like(time)
    t = 2 * cutoff * time
    sinc = torch.where(t == 0, torch.ones_like(t), torch.sin(math.pi * t) / (math.pi * t))
    kernel = 2 * cutoff * window * sinc
    return kernel / kernel.sum()


class _DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.pad_left = kernel_size // 2 + (kernel_size % 2) - 1
        self.pad_right = kernel_size // 2
        kernel = _kaiser_sinc_filter(0.5 / ratio, 0.6 / ratio, kernel_size)
        self.register_buffer("filter", kernel.view(1, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.to(x).expand(channels, -1, -1),
                        stride=self.ratio, groups=channels)


class UpSample1d(nn.Module):
    """Polyphase sinc interpolation; Hann variant is the deterministic BWE skip path."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None,
                 window_type: str = "kaiser", persistent: bool = True):
        super().__init__()
        self.ratio = ratio
        if window_type == "hann":
            rolloff = 0.99
            lowpass_width = 6
            width = math.ceil(lowpass_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            t = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            window = torch.cos(t.clamp(-lowpass_width, lowpass_width) * math.pi / lowpass_width / 2) ** 2
            kernel = (torch.sinc(t) * window * rolloff / ratio).view(1, 1, -1)
        elif window_type == "kaiser":
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * ratio + (self.kernel_size - ratio) // 2
            self.pad_right = self.pad * ratio + (self.kernel_size - ratio + 1) // 2
            kernel = _kaiser_sinc_filter(0.5 / ratio, 0.6 / ratio,
                                         self.kernel_size).view(1, 1, -1)
        else:
            raise ValueError(f"unsupported vocoder interpolation window: {window_type}")
        self.register_buffer("filter", kernel, persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        weights = self.filter.to(x).expand(channels, -1, -1)
        x = self.ratio * F.conv_transpose1d(x, weights, stride=self.ratio, groups=channels)
        return x[..., self.pad_left:-self.pad_right]


class _SnakeBeta(nn.Module):
    def __init__(self, channels: int, use_beta: bool = True):
        super().__init__()
        self.use_beta = use_beta
        self.alpha = nn.Parameter(torch.zeros(channels))
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp().view(1, -1, 1)
        beta = self.beta.exp().view(1, -1, 1) if self.use_beta else alpha
        return x + (1 / (beta + 1e-9)) * torch.sin(x * alpha).square()


class _AntiAliasAct1d(nn.Module):
    def __init__(self, channels: int, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.upsample = UpSample1d(ratio=ratio, kernel_size=kernel_size)
        self.act = _SnakeBeta(channels)
        self.downsample = _DownSample1d(ratio=ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class _ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: list[int],
                 act_fn: str, antialias: bool, antialias_ratio: int,
                 antialias_kernel_size: int, slope: float):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding="same")
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, padding="same")
            for _ in dilations
        ])
        def activation():
            if act_fn == "leaky_relu":
                return nn.LeakyReLU(slope)
            if antialias:
                return _AntiAliasAct1d(channels, antialias_ratio, antialias_kernel_size)
            return _SnakeBeta(channels, use_beta=act_fn == "snakebeta")
        self.acts1 = nn.ModuleList([activation() for _ in dilations])
        self.acts2 = nn.ModuleList([activation() for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for act1, conv1, act2, conv2 in zip(self.acts1, self.convs1, self.acts2, self.convs2):
            x = x + conv2(act2(conv1(act1(x))))
        return x


class _Vocoder(nn.Module):
    """Mel-to-stereo BigVGAN, named to match LTX-2.x checkpoint leaves."""

    def __init__(self, *, in_channels: int, hidden_channels: int, out_channels: int,
                 upsample_kernel_sizes: list[int], upsample_factors: list[int],
                 resnet_kernel_sizes: list[int], resnet_dilations: list[list[int]],
                 act_fn: str, antialias: bool, antialias_ratio: int,
                 antialias_kernel_size: int, final_act_fn: str | None,
                 final_bias: bool, leaky_relu_negative_slope: float = 0.1):
        super().__init__()
        if len(upsample_kernel_sizes) != len(upsample_factors):
            raise ValueError("LTX vocoder upsample kernel/stride mismatch")
        if len(resnet_kernel_sizes) != len(resnet_dilations):
            raise ValueError("LTX vocoder ResBlock kernel/dilation mismatch")
        if act_fn not in ("snakebeta", "snake", "leaky_relu"):
            raise ValueError(f"unsupported LTX vocoder activation: {act_fn}")
        self.num_upsample_layers = len(upsample_kernel_sizes)
        self.resnets_per_upsample = len(resnet_kernel_sizes)
        self.act_fn = act_fn
        self.negative_slope = leaky_relu_negative_slope
        self.final_act_fn = final_act_fn
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 7, padding=3)
        self.upsamplers = nn.ModuleList()
        self.resnets = nn.ModuleList()
        channels = hidden_channels
        for stride, kernel_size in zip(upsample_factors, upsample_kernel_sizes):
            next_channels = channels // 2
            self.upsamplers.append(nn.ConvTranspose1d(
                channels, next_channels, kernel_size, stride=stride,
                padding=(kernel_size - stride) // 2,
            ))
            for r_kernel, dilations in zip(resnet_kernel_sizes, resnet_dilations):
                self.resnets.append(_ResBlock(
                    next_channels, r_kernel, dilations, act_fn, antialias,
                    antialias_ratio, antialias_kernel_size,
                    leaky_relu_negative_slope,
                ))
            channels = next_channels
        if act_fn in ("snake", "snakebeta"):
            self.act_out = _AntiAliasAct1d(channels, antialias_ratio, antialias_kernel_size)
        else:
            self.act_out = nn.LeakyReLU()
        self.conv_out = nn.Conv1d(channels, out_channels, 7, padding=3, bias=final_bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # (B, stereo, time, mel_bins) -> (B, stereo * mel_bins, time)
        x = self.conv_in(mel.transpose(2, 3).flatten(1, 2))
        for i, upsampler in enumerate(self.upsamplers):
            if self.act_fn == "leaky_relu":
                x = F.leaky_relu(x, negative_slope=self.negative_slope)
            x = upsampler(x)
            first = i * self.resnets_per_upsample
            # Averages same-stage residual paths, as in original BigVGAN.
            x = torch.stack([self.resnets[j](x) for j in
                             range(first, first + self.resnets_per_upsample)]).mean(0)
        x = self.conv_out(self.act_out(x))
        if self.final_act_fn == "tanh":
            x = x.tanh()
        elif self.final_act_fn == "clamp":
            x = x.clamp(-1, 1)
        return x


class _CausalSTFT(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, window_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.window_length = window_length
        self.register_buffer("forward_basis", torch.zeros(filter_length + 2, 1, filter_length))
        self.register_buffer("inverse_basis", torch.zeros(filter_length + 2, 1, filter_length))

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        wave = F.pad(waveform, (max(0, self.window_length - self.hop_length), 0))
        spectrum = F.conv1d(wave, self.forward_basis, stride=self.hop_length)
        real, imag = spectrum.chunk(2, dim=1)
        return torch.sqrt(real.square() + imag.square()), torch.atan2(imag.float(), real.float()).to(real.dtype)


class _MelSTFT(nn.Module):
    def __init__(self, filter_length: int, hop_length: int,
                 window_length: int, num_mel_channels: int):
        super().__init__()
        self.stft_fn = _CausalSTFT(filter_length, hop_length, window_length)
        self.register_buffer("mel_basis", torch.zeros(num_mel_channels, filter_length // 2 + 1))

    def forward(self, waveform: torch.Tensor):
        magnitude, phase = self.stft_fn(waveform)
        energy = torch.norm(magnitude, dim=1)
        mel = torch.matmul(self.mel_basis.to(magnitude.dtype), magnitude)
        return mel.clamp(min=1e-5).log(), magnitude, phase, energy


class LTXVocoderWithBWE(nn.Module):
    """LTX-2.3/2.5 stereo vocoder with 16 kHz -> 48 kHz bandwidth extension."""

    def __init__(
        self, *, in_channels: int = 128, hidden_channels: int = 1536,
        out_channels: int = 2,
        upsample_kernel_sizes: list[int] = (11, 4, 4, 4, 4, 4),
        upsample_factors: list[int] = (5, 2, 2, 2, 2, 2),
        resnet_kernel_sizes: list[int] = (3, 7, 11),
        resnet_dilations: list[list[int]] = ((1, 3, 5),) * 3,
        act_fn: str = "snakebeta", leaky_relu_negative_slope: float = 0.1,
        antialias: bool = True, antialias_ratio: int = 2,
        antialias_kernel_size: int = 12, final_act_fn: str | None = None,
        final_bias: bool = False, bwe_in_channels: int = 128,
        bwe_hidden_channels: int = 512, bwe_out_channels: int = 2,
        bwe_upsample_kernel_sizes: list[int] = (12, 11, 4, 4, 4),
        bwe_upsample_factors: list[int] = (6, 5, 2, 2, 2),
        bwe_resnet_kernel_sizes: list[int] = (3, 7, 11),
        bwe_resnet_dilations: list[list[int]] = ((1, 3, 5),) * 3,
        bwe_act_fn: str = "snakebeta", bwe_leaky_relu_negative_slope: float = 0.1,
        bwe_antialias: bool = True, bwe_antialias_ratio: int = 2,
        bwe_antialias_kernel_size: int = 12, bwe_final_act_fn: str | None = None,
        bwe_final_bias: bool = False, filter_length: int = 512,
        hop_length: int = 80, window_length: int = 512,
        num_mel_channels: int = 64, input_sampling_rate: int = 16000,
        output_sampling_rate: int = 48000,
    ):
        super().__init__()
        if output_sampling_rate % input_sampling_rate:
            raise ValueError("LTX BWE requires an integral sample-rate ratio")
        self.hop_length = hop_length
        self.output_sampling_rate = output_sampling_rate
        self.input_sampling_rate = input_sampling_rate
        self.vocoder = _Vocoder(
            in_channels=in_channels, hidden_channels=hidden_channels,
            out_channels=out_channels, upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_factors=upsample_factors, resnet_kernel_sizes=resnet_kernel_sizes,
            resnet_dilations=resnet_dilations, act_fn=act_fn,
            antialias=antialias, antialias_ratio=antialias_ratio,
            antialias_kernel_size=antialias_kernel_size,
            final_act_fn=final_act_fn, final_bias=final_bias,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )
        self.bwe_generator = _Vocoder(
            in_channels=bwe_in_channels, hidden_channels=bwe_hidden_channels,
            out_channels=bwe_out_channels,
            upsample_kernel_sizes=bwe_upsample_kernel_sizes,
            upsample_factors=bwe_upsample_factors,
            resnet_kernel_sizes=bwe_resnet_kernel_sizes,
            resnet_dilations=bwe_resnet_dilations, act_fn=bwe_act_fn,
            antialias=bwe_antialias, antialias_ratio=bwe_antialias_ratio,
            antialias_kernel_size=bwe_antialias_kernel_size,
            final_act_fn=bwe_final_act_fn, final_bias=bwe_final_bias,
            leaky_relu_negative_slope=bwe_leaky_relu_negative_slope,
        )
        self.mel_stft = _MelSTFT(filter_length, hop_length, window_length, num_mel_channels)
        # The Hann filter is deterministic and not checkpoint-owned. Initializing it
        # on CPU even under torch.device('meta') keeps a valid runtime buffer and
        # preserves the exact non-persistent state_dict contract.
        with torch.device("cpu"):
            self.resampler = UpSample1d(
                ratio=output_sampling_rate // input_sampling_rate,
                window_type="hann", persistent=False,
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.vocoder.conv_in.weight.dtype

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        low = self.vocoder(mel_spec)
        batch, channels, num_samples = low.shape
        remainder = num_samples % self.hop_length
        if remainder:
            low_pad = F.pad(low, (0, self.hop_length - remainder))
        else:
            low_pad = low
        mel, _, _, _ = self.mel_stft(low_pad.flatten(0, 1))
        # BWE takes stereo mel channels as separate inputs.
        mel = mel.unflatten(0, (batch, channels)).transpose(2, 3)
        residual = self.bwe_generator(mel)
        skip = self.resampler(low)
        waveform = (residual + skip).clamp(-1, 1)
        target_samples = num_samples * self.output_sampling_rate // self.input_sampling_rate
        return waveform[..., :target_samples]


__all__ = ["LTXVocoderWithBWE"]
