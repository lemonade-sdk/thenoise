"""H3 audio codec — a DAC-lineage waveform encoder behind a BigVGAN vocoder.

Ported from ComfyUI's ``comfy/ldm/minimax/audio_vae.py`` with its module tree — and so its
checkpoint key names — verbatim; only the ops differ. The released file already has its
weight-norm parametrizations folded into plain weights.

Both halves are built: a clip's sound is decoded, and a sound a clip is asked to keep is
encoded. The encoder is mono and reads one channel at a time, the vocoder is mono too, and
the two stereo lanes are folded into the batch on the way in and out.

The latent is ``[B, 32, 2, T]`` — 32 channels over 2 stereo lanes at 40 frames per second.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from thenoise.utils.loader import load_codec
from thenoise.utils.media import pcm_resample
from thenoise.utils.safetensors import Header, Signature
from thenoise.utils.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

COMPUTE_DTYPE = torch.float32

SAMPLE_RATE = 32000
STEREO_LANES = 2

#: Latent geometry, as the checkpoint's own metadata states it.
LATENT_CHANNELS = 32
LATENT_DIM = 2048
ENCODER_DIM = 64
ENCODER_RATES = (2, 4, 4, 5, 5)
#: Heads the encoder's posterior attention runs over its 2048-wide frames.
ATTN_HEADS = 8
DECODER_DIM = 1024
UPSAMPLE_RATES = (5, 5, 2, 2, 2, 2, 2)
UPSAMPLE_KERNELS = (9, 9, 4, 4, 4, 4, 4)
RESBLOCK_KERNELS = (3, 7, 11)
RESBLOCK_DILATIONS = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

#: Samples one latent frame stands for, and the frames one second of audio is: what the
#: encoder squashes the vocoder restores, exactly.
SAMPLES_PER_LATENT = math.prod(ENCODER_RATES)
LATENTS_PER_SECOND = SAMPLE_RATE // SAMPLES_PER_LATENT

#: Kaiser-windowed sinc width for the anti-aliased activations.
RESAMPLING_KERNEL = 12

# --------------------------------------------------------------------- activations


def snake(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """``x + sin²(alpha·x) / beta`` — a periodic, learnable activation.

    ``1e-9`` is in the denominator rather than a clamp on ``beta``, as in the reference:
    a trained ``beta`` can sit near zero and stay valid.
    """
    t = torch.sin(alpha * x)
    return t.mul_(t).mul_((beta + 1e-9).reciprocal()).add_(x)


class SnakeBeta(nn.Module):
    """Snake with independent alpha and beta, both stored in the log domain."""

    def __init__(self, in_features: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(in_features))
        self.beta = nn.Parameter(torch.empty(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp().to(x).view(1, -1, 1)
        beta = self.beta.exp().to(x).view(1, -1, 1)
        return snake(x, alpha, beta)


# --------------------------------------------------------- alias-free resampling


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """The lowpass ``[1, 1, kernel_size]`` an alias-free activation resamples with.

    Designed, not stored, and normalized to unit sum: otherwise the constant component
    leaks at the new rate.
    """
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (filter_ / filter_.sum()).view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    """Polyphase upsampling by ``ratio``: interpolate, lowpass, trim to ``L * ratio``."""

    def __init__(self, ratio: int = 2, kernel_size: int = RESAMPLING_KERNEL):
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = F.conv_transpose1d(x, self.filter.expand(channels, -1, -1).to(x),
                               stride=self.ratio, groups=channels)
        return x.mul_(self.ratio)[..., self.pad_left:-self.pad_right]


class LowPassFilter1d(nn.Module):
    """The downsampling half: replicate-pad, then a strided lowpass."""

    def __init__(self, cutoff: float = 0.5, half_width: float = 0.6,
                 stride: int = 1, kernel_size: int = RESAMPLING_KERNEL):
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(channels, -1, -1).to(x),
                        stride=self.stride, groups=channels)


class DownSample1d(nn.Module):
    """The downsampling half of an alias-free activation, named as the checkpoint names it."""

    def __init__(self, ratio: int = 2, kernel_size: int = RESAMPLING_KERNEL):
        super().__init__()
        self.lowpass = LowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """``upsample ×2 → activate → downsample ×2``, so the activation sees no alias.

    A Snake in the time domain is a source of harmonics; activated at the original rate
    they fold back as noise. The round trip leaves the length alone.
    """

    def __init__(self, activation: nn.Module, ratio: int = 2,
                 kernel_size: int = RESAMPLING_KERNEL):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(ratio, kernel_size)
        self.downsample = DownSample1d(ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


# --------------------------------------------------------------------- encoder


class Snake1d(nn.Module):
    """Snake with one parameter per channel, which is what the encoder side trains."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(x)
        return snake(x, alpha, alpha)


class ResidualUnit(nn.Module):
    """Two activations and a dilated conv, then the shortcut, trimmed to match."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.block = nn.Sequential(
            Snake1d(channels),
            nn.Conv1d(channels, channels, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(channels),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad:
            x = x[..., pad:-pad]
        return y.add_(x)


class EncoderBlock(nn.Module):
    """Three residual units at the entering width, then a strided conv to the next."""

    def __init__(self, channels: int, stride: int):
        super().__init__()
        half = channels // 2
        self.block = nn.Sequential(
            ResidualUnit(half, dilation=1),
            ResidualUnit(half, dilation=3),
            ResidualUnit(half, dilation=9),
            Snake1d(half),
            nn.Conv1d(half, channels, kernel_size=2 * stride, stride=stride,
                      padding=math.ceil(stride / 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """Samples to the encoder's feature frame: one channel in, ``latent_dim`` out.

    Mono by construction -- the stereo lanes are folded into the batch upstream, which is
    the same trick the vocoder plays on the way out.
    """

    def __init__(self, d_model: int = ENCODER_DIM, strides=ENCODER_RATES,
                 d_latent: int = LATENT_DIM):
        super().__init__()
        block = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(EncoderBlock(d_model, stride=stride))
        block += [Snake1d(d_model), nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ------------------------------------------------------- encoder posterior head


class GeGluMlp(nn.Module):
    """GELU-gated MLP: two projections to the hidden width, one back."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gate = F.gelu(self.w0(x), approximate="tanh")
        return self.w2(gate.mul_(self.w1(x)))


class CausalAttention(nn.Module):
    """One causal attention over the latent frames, collapsed to ``out_dim``.

    The heads are averaged and the result pooled along the *head* axis to the posterior
    width, which is what the checkpoint's shapes say it does: ``out_dim`` is narrower than
    the head axis, so the pool is a projection and not a downsample in time.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.empty(in_dim))
        self.v_bias = nn.Parameter(torch.empty(in_dim))
        self.register_buffer("zero_k_bias", torch.empty(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, rows, channels = x.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias.to(x), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias.to(x))
        qkv = qkv.reshape(batch, rows, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        seen = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        pooled = F.adaptive_avg_pool1d(seen.mean(dim=1), self.out_dim)
        return self.proj(pooled)


class AttnProjection(nn.Module):
    """The encoder's head: attention and a strided linear both reading 2048-wide frames."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, int(out_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm3(x)).add_(self.attn(self.norm1(x)))
        return x.add_(self.mlp(self.norm2(x)))


# --------------------------------------------------------------------- BigVGAN


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Padding that keeps the length of a dilated ``kernel_size`` conv."""
    return (kernel_size * dilation - dilation) // 2


class AMPBlock1(nn.Module):
    """Anti-aliased multi-periodicity block: ``dilation``-spaced pairs, each a residual."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=d,
                      padding=get_padding(kernel_size, d))
            for d in dilation
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1,
                      padding=get_padding(kernel_size, 1))
            for _ in dilation
        )
        self.activations = nn.ModuleList(
            Activation1d(SnakeBeta(channels)) for _ in range(len(self.convs1) * 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, first, second):
            y = act1(x)
            y = act2(conv1(y))
            x = conv2(y).add_(x)
        return x


class BigVGAN(nn.Module):
    """The vocoder: latent frames to samples, with the encoder's rates undone in kind.

    ``use_bias_at_final`` and ``use_tanh_at_final`` are both off in this configuration;
    the output is clamped instead.
    """

    def __init__(self, num_mels: int = LATENT_DIM,
                 upsample_initial_channel: int = DECODER_DIM,
                 upsample_rates=UPSAMPLE_RATES,
                 upsample_kernel_sizes=UPSAMPLE_KERNELS,
                 resblock_kernel_sizes=RESBLOCK_KERNELS,
                 resblock_dilation_sizes=RESBLOCK_DILATIONS):
        super().__init__()
        if not len(upsample_rates) == len(upsample_kernel_sizes) or not upsample_rates:
            raise ValueError(
                f"{len(upsample_rates)} upsample rates against "
                f"{len(upsample_kernel_sizes)} kernel sizes"
            )
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsample_factor = math.prod(upsample_rates)

        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList(
            nn.ModuleList([
                nn.ConvTranspose1d(upsample_initial_channel // 2 ** i,
                                   upsample_initial_channel // 2 ** (i + 1),
                                   kernel, rate, padding=(kernel - rate) // 2)
            ])
            for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        )

        self.resblocks = nn.ModuleList()
        for i, _ in enumerate(self.ups):
            channels = upsample_initial_channel // 2 ** (i + 1)
            self.resblocks.extend(
                AMPBlock1(channels, kernel, dilations)
                for kernel, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            )

        self.activation_post = Activation1d(SnakeBeta(channels))
        self.conv_post = nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i][0](x)
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks).div_(self.num_kernels)
        x = self.activation_post(x)
        return self.conv_post(x).clamp_(-1.0, 1.0)


# --------------------------------------------------------------------- the codec


class AutoencoderKLH3Audio(nn.Module):
    """H3 audio codec: canonical ``[B, 32, 2, T]`` latents to samples in ``[-1, 1]``.

    A latent frame is 800 samples, so a clip's sound is small enough to decode in one
    pass — nothing like the video codec's chunking.
    """

    compute_dtype = COMPUTE_DTYPE

    z_dim = LATENT_CHANNELS
    sample_rate = SAMPLE_RATE
    output_sample_rate = SAMPLE_RATE
    samples_per_latent = SAMPLES_PER_LATENT
    latents_per_second = LATENTS_PER_SECOND

    def __init__(self, latent_channels: int = LATENT_CHANNELS, latent_dim: int = LATENT_DIM,
                 decoder_dim: int = DECODER_DIM, upsample_rates=UPSAMPLE_RATES,
                 upsample_kernel_sizes=UPSAMPLE_KERNELS,
                 resblock_kernel_sizes=RESBLOCK_KERNELS,
                 resblock_dilation_sizes=RESBLOCK_DILATIONS,
                 encoder_dim: int = ENCODER_DIM, encoder_rates=ENCODER_RATES,
                 attn_heads: int = ATTN_HEADS):
        super().__init__()
        self.stereo_lanes = STEREO_LANES
        self.encoder = Encoder(encoder_dim, tuple(encoder_rates), latent_dim)
        self.pre_block = AttnProjection(latent_dim, latent_channels, attn_heads)
        self.mean_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        # Trained, and never sampled: encode answers the posterior mean.
        self.logs_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.dec_in_proj = nn.Conv1d(latent_channels, latent_dim, 1)
        self.decoder = BigVGAN(num_mels=latent_dim, upsample_initial_channel=decoder_dim,
                               upsample_rates=upsample_rates,
                               upsample_kernel_sizes=upsample_kernel_sizes,
                               resblock_kernel_sizes=resblock_kernel_sizes,
                               resblock_dilation_sizes=resblock_dilation_sizes)
        self.register_buffer("latents_mean", torch.zeros(latent_channels), persistent=True)
        self.register_buffer("latents_std", torch.ones(latent_channels), persistent=True)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.decoder.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    def decode_output_length(self, latent_t: int) -> int:
        """Samples :meth:`decode_to_pcm` answers for ``latent_t`` latent frames."""
        return latent_t * self.decoder.upsample_factor

    def decode_to_pcm(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latent ``[B, 32, 2, T]`` -> ``[B, 2, L]`` in ``[-1, 1]``.

        The lanes fold into the batch: the vocoder is mono and each lane keeps its own
        copy of everything.
        """
        z = latents.to(self.device, self.dtype)
        if z.ndim != 4 or z.shape[1] != self.z_dim:
            raise ValueError(f"H3 audio expects a [B,{self.z_dim},{self.stereo_lanes},T] latent")
        if z.shape[2] != self.stereo_lanes:
            raise ValueError(
                f"H3 audio latents carry {self.stereo_lanes} stereo lanes, got {z.shape[2]}"
            )

        batch, channels, lanes, frames = z.shape
        z = z.permute(0, 2, 1, 3).reshape(batch * lanes, channels, frames)
        z = z * self.latents_std.view(1, -1, 1).to(z) + self.latents_mean.view(1, -1, 1).to(z)
        samples = self.decoder(self.dec_in_proj(z))
        return samples.reshape(batch, lanes, -1)

    def encode_to_latents(self, pcm, sample_rate: Optional[int] = None) -> torch.Tensor:
        """Samples in ``[-1, 1]`` -> the normalized canonical latent ``[B, 32, 2, T]``.

        The posterior *mean*: the checkpoint trains a spread that inference never samples,
        so a condition's latent does not move with the run's seed. Mono arrives as two equal
        lanes, and any other sample rate is moved here -- a latent frame is 800 samples at
        32 kHz, so the rate is part of the geometry rather than a preference.

        The tail is right-padded to a whole latent frame, so ``T`` is the ceiling of what
        was handed over.
        """
        samples = pcm_resample(pcm, SAMPLE_RATE if sample_rate is None else sample_rate,
                               self.sample_rate)
        if samples.shape[0] == 1:
            samples = np.repeat(samples, self.stereo_lanes, axis=0)
        x = torch.from_numpy(samples)[None].to(self.device, self.dtype)
        batch, lanes, length = x.shape
        frames = -(-length // self.samples_per_latent)
        x = F.pad(x, (0, frames * self.samples_per_latent - length))
        x = x.reshape(batch * lanes, 1, -1)
        x = self.pre_block(self.encoder(x).transpose(1, 2)).transpose(1, 2)
        z = self.mean_proj(x)
        z = ((z - self.latents_mean.view(1, -1, 1).to(z))
             / self.latents_std.view(1, -1, 1).to(z))
        return z.reshape(batch, lanes, self.z_dim, frames).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_to_pcm(x)


# --------------------------------------------------------------------- loader


#: What the audio checkpoint calls its own geometry inside ``__metadata__``.
AUDIO_GEOMETRY_SECTION = "minimax_h3_audio_vae"

#: The vocoder's own claim. An image or video codec has none of it, and the audio
#: encoder alone has none of the ``decoder.`` half either.
H3_AUDIO_SIGNATURE = Signature("H3 audio codec",
                               keys=("dec_in_proj.weight", "decoder.conv_post.weight",
                                     "decoder.resblocks.0.convs1.0.weight"))


def _audio_geometry(header: Header) -> dict:
    """The widths and rates the file states, falling back to the release's constants.

    A file whose ``dec_in_proj`` does not answer ``latent_dim`` channels is not this codec.
    """
    meta = header.metadata_json(AUDIO_GEOMETRY_SECTION)
    kwargs = meta.get("kwargs") or {}
    dec_in = header.shapes.get("dec_in_proj.weight")
    if dec_in is None:
        raise ValueError(f"{header.path}: 'dec_in_proj.weight' is missing "
                         "(not an H3 audio VAE?)")

    claimed_rate = int(meta.get("sample_rate", SAMPLE_RATE))
    if claimed_rate != SAMPLE_RATE:
        raise ValueError(
            f"{header.path}: a codec that samples at {claimed_rate} Hz would fill a latent "
            f"frame with {claimed_rate * SAMPLES_PER_LATENT // SAMPLE_RATE} samples of "
            f"sound, and {LATENTS_PER_SECOND} latents per second is the video's clock"
        )

    geometry = {
        "latent_channels": int(meta.get("latent_channels", dec_in[1])),
        "latent_dim": int(kwargs.get("latent_dim", dec_in[0])),
        "decoder_dim": int(kwargs.get("decoder_dim", DECODER_DIM)),
        "upsample_rates": tuple(kwargs.get("decoder_rates", UPSAMPLE_RATES)),
        "encoder_dim": int(kwargs.get("encoder_dim", ENCODER_DIM)),
        "encoder_rates": tuple(kwargs.get("encoder_rates", ENCODER_RATES)),
    }
    if math.prod(geometry["encoder_rates"]) != SAMPLES_PER_LATENT:
        raise ValueError(
            f"{header.path}: encoder rates {geometry['encoder_rates']} squash to "
            f"{math.prod(geometry['encoder_rates'])} samples per latent frame, not the "
            f"{SAMPLES_PER_LATENT} the vocoder restores"
        )
    if math.prod(geometry["upsample_rates"]) != SAMPLES_PER_LATENT:
        raise ValueError(
            f"{header.path}: decoder rates {geometry['upsample_rates']} restore "
            f"{math.prod(geometry['upsample_rates'])} samples per latent frame, not the "
            f"{SAMPLES_PER_LATENT} the encoder's rates squash and {LATENTS_PER_SECOND} "
            "latents per second assume; the latent grid and the sample rate would disagree"
        )
    geometry["upsample_kernel_sizes"] = UPSAMPLE_KERNELS
    return geometry


def load_h3_audio_vae(
    vae_path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> AutoencoderKLH3Audio:
    """Load the H3 audio VAE from ``vae_path``.

    The codec runs in FP32 (:attr:`AutoencoderKLH3Audio.compute_dtype`); Snake's
    exponentials and the sinc filters want the range, and the checkpoint ships FP32.
    """

    def construct(header: Header) -> AutoencoderKLH3Audio:
        geometry = _audio_geometry(header)
        logger.info("H3 audio VAE: %d-channel latent, %d-wide vocoder over %s",
                    geometry["latent_channels"], geometry["decoder_dim"],
                    "×".join(str(r) for r in geometry["upsample_rates"]))
        return AutoencoderKLH3Audio(**geometry)

    return load_codec(vae_path, device,
                      what="H3 audio VAE",
                      signature=H3_AUDIO_SIGNATURE,
                      construct=construct,
                      dtype=dtype)


__all__ = [
    "AutoencoderKLH3Audio",
    "load_h3_audio_vae",
    "LATENTS_PER_SECOND",
    "SAMPLES_PER_LATENT",
    "SAMPLE_RATE",
    "STEREO_LANES",
]
