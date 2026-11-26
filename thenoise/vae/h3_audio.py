"""H3 audio codec — a DAC-lineage waveform encoder behind a BigVGAN vocoder.

Ported from ComfyUI's ``comfy/ldm/minimax/audio_vae.py`` with its module tree — and so its
checkpoint key names — verbatim; only the ops differ. The released file already has its
weight-norm parametrizations folded into plain weights.

Only the vocoder half is built: a clip's sound is decoded, never re-encoded, so
:func:`load_h3_audio_vae` drops the encoder's tensors instead of holding them.

The latent is ``[B, 32, 2, T]`` — 32 channels over 2 stereo lanes at 40 frames per second.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.utils.loader import load_codec
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

#: The encoder-side tensors a decode never reads, dropped at load rather than built.
ENCODER_PREFIXES = ("encoder.", "pre_block.", "mean_proj.", "logs_proj.")


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
                 resblock_dilation_sizes=RESBLOCK_DILATIONS):
        super().__init__()
        self.stereo_lanes = STEREO_LANES
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

    geometry = {
        "latent_channels": int(meta.get("latent_channels", dec_in[1])),
        "latent_dim": int(kwargs.get("latent_dim", dec_in[0])),
        "decoder_dim": int(kwargs.get("decoder_dim", DECODER_DIM)),
        "upsample_rates": tuple(kwargs.get("decoder_rates", UPSAMPLE_RATES)),
    }
    if math.prod(geometry["upsample_rates"]) != SAMPLES_PER_LATENT:
        raise ValueError(
            f"{header.path}: decoder rates {geometry['upsample_rates']} restore "
            f"{math.prod(geometry['upsample_rates'])} samples per latent frame, not the "
            f"{SAMPLES_PER_LATENT} the encoder's rates squash and {LATENTS_PER_SECOND} "
            "latents per second assume; the latent grid and the sample rate would disagree"
        )
    geometry["upsample_kernel_sizes"] = UPSAMPLE_KERNELS
    return geometry


def _audio_state(state_dict: dict, model: torch.nn.Module) -> dict:
    """Drop the encoder half, which this module does not build."""
    return {k: v for k, v in state_dict.items()
            if not k.startswith(ENCODER_PREFIXES)}


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
                      state_map=_audio_state,
                      dtype=dtype)


__all__ = [
    "AutoencoderKLH3Audio",
    "load_h3_audio_vae",
    "LATENTS_PER_SECOND",
    "SAMPLES_PER_LATENT",
    "SAMPLE_RATE",
    "STEREO_LANES",
]
