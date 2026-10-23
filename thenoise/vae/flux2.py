# Flux.2 VAE, ported from kohya-ss/musubi-tuner's
# ``flux2_models.AutoEncoder`` (itself a copy of the Black Forest Labs FLUX repo,
# Apache-2.0).
#
# Flux.2 latents are *packed*: a 32-channel latent packed 2x2 into 128 channels
# (16x spatial compression in packed space), normalized via BatchNorm. The DiT
# denoises that normalized 128-channel packed latent, so the canonical latent here
# is ``[B, 128, H//16, W//16]``. Decode un-normalizes (``z*sqrt(var) + mean``),
# unpacks to ``[B, 32, H//8, W//8]``, runs the decoder, and clamps to [-1, 1];
# encode is the inverse.

# Copyright 2023 The HuggingFace Team. Licensed under the Apache-2.0 License.
# Copyright 2024 Black Forest Labs. Flux is released under the Apache-2.0 License.
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.utils.attention import single_head_attention
from .fused import (
    FusedResnetMixin,
    fused_available,
    group_norm_only,
    interleaved_view,
    pointwise,
    region,
    strip_conv,
    strip_down,
)

from thenoise.utils.safetensors import load_safetensors
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


#: The two halves' shape, stated once. ``_CH_MULT`` scales a base width per stage, so
#: the Flux.2 releases differ only in that base: every checkpoint *encodes* at 128
#: (stages 128/256/512/512), while BFL's distilled `FLUX.2-small-decoder
#: <https://huggingface.co/black-forest-labs/FLUX.2-small-decoder>`_ *decodes* at 96
#: (96/192/384/384) — ~28M decoder parameters instead of ~50M, ~1.4x faster decode and
#: ~1.4x less decode memory, at bit-identical latents and the same encoder. Point
#: ``--vae`` at either file: the loader reads the width from the weights.
_CH_MULT = (1, 2, 4, 4)
_Z_CHANNELS = 32  # raw latent channels, 2x2-packed into the canonical 128
_PIXEL_CHANNELS = 3
ENCODER_CH = 128
FULL_DECODER_CH = 128
SMALL_DECODER_CH = 96
#: The widths that have shipped, for the log line.
_DECODER_VARIANTS = {FULL_DECODER_CH: "full", SMALL_DECODER_CH: "small"}


def swish(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def _patchify(z: torch.Tensor) -> torch.Tensor:
    """32ch -> 128ch via 2x2 spatial packing."""
    B, C, H, W = z.shape
    return (
        z.float()
        .reshape(B, C, H // 2, 2, W // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(B, C * 4, H // 2, W // 2)
    )


def _unpatchify(z: torch.Tensor) -> torch.Tensor:
    """128ch -> 32ch via 2x2 spatial unpacking."""
    B, C, H, W = z.shape
    return (
        z.float()
        .reshape(B, C // 4, 2, 2, H, W)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(B, C // 4, H * 2, W * 2)
    )


class _AttnBlock(nn.Module):
    def __init__(self, in_channels: int, fused: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.fused = fused
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: torch.Tensor, fused: bool = False) -> torch.Tensor:
        h_ = group_norm_only(h_, self.norm) if fused else self.norm(h_)
        q = pointwise(self.q, h_) if fused else self.q(h_)
        k = pointwise(self.k, h_) if fused else self.k(h_)
        v = pointwise(self.v, h_) if fused else self.v(h_)
        b, c, h, w = q.shape
        # ``flatten(2)`` and ``view(b, h, w, c)``, both ends free for the channels-last
        # image the fused path feeds this block.
        q = q.flatten(2).transpose(1, 2).unsqueeze(1)
        k = k.flatten(2).transpose(1, 2).unsqueeze(1)
        v = v.flatten(2).transpose(1, 2).unsqueeze(1)
        h_ = single_head_attention(q, k, v)
        return h_.squeeze(1).view(b, h, w, c).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.fused and fused_available(x.device, x.dtype)
        out = self.attention(x, fused)
        return x + (pointwise(self.proj_out, out) if fused else self.proj_out(out))


class _ResnetBlock(FusedResnetMixin, nn.Module):
    def __init__(self, in_channels: int, out_channels: int, fused: bool = False):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fused = fused

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.fused and fused_available(x.device, x.dtype):
            return self.fused_forward(x, hidden)
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class _Upsample(nn.Module):
    def __init__(self, in_channels: int, fused: bool = False):
        super().__init__()
        self.fused = fused
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, out: Optional[torch.Tensor] = None):
        if self.fused and fused_available(x.device, x.dtype):
            return strip_conv(self.conv, x, scale=2, out=out)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class _Downsample(nn.Module):
    def __init__(self, in_channels: int, fused: bool = False):
        super().__init__()
        self.fused = fused
        # No asymmetric padding in torch conv, so pad manually then stride-2 conv.
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        if self.fused and fused_available(x.device, x.dtype):
            return strip_down(self.conv, x)
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class _Encoder(nn.Module):
    """Flux.2 encoder: pixels -> 32ch raw latent (8x spatial down), then quant_conv."""

    def __init__(
        self,
        ch: int = ENCODER_CH,
        in_channels: int = _PIXEL_CHANNELS,
        ch_mult: tuple[int, ...] = _CH_MULT,
        num_res_blocks: int = 2,
        z_channels: int = _Z_CHANNELS,
        fused: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.fused = fused

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = None
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(_ResnetBlock(in_channels=block_in, out_channels=block_out, fused=fused))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn  # attention only in the mid block
            if i_level != self.num_resolutions - 1:
                down.downsample = _Downsample(block_in, fused=fused)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(in_channels=block_in, out_channels=block_in, fused=fused)
        self.mid.attn_1 = _AttnBlock(block_in, fused=fused)
        self.mid.block_2 = _ResnetBlock(in_channels=block_in, out_channels=block_in, fused=fused)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)
        # The 2*z_channels (mean+logvar) output; the caller takes the gaussian mean.
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, kernel_size=1)

    @property
    def dtype(self):
        return next(self.down.parameters()).dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.fused and fused_available(x.device, x.dtype)
        if fused:
            x = x.contiguous(memory_format=torch.channels_last)
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        if fused:
            h = strip_conv(self.conv_out, h, self.norm_out)
        else:
            h = self.conv_out(swish(self.norm_out(h)))
        return self.quant_conv(h)  # [B, 2*z_channels, H//8, W//8]


class _Decoder(nn.Module):
    """Flux.2 decoder: z -> pixels (32ch raw -> 3ch, 8x spatial up)."""

    def __init__(
        self,
        ch: int = FULL_DECODER_CH,
        out_ch: int = _PIXEL_CHANNELS,
        ch_mult: tuple[int, ...] = _CH_MULT,
        num_res_blocks: int = 2,
        z_channels: int = _Z_CHANNELS,
        fused: bool = False,
    ):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.fused = fused
        self.in_channels = z_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(in_channels=block_in, out_channels=block_in, fused=fused)
        self.mid.attn_1 = _AttnBlock(block_in, fused=fused)
        self.mid.block_2 = _ResnetBlock(in_channels=block_in, out_channels=block_in, fused=fused)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(_ResnetBlock(in_channels=block_in, out_channels=block_out, fused=fused))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in, fused=fused)
            self.up.insert(0, up)  # prepend to keep consistent order

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    @property
    def dtype(self):
        return next(self.up.parameters()).dtype

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.fused and fused_available(z.device, z.dtype):
            # The shared buffer works on one image, so a batch is decoded one at a time.
            out = [self._forward_fused(z[i : i + 1]) for i in range(z.shape[0])]
            return out[0] if len(out) == 1 else torch.cat(out)

        z = self.post_quant_conv(z)
        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = h.to(upscale_dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        return self.conv_out(h)

    def _forward_fused(self, z: torch.Tensor) -> torch.Tensor:
        """One image, on one buffer the size of the largest upsample output.

        A level scratches in the region just before the residual stream it reads at the
        end of the buffer, and its upsample writes the next level over that end: four
        elements produced per element consumed, from further back, so the write front
        never catches the read front. A block halving the channels cannot use a region —
        at the last level the buffer holds nothing but the stream — and keeps hidden
        state and result inside it instead
        (:meth:`FusedResnetMixin.halves_in_place`).
        """
        levels = self.num_resolutions
        h = self.conv_in(self.post_quant_conv(z.contiguous(memory_format=torch.channels_last)))
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        buf = torch.empty(
            max(
                self.up[i].block[-1].out_channels
                * (h.shape[2] << (levels - i))
                * (h.shape[3] << (levels - i))
                for i in range(1, levels)
            ),
            dtype=h.dtype,
            device=h.device,
        )
        for i_level in reversed(range(levels)):
            level = self.up[i_level]
            # One scratch region per level: only one block's hidden state is live at a time.
            hidden = None
            for i_block, block in enumerate(level.block):
                if hidden is None:
                    if block.halves_in_place(h):
                        hidden = interleaved_view(h, block.out_channels, 0)
                    else:
                        hidden = region(
                            buf,
                            buf.numel() - (h.numel() if i_level < levels - 1 else 0),
                            block.out_channels,
                            h.shape[2],
                            h.shape[3],
                        )
                h = block(h, hidden)
                if len(level.attn) > 0:
                    h = level.attn[i_block](h)
            if i_level != 0:
                h = level.upsample(
                    h, out=region(buf, buf.numel(), h.shape[1], 2 * h.shape[2], 2 * h.shape[3])
                )
        return strip_conv(self.conv_out, h, self.norm_out)


class AutoencoderKLFlux2(nn.Module):
    """Flux.2 VAE (encoder + decoder), with musubi/Flux key naming.

    Accepts/produces the canonical packed latent ``[B, 128, H//16, W//16]``
    (normalized); pixels are in [-1, 1] both ways. The latent is the same whichever
    decoder is attached, so a small-decoder checkpoint drops into any Flux.2 model.
    """

    z_dim = 128  # packed latent channels
    spatial_compression = 16  # pixel / packed-latent ratio
    pixel_channels = _PIXEL_CHANNELS
    bn_eps = 1e-4

    #: This AE computes in FP16 whatever the DiT next to it runs in: the latents are
    #: BatchNorm-normalised, so nothing here leans on BF16's range while a conv of this
    #: depth does lean on the mantissa, and it is the only dtype the fused kernels serve
    #: on ROCm (see :func:`~thenoise.vae.fused.fused_available`).
    compute_dtype = torch.float16

    def __init__(
        self,
        channels: int = 128,
        encoder_ch: int = ENCODER_CH,
        decoder_ch: int = FULL_DECODER_CH,
        fused: bool = True,
    ):
        """``channels`` is the packed latent width; the rest are the halves' base widths.

        Both widths are arguments because the releases differ (see :data:`_CH_MULT`):
        :func:`load_flux2_vae` reads them off the checkpoint, so naming them by hand is
        only for building one from scratch — or a cheap one, in the tests.

        ``fused`` asks every stage to run through :mod:`thenoise.vae.fused`, each still
        falling back to its plain path wherever those kernels are not served.
        """
        super().__init__()
        self.encoder_ch = encoder_ch
        self.decoder_ch = decoder_ch
        self.bn = nn.BatchNorm2d(channels, eps=self.bn_eps, momentum=0.1, affine=False, track_running_stats=True)
        self.encoder = _Encoder(ch=encoder_ch, fused=fused)
        self.decoder = _Decoder(ch=decoder_ch, fused=fused)

    @property
    def dtype(self):
        return next(self.decoder.parameters()).dtype

    @property
    def device(self):
        return next(self.decoder.parameters()).device

    def inv_normalize(self, z: torch.Tensor) -> torch.Tensor:
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Raw packed latent -> normalized canonical latent (``(z - mean) / sqrt(var)``).

        Only the BN's running stats are read; its ``forward`` never runs.
        """
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return (z.float() - m) / s

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels (``[B, C, H, W]`` in [-1, 1]) -> canonical packed latent."""
        z = self.encoder(pixels.to(device=self.device, dtype=self.encoder.dtype))  # [B, 2*z_channels, H//8, W//8]
        z = z.chunk(2, dim=1)[0]  # gaussian mean (eval mode) -> [B, z_channels, ...]
        z = _patchify(z)  # [B, 32, H//8, W//8] -> [B, 128, H//16, W//16]
        return self.normalize(z)

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the canonical packed latent to pixels in [-1, 1]."""
        z = self.inv_normalize(latents.float())  # undo BN -> raw packed 128ch
        z = _unpatchify(z).to(self.decoder.dtype)  # [B, 32, H//8, W//8]
        image = self.decoder(z)
        return image.clamp(-1.0, 1.0)


_HALF_WIDTHS = {"encoder": (1, _PIXEL_CHANNELS), "decoder": (_CH_MULT[-1], _Z_CHANNELS)}


def _half_width(state_dict: dict, half: str, vae_path: str) -> int:
    """Base channel width of one half of ``state_dict`` — 96 for a small decoder, 128 otherwise.

    Each half's ``conv_in`` states it, so no config file is needed and the strict load
    right after checks the rest of the layout. Requiring the input channels — 3 pixels,
    32 raw latent — is what separates a Flux.2 AE from any VAE that happens to carry a
    ``decoder.conv_in.weight``.
    """
    scale, in_channels = _HALF_WIDTHS[half]
    key = f"{half}.conv_in.weight"
    weight = state_dict.get(key)
    if weight is None or weight.dim() != 4 or weight.shape[1] != in_channels:
        raise ValueError(
            f"'{key}' taking {in_channels} channels on its input is missing from "
            f"{vae_path} (not a Flux.2 VAE?)"
        )
    if weight.shape[0] % scale:
        raise ValueError(
            f"{vae_path}: a {weight.shape[0]}-channel {half} input is not a scaled "
            f"{list(_CH_MULT)} stage layout (not a Flux.2 {half}?)"
        )
    return weight.shape[0] // scale


def load_flux2_vae(
    vae_path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> AutoencoderKLFlux2:
    """Load the Flux.2 VAE weights from ``vae_path``.

    The AE runs in its own :attr:`AutoencoderKLFlux2.compute_dtype`; ``dtype`` is an
    override, not a request for FP32.
    """
    device = torch.device(device)
    if dtype is None:
        dtype = AutoencoderKLFlux2.compute_dtype
    logger.info("Loading Flux.2 VAE from %s", vae_path)
    state_dict = load_safetensors(vae_path, device=device)

    keep = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("encoder.") or k.startswith("decoder.") or k.startswith("bn.")
    }
    if not keep:
        raise ValueError(f"No 'encoder.*'/'decoder.*'/'bn.*' keys found in {vae_path} (not a Flux.2 VAE?)")

    encoder_ch = _half_width(keep, "encoder", vae_path)
    decoder_ch = _half_width(keep, "decoder", vae_path)
    vae = AutoencoderKLFlux2(encoder_ch=encoder_ch, decoder_ch=decoder_ch)
    info = vae.load_state_dict(keep, strict=True, assign=True)
    logger.info(
        "Loaded Flux.2 VAE (%s %d-channel decoder): %s",
        _DECODER_VARIANTS.get(decoder_ch, "unknown"),
        decoder_ch,
        info,
    )

    vae.to(device, dtype)
    return vae


__all__ = [
    "AutoencoderKLFlux2",
    "load_flux2_vae",
    "FULL_DECODER_CH",
    "SMALL_DECODER_CH",
]
