# Copyright 2025 The Wan Team and The HuggingFace Team Authors.
# Independent TheNoise implementation of the Wan 2.2 video VAE architecture.
# SPDX-License-Identifier: Apache-2.0
"""Native causal Wan video VAE with model-specific compute precision."""
from __future__ import annotations

import logging
from collections import namedtuple
from types import SimpleNamespace
from typing import Callable, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from thenoise.utils.loader import load_dit
from thenoise.utils.wan_weights import wan_vae_key, wan_vae21_key
from thenoise.vae.wan22 import (
    COMPUTE_DTYPE as WAN22_COMPUTE_DTYPE,
    Wan22AttentionBlock, WAN22_LATENTS_MEAN, WAN22_LATENTS_STD,
)

logger = logging.getLogger(__name__)

CACHE_T = 2
DecoderOutput = namedtuple("DecoderOutput", "sample")
EncoderOutput = namedtuple("EncoderOutput", "latent_dist")

#: The Wan 2.1 16ch codec is not the 2.2 family: it predates the activation range
#: that forced BF16 there, and its released weights ship FP16.
WAN21_COMPUTE_DTYPE = torch.float16

# Wan 2.1's channelwise latent stats (same latent format as the 14B experts).
_WAN21_LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
_WAN21_LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


class _GaussianPosterior:
    def __init__(self, moments: Tensor):
        self.mean, self.logvar = moments.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)

    def mode(self) -> Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        # A CPU RNG must also work with a GPU-resident VAE.
        random_device = self.mean.device
        if generator is not None and generator.device.type == "cpu":
            random_device = torch.device("cpu")
        eps = torch.randn(self.mean.shape, generator=generator,
                          device=random_device, dtype=self.mean.dtype).to(self.mean.device)
        return self.mean + torch.exp(0.5 * self.logvar) * eps


class _AvgDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.factor_t, self.factor_s = factor_t, factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels:
            raise ValueError("invalid Wan VAE residual downsample channel ratio")
        self.out_channels = out_channels
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: Tensor) -> Tensor:
        pt = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pt, 0))
        b, c, t, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.view(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, self.out_channels, self.group_size, t // ft, h // fs, w // fs)
        return x.mean(dim=2)


class _DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.factor_t, self.factor_s = factor_t, factor_s
        n = out_channels * factor_t * factor_s * factor_s
        if n % in_channels:
            raise ValueError("invalid Wan VAE residual upsample channel ratio")
        self.out_channels = out_channels
        self.repeats = n // in_channels

    def forward(self, x: Tensor, first_chunk: bool = False) -> Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(x.size(0), self.out_channels, self.factor_t, self.factor_s,
                   self.factor_s, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * self.factor_t,
                   x.size(4) * self.factor_s, x.size(6) * self.factor_s)
        return x[:, :, self.factor_t - 1:] if first_chunk else x


class _CausalConv3D(nn.Conv3d):
    """Same parameter names and left-temporal-padding semantics as Wan."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._causal_padding = (self.padding[2], self.padding[2], self.padding[1],
                                self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x: Tensor, cache_x: Tensor | None = None) -> Tensor:
        pad = list(self._causal_padding)
        if cache_x is not None and pad[4]:
            x = torch.cat((cache_x.to(x.device), x), dim=2)
            pad[4] -= cache_x.shape[2]
        return F.conv3d(F.pad(x, pad), self.weight, self.bias, self.stride,
                        (0, 0, 0), self.dilation, self.groups)


class _RMSNorm(nn.Module):
    def __init__(self, channels: int, *, images: bool = True):
        super().__init__()
        self.scale = channels ** 0.5
        self.gamma = nn.Parameter(torch.ones((channels, 1, 1) if images else (channels, 1, 1, 1)))
        self.bias = 0.0

    def forward(self, x: Tensor) -> Tensor:
        norm = F.normalize(x.float() if x.dtype in (torch.float16, torch.bfloat16) else x, dim=1)
        return norm.to(x.dtype) * self.scale * self.gamma + self.bias


class _Upsample(nn.Upsample):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).to(x.dtype)


class _Resample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None):
        super().__init__()
        self.mode = mode
        od = upsample_out_dim if upsample_out_dim is not None else dim // 2
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(_Upsample(scale_factor=(2, 2), mode="nearest-exact"),
                                          nn.Conv2d(dim, od, 3, padding=1))
            if mode == "upsample3d":
                self.time_conv = _CausalConv3D(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            if mode == "downsample3d":
                self.time_conv = _CausalConv3D(dim, dim, (3, 1, 1), stride=(2, 1, 1))
        else:
            raise ValueError(f"unsupported Wan VAE resample mode: {mode}")

    def forward(self, x: Tensor, feat_cache: list | None = None, feat_idx: list[int] | None = None) -> Tensor:
        b, c, _, h, w = x.shape
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:].clone()
                old = feat_cache[idx]
                if cache_x.shape[2] < 2:
                    if isinstance(old, str):
                        cache_x = torch.cat((torch.zeros_like(cache_x), cache_x), dim=2)
                    else:
                        cache_x = torch.cat((old[:, :, -1:], cache_x), dim=2)
                x = self.time_conv(x) if isinstance(old, str) else self.time_conv(x, old)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                t = x.shape[2]
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), dim=3).reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
            else:
                cache_x = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat((feat_cache[idx][:, :, -1:], x), dim=2))
                feat_cache[idx] = cache_x
            feat_idx[0] += 1
        return x


def _cached_conv(conv: _CausalConv3D, x: Tensor, cache: list | None, idx: list[int] | None) -> Tensor:
    if cache is None:
        return conv(x)
    i = idx[0]
    last = x[:, :, -CACHE_T:].clone()
    if last.shape[2] < 2 and cache[i] is not None:
        last = torch.cat((cache[i][:, :, -1:], last), dim=2)
    out = conv(x, cache[i])
    cache[i] = last
    idx[0] += 1
    return out


class _ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.norm1 = _RMSNorm(in_dim, images=False)
        self.conv1 = _CausalConv3D(in_dim, out_dim, 3, padding=1)
        self.norm2 = _RMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _CausalConv3D(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = _CausalConv3D(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor, feat_cache=None, feat_idx=None) -> Tensor:
        shortcut = self.conv_shortcut(x)
        x = _cached_conv(self.conv1, F.silu(self.norm1(x)), feat_cache, feat_idx)
        x = _cached_conv(self.conv2, self.dropout(F.silu(self.norm2(x))), feat_cache, feat_idx)
        return x + shortcut


class _AttentionBlock(Wan22AttentionBlock):
    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frames = super().forward(frames)
        return frames.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class _MidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.resnets = nn.ModuleList([_ResidualBlock(dim, dim, dropout), _ResidualBlock(dim, dim, dropout)])
        self.attentions = nn.ModuleList([_AttentionBlock(dim)])

    def forward(self, x: Tensor, feat_cache=None, feat_idx=None) -> Tensor:
        x = self.resnets[0](x, feat_cache, feat_idx)
        return self.resnets[1](self.attentions[0](x), feat_cache, feat_idx)


class _ResidualDownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_res_blocks, *, temporal, down, residual=True):
        super().__init__()
        self.avg_shortcut = (_AvgDown3D(in_dim, out_dim, 2 if temporal else 1,
                                        2 if down else 1) if residual else None)
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(_ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = _Resample(out_dim, "downsample3d" if temporal else "downsample2d") if down else None

    def forward(self, x, feat_cache=None, feat_idx=None):
        original = x.clone() if self.avg_shortcut is not None else None
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache, feat_idx)
        return x + self.avg_shortcut(original) if original is not None else x


class _Encoder3D(nn.Module):
    def __init__(self, *, in_channels, dim, z_dim, dim_mult, num_res_blocks,
                 temporal_downsample, dropout, residual=True):
        super().__init__()
        dims = [dim * v for v in [1, *dim_mult]]
        self.conv_in = _CausalConv3D(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([
            _ResidualDownBlock(dims[i], dims[i + 1], dropout, num_res_blocks,
                               temporal=(temporal_downsample[i] if i < len(dim_mult) - 1 else False),
                               down=(i < len(dim_mult) - 1), residual=residual)
            for i in range(len(dim_mult))
        ])
        self.mid_block = _MidBlock(dims[-1], dropout)
        self.norm_out = _RMSNorm(dims[-1], images=False)
        self.conv_out = _CausalConv3D(dims[-1], z_dim, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=None):
        x = _cached_conv(self.conv_in, x, feat_cache, feat_idx)
        for block in self.down_blocks:
            x = block(x, feat_cache, feat_idx)
        x = self.mid_block(x, feat_cache, feat_idx)
        return _cached_conv(self.conv_out, F.silu(self.norm_out(x)), feat_cache, feat_idx)


class _ResidualUpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_res_blocks, dropout, *, temporal, up, residual=True):
        super().__init__()
        self.avg_shortcut = (_DupUp3D(in_dim, out_dim, 2 if temporal else 1, 2)
                             if up and residual else None)
        resnets = []
        for _ in range(num_res_blocks + 1):
            resnets.append(_ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsampler = (_Resample(out_dim, "upsample3d" if temporal else "upsample2d",
                                    out_dim if residual else out_dim // 2) if up else None)

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        shortcut = x.clone() if self.avg_shortcut is not None else None
        for block in self.resnets:
            x = block(x, feat_cache, feat_idx)
        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache, feat_idx)
        return x + self.avg_shortcut(shortcut, first_chunk=first_chunk) if self.avg_shortcut else x


class _Decoder3D(nn.Module):
    def __init__(self, *, dim, z_dim, dim_mult, num_res_blocks, temporal_upsample,
                 dropout, out_channels, residual=True):
        super().__init__()
        dims = [dim * v for v in [dim_mult[-1], *dim_mult[::-1]]]
        self.conv_in = _CausalConv3D(z_dim, dims[0], 3, padding=1)
        self.mid_block = _MidBlock(dims[0], dropout)
        self.up_blocks = nn.ModuleList([
            _ResidualUpBlock(dims[i] if residual or i == 0 else dims[i] // 2,
                             dims[i + 1], num_res_blocks, dropout,
                             temporal=(temporal_upsample[i] if i < len(dim_mult) - 1 else False),
                             up=(i < len(dim_mult) - 1), residual=residual)
            for i in range(len(dim_mult))
        ])
        self.norm_out = _RMSNorm(dims[-1], images=False)
        self.conv_out = _CausalConv3D(dims[-1], out_channels, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        x = _cached_conv(self.conv_in, x, feat_cache, feat_idx)
        x = self.mid_block(x, feat_cache, feat_idx)
        for block in self.up_blocks:
            x = block(x, feat_cache, feat_idx, first_chunk=first_chunk)
        return _cached_conv(self.conv_out, F.silu(self.norm_out(x)), feat_cache, feat_idx)


def _patchify(x: Tensor, p: int) -> Tensor:
    if p == 1:
        return x
    b, c, t, h, w = x.shape
    if h % p or w % p:
        raise ValueError("Wan video size must be divisible by VAE patch size")
    return (x.view(b, c, t, h // p, p, w // p, p)
            .permute(0, 1, 6, 4, 2, 3, 5).contiguous()
            .view(b, c * p * p, t, h // p, w // p))


def _unpatchify(x: Tensor, p: int) -> Tensor:
    if p == 1:
        return x
    b, cp, t, h, w = x.shape
    c = cp // (p * p)
    return (x.view(b, c, p, p, t, h, w)
            .permute(0, 1, 4, 5, 3, 6, 2).contiguous()
            .view(b, c, t, h * p, w * p))




def _comfy_vae_config(overrides: dict | None = None) -> SimpleNamespace:
    cfg = dict(
        base_dim=160, decoder_base_dim=256, z_dim=48,
        dim_mult=[1, 2, 4, 4], num_res_blocks=2,
        temperal_downsample=[False, True, True],
        dropout=0.0, is_residual=True, in_channels=12,
        out_channels=12, patch_size=2, scale_factor_temporal=4,
        scale_factor_spatial=16, attn_scales=[],
        compute_dtype=WAN22_COMPUTE_DTYPE,
        latents_mean=list(WAN22_LATENTS_MEAN), latents_std=list(WAN22_LATENTS_STD),
    )
    if overrides:
        cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _wan21_vae_config() -> dict:
    """Overrides turning the 2.2 config into the Wan 2.1 16ch codec."""
    return dict(base_dim=96, decoder_base_dim=96, z_dim=16,
                in_channels=3, out_channels=3, patch_size=1,
                temperal_downsample=[False, True, True],
                scale_factor_spatial=8, is_residual=False,
                compute_dtype=WAN21_COMPUTE_DTYPE,
                latents_mean=list(_WAN21_LATENTS_MEAN),
                latents_std=list(_WAN21_LATENTS_STD))


class WanNativeVideoVAE(nn.Module):
    """Independent causal 3D Wan video VAE with compatible encode/decode APIs.

    The 48ch 2.2 and 16ch 2.1 variants share causal building blocks. The cached decoder
    consumes latent frames individually, emitting one RGB frame for the first
    latent, then four for each subsequent latent. Cache state never crosses
    separate encode/decode calls. Precision is set at checkpoint load time and
    recorded on the config as ``compute_dtype``.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        mult = list(config.dim_mult)
        down = list(config.temperal_downsample)
        self.encoder = _Encoder3D(
            in_channels=config.in_channels, dim=config.base_dim, z_dim=config.z_dim * 2,
            dim_mult=mult, num_res_blocks=config.num_res_blocks,
            temporal_downsample=down, dropout=config.dropout, residual=config.is_residual,
        )
        self.quant_conv = _CausalConv3D(config.z_dim * 2, config.z_dim * 2, 1)
        self.post_quant_conv = _CausalConv3D(config.z_dim, config.z_dim, 1)
        self.decoder = _Decoder3D(
            dim=config.decoder_base_dim, z_dim=config.z_dim, dim_mult=mult,
            num_res_blocks=config.num_res_blocks, temporal_upsample=down[::-1],
            dropout=config.dropout, out_channels=config.out_channels,
            residual=config.is_residual,
        )
        self.clear_cache()
        self.use_tiling = False  # no silent fallback to unverified spatial tiles
        self.use_slicing = False

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.config.compute_dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def clear_cache(self):
        # Called at beginning/end of every API call; caches never leak across jobs.
        self._feat_map = [None] * sum(isinstance(m, _CausalConv3D) for m in self.decoder.modules())
        self._enc_feat_map = [None] * sum(isinstance(m, _CausalConv3D) for m in self.encoder.modules())

    def _onload_if_offloaded(self) -> None:
        # encode/decode bypass forward(), so trigger CPU-offload hooks here.
        hook = getattr(self, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(self)

    def encode(self, x: Tensor, return_dict: bool = True):
        if x.ndim != 5 or x.shape[1] != 3 or (x.shape[2] - 1) % 4:
            raise ValueError("Wan VAE expects [B,3,4*n+1,H,W] video")
        self._begin()
        try:
            x = _patchify(x.to(dtype=self.dtype), self.config.patch_size)
            moments = []
            for i, start in enumerate(range(0, x.shape[2], 4)):
                # First frame alone; remaining frames in groups of four.
                chunk = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1):1 + 4 * i]
                moments.append(self.encoder(chunk, self._enc_feat_map, [0]))
            h = self.quant_conv(torch.cat(moments, dim=2))
            posterior = _GaussianPosterior(h)
            return EncoderOutput(posterior) if return_dict else (posterior,)
        finally:
            self.clear_cache()

    def _begin(self) -> None:
        """Enter one encode/decode call: the module's place, then a clean cache."""
        if self.dtype not in (self.compute_dtype, torch.float32):
            # FP32 is always allowed: it cannot overflow, it only costs. What is
            # refused is a precision nobody measured this codec at.
            raise RuntimeError(
                f"Wan causal video VAE runs in {self.compute_dtype}, not {self.dtype}"
            )
        self._onload_if_offloaded()
        self.clear_cache()

    def decode(self, z: Tensor, return_dict: bool = True):
        out = torch.cat(list(self.decode_frames(z)), dim=2)
        return DecoderOutput(out) if return_dict else (out,)

    def decode_frames(self, z: Tensor) -> Iterator[Tensor]:
        """Pixels of each latent frame, handed out as the decoder produces them.

        The causal caches make the decoder sequential whatever it is asked to do, so
        yielding costs nothing and keeps a few frames alive instead of the clip: a
        caller that encodes as it goes never allocates ``[3, T, H, W]`` at all.
        """
        if z.ndim != 5 or z.shape[1] != self.config.z_dim or z.shape[2] < 1:
            raise ValueError(f"Wan VAE expects [B,{self.config.z_dim},T,H,W] latent video")
        self._begin()
        return self._decode_frames(z)

    def _decode_frames(self, z: Tensor) -> Iterator[Tensor]:
        try:
            x = self.post_quant_conv(z.to(dtype=self.dtype))
            for i in range(x.shape[2]):
                chunk = self.decoder(x[:, :, i:i + 1], self._feat_map, [0],
                                     first_chunk=(i == 0))
                yield _unpatchify(chunk, self.config.patch_size).clamp(-1.0, 1.0)
        finally:
            self.clear_cache()

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x).latent_dist.mode()).sample



def load_wan_native_video_vae(
    vae_path: str,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    *,
    wan21: bool = False,
    config: SimpleNamespace | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> WanNativeVideoVAE:
    """Load a Wan video VAE from ``vae_path``.

    The file is read with its own dtypes and the module then moved to ``dtype``, so
    the read never holds a second copy of the codec; ``dtype`` defaults to the
    codec's own ``compute_dtype``. ``config`` replaces the geometry, which every
    released file shares.
    """
    cfg = config or (_comfy_vae_config(_wan21_vae_config()) if wan21 else _comfy_vae_config())
    key_map = ((lambda key: wan_vae21_key(key, num_res_blocks=cfg.num_res_blocks)) if wan21
               else (lambda key: wan_vae_key(key, num_res_blocks=cfg.num_res_blocks)))
    logger.info("Loading Wan %s video VAE from %s", "2.1" if wan21 else "2.2", vae_path)
    with torch.device("meta"):
        model = WanNativeVideoVAE(cfg)
    load_dit(model, vae_path, device=device, dtype=None, key_map=key_map,
             check_cancel=check_cancel)
    model.to(dtype=dtype or cfg.compute_dtype)
    model.clear_cache()
    return model.eval()
