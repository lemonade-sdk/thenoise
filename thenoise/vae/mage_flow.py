"""Mage-VAE — a symmetric *one-step diffusion codec*.

Ported from ComfyUI's ``comfy/ldm/mage_flow/vae.py`` (MIT), which is the microsoft/Mage
codec. Only the ops change (``comfy.ops`` -> ``torch.nn``, ComfyUI's attention helper ->
local SDPA); the module tree, and therefore the checkpoint key names, are verbatim.

Both directions are a single forward at ``t = 0``:

  * **encode** — ``DConvEncoder`` predicts the latent of an image in one step, with a
    *zero latent* fed in as ``z_t`` and the image as the conditioning. Its ``proj_out``
    is 256-wide (mean + logvar) and the mean is taken, so encode is deterministic.
  * **decode** — ``DConvDenoiser`` predicts the image in one step, with a *zero noise*
    image and the latent injected through ``CoDDecoder`` (``y_embedder.decoder``), plus
    a Nerf/DCT patch-position embedding on the pixel path.

The latent is **128 channels at 16x**, used raw: no scaling, no shift, no BatchNorm, no
2x2 packing.

The export also ships a Flux.2 encoder — the codec whose latent space this one was
anchored on during training — under ``pipeline.y_embedder.encoder.*``. Nothing in the
decode path uses it, so ``load_mage_vae`` drops it.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.utils.safetensors import MemoryEfficientSafeOpen, load_safetensors
from thenoise.utils.setup_logging import setup_logging
from .flux2 import AutoencoderKLFlux2, load_flux2_vae

setup_logging()
import logging

logger = logging.getLogger(__name__)


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def Normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation, broadcast over space for a 4D tensor and over tokens for 3D."""
    if x.dim() == 4:
        b, c = x.shape[:2]
        return x * (1 + scale.view(b, c, 1, 1)) + shift.view(b, c, 1, 1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-head attention over the token axis: ``[B, C, L] -> [B, C, L]``.

    Written out as matmul + softmax rather than ``F.scaled_dot_product_attention``:
    the fused ROCm SDPA backends are unreliable inside VAE decoders (broken pixels on
    gfx1151, a hard ``profiler is not initialized`` error on others). The window is
    32x32 tokens, so the explicit score matrix is not a memory concern.
    """
    b, c, length = q.shape
    q, k, v = (t.view(b, 1, length, c) for t in (q, k, v))
    attn = (q @ k.transpose(-2, -1)) / (c**0.5)  # SDPA's default 1/sqrt(head_dim)
    attn = attn.softmax(dim=-1)
    return (attn @ v).transpose(1, 2).reshape(b, c, length)


class LayerNorm2d(nn.LayerNorm):
    """Channel-last LayerNorm on a ``[B, C, H, W]`` tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class TimestepEmbedder(nn.Module):
    """DConv-style timestep MLP (``max_period=10000``, 256 frequencies)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(dtype))


class BottleneckPatchEmbed(nn.Module):
    """Image patch embed concatenated with a per-patch conditioning vector."""

    def __init__(self, patch_size=16, in_chans=3, pca_dim=128, embed_dim=384, bias=True):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim + embed_dim, embed_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.proj2(torch.cat([self.proj1(x), cond], dim=1))


class DiCoBlock(nn.Module):
    """DConv block: pointwise/depthwise conv mix + channel attention, adaLN-modulated."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )

        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn, hidden_size, 1, bias=True)

        self.norm1 = LayerNorm2d(hidden_size, affine=False)
        self.norm2 = LayerNorm2d(hidden_size, affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, inp: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(inp), shift_msa, scale_msa)
        x = F.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)
        x = inp + gate_msa[..., None, None] * x
        x = x + gate_mlp[..., None, None] * self.conv5(
            F.gelu(self.conv4(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        )
        return x


class EncoderDiCoBlock(nn.Module):
    """The same block without adaLN modulation, for the encoder's image head."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size, bias=True)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        )
        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn, hidden_size, 1, bias=True)
        self.norm1 = LayerNorm2d(hidden_size)
        self.norm2 = LayerNorm2d(hidden_size)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = F.gelu(self.conv2(self.conv1(x)))
        x = x * self.ca(x)
        x = self.conv3(x)
        x = inp + x
        return x + self.conv5(F.gelu(self.conv4(self.norm2(x))))


class NerfEmbedder(nn.Module):
    """Patch-position embedder: patch pixels plus a fixed DCT basis of their positions."""

    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True),
        )

    def fetch_pos(self, patch_size: int, device, dtype) -> torch.Tensor:
        pos = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos, pos, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        freqs = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
        fx = freqs[None, :, None]
        fy = freqs[None, None, :]
        coeffs = (1 + fx * fy) ** -1
        dct_x = torch.cos(pos_x * fx * torch.pi)
        dct_y = torch.cos(pos_y * fy * torch.pi)
        return (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _p2, _ = x.shape
        ps = int(x.shape[1] ** 0.5)
        dct = self.fetch_pos(ps, x.device, x.dtype).expand(b, -1, -1)
        return self.embedder(torch.cat([x, dct], dim=-1))


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class MLPResBlock(nn.Module):
    """Token-space residual MLP with adaLN conditioning (the per-patch decoder path)."""

    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = self.in_ln(x) * (1 + scale) + shift
        return x + gate * self.mlp(h)


class SimpleMLPAdaLN(nn.Module):
    """Per-patch MLP that turns Nerf features into pixel features, conditioned on ``s``."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        patch_size: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.patch_size = patch_size

        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList(MLPResBlock(model_channels) for _ in range(num_res_blocks))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        c = self.cond_embed(c).reshape(c.shape[0], self.patch_size**2, -1)
        for block in self.res_blocks:
            x = block(x, c)
        return x


class ResnetBlock(nn.Module):
    """GroupNorm + conv ResBlock (the CoD decoder's spatial backbone)."""

    def __init__(self, *, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(nonlinearity(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Self-attention restricted to ``patch_size x patch_size`` windows.

    A map whose sides are not multiples of the window is replicate-padded and
    cropped again.
    """

    def __init__(self, in_channels: int, patch_size: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        d = self.patch_size
        b, c, H, W = q.shape
        pad_h = (d - H % d) % d
        pad_w = (d - W % d) % d
        if pad_h or pad_w:
            q = F.pad(q, (0, pad_w, 0, pad_h), mode="replicate")
            k = F.pad(k, (0, pad_w, 0, pad_h), mode="replicate")
            v = F.pad(v, (0, pad_w, 0, pad_h), mode="replicate")
        _, _, H_pad, W_pad = q.shape
        nph, npw = H_pad // d, W_pad // d
        np_ = nph * npw

        def to_patches(t: torch.Tensor) -> torch.Tensor:
            return (
                t.reshape(b, c, nph, d, npw, d).permute(0, 2, 4, 1, 3, 5).reshape(b * np_, c, d * d)
            )

        h_ = _attention(to_patches(q), to_patches(k), to_patches(v))
        h_ = h_.reshape(b, nph, npw, c, d, d).permute(0, 3, 1, 4, 2, 5).reshape(b, c, H_pad, W_pad)
        if pad_h or pad_w:
            h_ = h_[:, :, :H, :W]
        return x + self.proj_out(h_)


class CoDDecoder(nn.Module):
    """CoD decoder: latent -> the conditioning features the denoiser is modulated by."""

    def __init__(self, out_ch: int = 384, z_ch: int = 128):
        super().__init__()
        self.conv_in = nn.Conv2d(z_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.block = nn.Sequential(
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
        )
        self.norm_out = Normalize(out_ch)
        self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.block(self.conv_in(z))
        return self.conv_out(nonlinearity(self.norm_out(h)))


class DConvEncoder(nn.Module):
    """Encoder: one-step prediction of the (mean, logvar) latent of an image."""

    def __init__(
        self,
        z_ch: int = 128,
        hidden_size: int = 384,
        num_blocks: int = 21,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        head_size: int = 768,
        num_head_blocks: int = 2,
        out_ch_mult: int = 2,
    ):
        super().__init__()
        self.z_ch = z_ch
        self.patch_size = patch_size
        self.patch_cond_embed = nn.Conv2d(3, head_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.head_blocks = nn.ModuleList(
            [EncoderDiCoBlock(head_size, mlp_ratio=mlp_ratio) for _ in range(num_head_blocks)]
        )
        self.proj_down = nn.Conv2d(head_size, hidden_size, kernel_size=1, bias=True)
        self.z_proj = nn.Conv2d(z_ch, hidden_size, kernel_size=1, bias=True)
        self.fuse_proj = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_blocks)]
        )
        self.norm_out = LayerNorm2d(hidden_size)
        self.proj_out = nn.Conv2d(hidden_size, z_ch * out_ch_mult, kernel_size=1, bias=True)

    def forward_pred(self, z_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict from the (zero) latent ``z_t`` at time ``t``, conditioned on image ``y``.

        Returns ``[B, 2*z_ch, h, w]``: the posterior mean and log-var side by side.
        """
        cond = self.patch_cond_embed(y)
        for block in self.head_blocks:
            cond = block(cond)
        cond = self.proj_down(cond)

        s = self.fuse_proj(torch.cat([cond, self.z_proj(z_t)], dim=1))
        c = self.t_embedder(t.view(-1), y.dtype)
        for block in self.blocks:
            s = block(s, c)
        return self.proj_out(self.norm_out(s))


class YEmbedder(nn.Module):
    """The decoder's conditioning branch."""

    def __init__(self, ch: int = 384, z_ch: int = 128):
        super().__init__()
        self.decoder = CoDDecoder(out_ch=ch, z_ch=z_ch)


class DConvDenoiser(nn.Module):
    """One-step denoiser: latent (via the CoD conditioning) + zero noise -> image."""

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 384,
        hidden_size_x: int = 32,
        mlp_ratio: float = 4.0,
        num_blocks: int = 24,
        num_cond_blocks: int = 21,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_cond_blocks = num_cond_blocks

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder_x = nn.Conv2d(hidden_size, hidden_size_x * patch_size**2, 1, 1, 0)
        self.x_embedder = NerfEmbedder(in_channels + hidden_size_x, hidden_size_x, max_freqs=8)
        self.s_embedder = BottleneckPatchEmbed(patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)
        self.blocks = nn.ModuleList(
            [DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_cond_blocks)]
        )
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_blocks - num_cond_blocks,
            patch_size=patch_size,
        )
        self.final_layer = NerfFinalLayer(hidden_size_x, in_channels)
        self.y_embedder = YEmbedder(ch=hidden_size, z_ch=bottleneck_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        c = self.t_embedder(t.view(-1), x.dtype)

        s = self.s_embedder(x, cond)
        for block in self.blocks:
            s = block(s, c)

        length = s.shape[-2] * s.shape[-1]
        s = s.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)

        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = torch.cat([x, self.y_embedder_x(cond).flatten(2)], dim=1)
        x = x.reshape(b, -1, self.patch_size**2, length).permute(0, 3, 2, 1).flatten(0, 1)
        x = self.x_embedder(x)

        x = self.dec_net(x, s)
        x = self.final_layer(x)
        x = x.transpose(1, 2).reshape(b, length, -1)
        return F.fold(
            x.transpose(1, 2).contiguous(),
            (h, w),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )


class AutoencoderKLMageFlow(nn.Module):
    """Mage-VAE: the 128ch/16x raw-latent codec, encoder + decoder in one module.

    ``encode_pixels_to_latents`` takes pixels ``[B, C, H, W]`` in [-1, 1] and returns
    the canonical latent ``[B, 128, H/16, W/16]``; ``decode_to_pixels`` is its inverse
    and clamps to [-1, 1].
    """

    z_dim = 128
    spatial_compression = 16
    pixel_channels = 3

    def __init__(self):
        super().__init__()
        self.dconv_encoder = DConvEncoder()
        self.decoder_model = DConvDenoiser()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.decoder_model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.decoder_model.parameters()).device

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels ``[B, C, H, W]`` in [-1, 1] -> canonical latent (posterior mean)."""
        x = pixels.to(device=self.device, dtype=self.dtype)
        b, _, H, W = x.shape
        ps = self.dconv_encoder.patch_size
        # The one-step prediction is conditioned on a zero latent.
        z_t = torch.zeros(b, self.dconv_encoder.z_ch, H // ps, W // ps, device=x.device, dtype=x.dtype)
        t = torch.zeros(b, device=x.device, dtype=x.dtype)
        out = self.dconv_encoder.forward_pred(z_t, t, x)
        return out[:, : self.z_dim]  # posterior mean

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Canonical latent -> pixels ``[B, 3, H, W]`` in [-1, 1]."""
        z = latents.to(device=self.device, dtype=self.dtype)
        cond = self.decoder_model.y_embedder.decoder(z)
        b = z.shape[0]
        H = z.shape[2] * self.spatial_compression
        W = z.shape[3] * self.spatial_compression
        noise = torch.zeros(b, 3, H, W, device=z.device, dtype=z.dtype)
        t = torch.zeros(b, device=z.device, dtype=z.dtype)
        return self.decoder_model(noise, t, cond).clamp(-1.0, 1.0)


def load_mage_vae(
    vae_path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> AutoencoderKLMageFlow:
    """Load the Mage-VAE, remapping the export's training names onto the module tree.

    The file is a training artifact: the encoder lives under ``student.dconv_encoder.*``
    and the denoiser under ``pipeline.*``.
    """
    device = torch.device(device)
    logger.info("Loading Mage-VAE from %s", vae_path)
    state_dict = load_safetensors(vae_path, device=device)

    remapped = {}
    for key, tensor in state_dict.items():
        if key.startswith("student.dconv_encoder."):
            remapped[key[len("student.") :]] = tensor
        elif key.startswith("pipeline.y_embedder.encoder."):
            continue  # the anchor Flux.2 encoder
        elif key.startswith("pipeline."):
            remapped["decoder_model." + key[len("pipeline.") :]] = tensor
        else:
            raise ValueError(f"unexpected Mage-VAE key in {vae_path}: {key!r}")
    if not remapped:
        raise ValueError(f"No 'student.dconv_encoder.*'/'pipeline.*' keys in {vae_path} (not a Mage-VAE?)")

    vae = AutoencoderKLMageFlow()
    if dtype is not None:
        remapped = {k: v.to(dtype) for k, v in remapped.items()}
    info = vae.load_state_dict(remapped, strict=True, assign=True)
    logger.info("Loaded Mage-VAE: %s", info)

    vae.to(device)
    return vae


#: The key prefixes of the two codecs a Mage-Flow run can be pointed at. Both latents
#: are 128 channels at 16x, so only the file says which one is in use.
_MAGE_PREFIXES = ("student.", "pipeline.")
_FLUX2_PREFIXES = ("encoder.", "decoder.", "bn.")


def load_mage_family_vae(
    vae_path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> Union[AutoencoderKLMageFlow, AutoencoderKLFlux2]:
    """Load whichever codec ``vae_path`` holds: the Mage-VAE or the Flux.2 AE."""
    with MemoryEfficientSafeOpen(vae_path) as f:
        keys = list(f.keys())

    if any(key.startswith(_MAGE_PREFIXES) for key in keys):
        return load_mage_vae(vae_path, device, dtype=dtype)
    if any(key.startswith(_FLUX2_PREFIXES) for key in keys):
        logger.info("Mage-Flow codec: the Flux.2 AE rather than the Mage-VAE (%s)", vae_path)
        return load_flux2_vae(vae_path, device, dtype=dtype)
    raise ValueError(
        f"{vae_path} is neither a Mage-VAE (no 'student.*'/'pipeline.*' keys) nor a "
        "Flux.2 AE (no 'encoder.*'/'decoder.*'/'bn.*' keys)"
    )


__all__ = ["AutoencoderKLMageFlow", "load_mage_family_vae", "load_mage_vae"]
