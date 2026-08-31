# SDXL LDM UNet architecture.
#
# Stable Diffusion XL's UNet keeps the classic
# CompVis/LDM ``UNetModel`` layout (``input_blocks`` / ``middle_block`` /
# ``output_blocks`` with LDM-style resnets and diffusers-style transformer
# blocks) but reallocates the transformer depth: attention is dropped entirely
# at the lowest-resolution (320ch) stage, the 640ch stage uses 2 transformer
# blocks, and the 1280ch stage (plus the middle block) uses 10 transformer
# blocks. This matches ComfyUI's SDXL config ``transformer_depth``:
# ``[0, 0, 2, 2, 10, 10]`` (input), ``transformer_depth_middle: 10``,
# ``transformer_depth_output: [0, 0, 0, 2, 2, 2, 10, 10, 10]``.
#
# Weight key names are unchanged from the official / ComfyUI checkpoint, so the
# single-file and split checkpoints load as-is:
#   time_embed.{0,2}            Linear(320,1280) SiLU Linear(1280,1280)
#   label_emb.0.{0,2}           Linear(2816,1280) SiLU Linear(1280,1280)
#   input_blocks.N.M.*          conv/resnet/transformer/downsample
#   middle_block.M.*            resnet/transformer
#   output_blocks.N.M.*         resnet/transformer/upsample
#   out.{0,2}                   GroupNorm(320) Conv2d(320,4)
#
# The model is an epsilon (noise-prediction) discrete model, not a flow model:
# ``forward`` returns the predicted noise given a discrete timestep index
# (0..999). The timestep embedding is ComfyUI's ``timestep_embedding``
# (cos-then-sin, max_period 10000) fed through ``time_embed``, and the
# ADM vector ``y`` (2816 = pooled text 1280 + size embedding 1536) is mapped by
# ``label_emb`` and ADDED to the time embedding.
#
# Copyright 2025 OnomaAI (Illustrious-XL) / Stability AI (SDXL base), under
# the CreativeML Open RAIL-M / SDXL licenses.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.utils.attention import attention
from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


#: Channel count of the UNet's first stage (also the time-embedding input dim).
MODEL_CHANNELS = 320
#: ``time_embed`` / ``label_emb`` output width (320 * 4).
TIME_EMBED_DIM = 1280
#: Cross-attention context dim (77 x 2048 dual-CLIP concatenation).
CONTEXT_DIM = 2048
#: Per-head channel count for the transformer attention (heads = dim/64).
HEAD_DIM = 64
#: Group count for the resnet / transformer-input GroupNorms.
NORM_GROUPS = 32


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embedding (ComfyUI's, cos-then-sin).

    ``timesteps`` are discrete timestep indices in [0, 1000). Returns
    ``[B, dim]``. Matches ``comfy/ldm/modules/diffusionmodules/util.py``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResnetBlock2D(nn.Module):
    """LDM/CompVis ``ResnetBlock2D`` with group-norm, SiLU and a time embedding.

    Keys: ``in_layers.0`` (GN), ``in_layers.2`` (conv), ``emb_layers.1`` (time
    MLP), ``out_layers.0`` (GN), ``out_layers.3`` (conv), ``skip_connection``.
    """

    def __init__(self, in_channels, out_channels, time_embed_dim=TIME_EMBED_DIM):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(NORM_GROUPS, in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(NORM_GROUPS, out_channels, eps=1e-5),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, temb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(temb).type_as(h)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Attention(nn.Module):
    """diffusers-style MHA with fused q/k/v and an ``out`` projection.

    Keys: ``to_q``, ``to_k``, ``to_v``, ``to_out.0``. Cross-attention uses a
    different ``context_dim`` for k/v (2048 for the text stream).
    """

    def __init__(self, dim, num_heads, head_dim, context_dim=None):
        super().__init__()
        context_dim = context_dim if context_dim is not None else dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # SDXL transformer attention has NO bias on q/k/v; only ``to_out.0``.
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.0))

    def forward(self, hidden, context=None):
        context = context if context is not None else hidden
        B, L, _ = hidden.shape
        q = self.to_q(hidden).view(B, -1, self.num_heads, self.head_dim)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim)
        out = attention([q, k, v], drop_rate=0.0)
        return self.to_out(out)


class GEGLU(nn.Module):
    """GEGLU gated feed-forward entry. Key ``proj`` projects to 2*inner_dim."""

    def __init__(self, dim, inner_dim):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * inner_dim)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """``net.0`` = GEGLU, ``net.1`` = dropout, ``net.2`` = Linear(inner, dim)."""

    def __init__(self, dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(GEGLU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """One pre-norm attention block: self-attn, cross-attn, GEGLU feed-forward."""

    def __init__(self, dim, num_heads, head_dim, context_dim=CONTEXT_DIM):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn1 = Attention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.attn2 = Attention(dim, num_heads, head_dim, context_dim=context_dim)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim, inner_dim=4 * dim)

    def forward(self, x, context):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class Transformer2DModel(nn.Module):
    """diffusers ``Transformer2DModel`` (``use_linear_projection=True``).

    Keys: ``norm`` (GroupNorm), ``proj_in``, ``transformer_blocks.N.*``,
    ``proj_out``. Operates on a ``[B, C, H, W]`` feature map.
    """

    def __init__(self, dim, num_heads, num_layers, context_dim=CONTEXT_DIM):
        super().__init__()
        head_dim = HEAD_DIM
        self.norm = nn.GroupNorm(NORM_GROUPS, dim, eps=1e-6)
        self.proj_in = nn.Linear(dim, dim)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(dim, num_heads, head_dim, context_dim) for _ in range(num_layers)]
        )
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_in = x
        h = self.norm(x)
        h = self.proj_in(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B,C,H,W]
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        for block in self.transformer_blocks:
            h = block(h, context)
        h = self.proj_out(h)
        h = h.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return h + x_in


class Downsample(nn.Module):
    """Strided conv downsampler. Key: ``op`` (Conv2d 3x3, stride 2)."""

    def __init__(self, channels):
        super().__init__()
        # SDXL uses symmetric ``padding=1`` on the stride-2 conv (ComfyUI's
        # ``Downsample``), not the LDM ``(0, 1, 0, 1)`` manual pad. The manual
        # asymmetric pad produced a ~5% different latent and degraded images.
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """Nearest upsample + conv. Key: ``conv`` (Conv2d 3x3)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SdxlUNet(nn.Module):
    """The SDXL UNet (LDM ``UNetModel`` layout).

    Args:
        in_channels: latent channels (4).
        out_channels: 4.
        context_dim: 2048.
        adm_in_channels: 2816 (pooled text 1280 + size embedding 1536).
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        context_dim=CONTEXT_DIM,
        adm_in_channels=2816,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adm_in_channels = adm_in_channels

        ch = MODEL_CHANNELS          # 320
        ch2 = MODEL_CHANNELS * 2     # 640
        ch4 = MODEL_CHANNELS * 4     # 1280
        td = TIME_EMBED_DIM          # 1280

        # time_embed / label_emb. ``label_emb`` is doubly-nested (keys
        # ``label_emb.0.0`` / ``label_emb.0.2``) matching the checkpoint.
        self.time_embed = nn.Sequential(
            nn.Linear(ch, td), nn.SiLU(), nn.Linear(td, td)
        )
        self.label_emb = nn.Sequential(
            nn.Sequential(nn.Linear(adm_in_channels, td), nn.SiLU(), nn.Linear(td, td))
        )

        # ----------------------------------------------------- input blocks
        # 0: conv_in 4->320 ; 1,2: resnet 320 (no attention)
        # 3: downsample 320 ; 4,5: resnet 640 + transformer x2
        # 6: downsample 640 ; 7,8: resnet 1280 + transformer x10
        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(in_channels, ch, 3, padding=1)),
                nn.Sequential(ResnetBlock2D(ch, ch)),
                nn.Sequential(ResnetBlock2D(ch, ch)),
                nn.Sequential(Downsample(ch)),
                nn.Sequential(
                    ResnetBlock2D(ch, ch2), Transformer2DModel(ch2, num_heads=10, num_layers=2)
                ),
                nn.Sequential(
                    ResnetBlock2D(ch2, ch2), Transformer2DModel(ch2, num_heads=10, num_layers=2)
                ),
                nn.Sequential(Downsample(ch2)),
                nn.Sequential(
                    ResnetBlock2D(ch2, ch4), Transformer2DModel(ch4, num_heads=20, num_layers=10)
                ),
                nn.Sequential(
                    ResnetBlock2D(ch4, ch4), Transformer2DModel(ch4, num_heads=20, num_layers=10)
                ),
            ]
        )

        # ----------------------------------------------------- middle block
        self.middle_block = nn.Sequential(
            ResnetBlock2D(ch4, ch4),
            Transformer2DModel(ch4, num_heads=20, num_layers=10),
            ResnetBlock2D(ch4, ch4),
        )

        # ---------------------------------------------------- output blocks
        # Each output resnet consumes ``cat(h, skip)``; in-channels follow the
        # skip pairing (hs: [320,320,320,320,640,640,640,1280,1280]):
        #   0,1: 2560 -> 1280 + transformer x10
        #   2:   1920 -> 1280 + transformer x10 + upsample
        #   3:   1920 -> 640 + transformer x2
        #   4:   1280 -> 640 + transformer x2
        #   5:   960  -> 640 + transformer x2 + upsample
        #   6:   960  -> 320 ; 7,8: 640 -> 320 (no attention)
        self.output_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ResnetBlock2D(2 * ch4, ch4), Transformer2DModel(ch4, num_heads=20, num_layers=10)
                ),
                nn.Sequential(
                    ResnetBlock2D(2 * ch4, ch4), Transformer2DModel(ch4, num_heads=20, num_layers=10)
                ),
                nn.Sequential(
                    ResnetBlock2D(ch4 + ch2, ch4),
                    Transformer2DModel(ch4, num_heads=20, num_layers=10),
                    Upsample(ch4),
                ),
                nn.Sequential(
                    ResnetBlock2D(ch4 + ch2, ch2), Transformer2DModel(ch2, num_heads=10, num_layers=2)
                ),
                nn.Sequential(
                    ResnetBlock2D(2 * ch2, ch2), Transformer2DModel(ch2, num_heads=10, num_layers=2)
                ),
                nn.Sequential(
                    ResnetBlock2D(ch2 + ch, ch2),
                    Transformer2DModel(ch2, num_heads=10, num_layers=2),
                    Upsample(ch2),
                ),
                nn.Sequential(ResnetBlock2D(ch2 + ch, ch)),
                nn.Sequential(ResnetBlock2D(2 * ch, ch)),
                nn.Sequential(ResnetBlock2D(2 * ch, ch)),
            ]
        )

        # --------------------------------------------------------- head
        self.out = nn.Sequential(
            nn.GroupNorm(NORM_GROUPS, ch, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    # ------------------------------------------------------------ helpers
    def _run_seq(self, seq, x, emb, context):
        for m in seq:
            if isinstance(m, ResnetBlock2D):
                x = m(x, emb)
            elif isinstance(m, Transformer2DModel):
                x = m(x, context)
            else:
                x = m(x)
        return x

    # ------------------------------------------------------------ forward
    @torch.compile(fullgraph=True)
    def forward(self, x, t, y, context):
        """Denoise one step.

        Args:
            x: latent ``[B, 4, H, W]`` (scaled by the scheduler's init sigma).
            t: discrete timestep indices ``[B]`` in [0, 1000).
            y: ADM vector ``[B, 2816]`` (pooled text + size embedding).
            context: cross-attention text ``[B, 77, 2048]``.

        Returns:
            the predicted noise ``[B, 4, H, W]`` (epsilon prediction).
        """
        t_emb = timestep_embedding(t, MODEL_CHANNELS).to(x.dtype)
        emb = self.time_embed(t_emb)
        emb = emb + self.label_emb(y)

        h = x
        hs = []
        for block in self.input_blocks:
            h = self._run_seq(block, h, emb, context)
            hs.append(h)

        h = self._run_seq(self.middle_block, h, emb, context)

        for block in self.output_blocks:
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            h = self._run_seq(block, h, emb, context)

        return self.out(h)
