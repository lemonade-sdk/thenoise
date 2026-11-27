from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from thenoise.sparse import SparseChannel2Spatial, SparseTensor, SubmanifoldConv3d
from thenoise.utils.loader import Header, load_codec

COMPUTE_DTYPE = torch.bfloat16
SUBDIVISION = 2
NORM_EPS = 1e-6


class SparseConvNeXtBlock3d(nn.Module):
    """Conv, layernorm, MLP, with the input added back on the same lattice."""

    def __init__(self, channels: int, mlp_hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=NORM_EPS)
        self.conv = SubmanifoldConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(nn.Linear(channels, mlp_hidden), nn.SiLU(inplace=True),
                                 nn.Linear(mlp_hidden, channels))

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = self.conv(x)
        h = h.replace(self.mlp(self.norm(h.feats)))
        return h.replace(h.feats + x.feats)


class SubdivisionUpBlock(nn.Module):
    """Refine by two, on this block's own head or one handed in; the head reads the block's
    raw input and a cell is born where it is positive.
    """

    def __init__(self, channels: int, out_channels: int, pred_subdiv: bool) -> None:
        super().__init__()
        self.factor = SUBDIVISION
        if channels % self.factor ** 3:
            raise ValueError(f"{channels} channels do not split into {self.factor ** 3} cells")
        self.per_child = channels // self.factor ** 3
        self.out_channels = out_channels
        if out_channels % self.per_child:
            raise ValueError(f"{out_channels} channels is not a whole number of parent "
                             f"channels ({self.per_child})")
        self.norm1 = nn.LayerNorm(channels, eps=NORM_EPS)
        self.norm2 = nn.LayerNorm(out_channels, elementwise_affine=False, eps=NORM_EPS)
        self.conv1 = SubmanifoldConv3d(channels, out_channels * self.factor ** 3, 3)
        self.conv2 = SubmanifoldConv3d(out_channels, out_channels, 3)
        self.to_subdiv = (nn.Linear(channels, self.factor ** 3) if pred_subdiv else None)
        self.updown = SparseChannel2Spatial(self.factor)

    def forward(self, x: SparseTensor, subdiv: Optional[torch.Tensor] = None
                ) -> Tuple[SparseTensor, Optional[torch.Tensor]]:
        predicted = None if self.to_subdiv is None else self.to_subdiv(x.feats)
        logits = predicted if predicted is not None else subdiv
        keep = None if logits is None else (logits.reshape(len(x), self.factor ** 3) > 0)

        h = self.conv1(x.replace(F.silu(self.norm1(x.feats))))
        born, source = self.updown.plan(h, keep)
        carried = x.feats.reshape(len(x) * self.factor ** 3, self.per_child)[source]

        h = self.conv2(born.replace(F.silu(self.norm2(born.feats))))
        repeat = self.out_channels // self.per_child
        h = h.replace((h.feats.view(-1, self.per_child, repeat)
                       + carried.unsqueeze(-1).to(h.feats.dtype)).view(-1, self.out_channels))
        return h, predicted


class SlatDecoder(nn.Module):
    """A run of NeXt blocks per level, each closed by a refinement — ``blocks.<level>`` in
    the file, refinement last. The layernorm before the read-out has no parameters, hence no
    weights in the file either.
    """

    compute_dtype = COMPUTE_DTYPE

    def __init__(self, latent_channels: int, channels: Sequence[int],
                 mlp_hidden: Sequence[int], level_blocks: Sequence[int],
                 up_channels: Sequence[int], out_channels: int,
                 pred_subdiv: bool) -> None:
        super().__init__()
        if not (len(channels) == len(mlp_hidden) == len(level_blocks) == len(up_channels)):
            raise ValueError("a level needs its width, its MLP hidden width, its block count "
                             "and the width its refinement lands on")
        self.out_channels = out_channels
        self.pred_subdiv = pred_subdiv
        self.from_latent = nn.Linear(latent_channels, channels[0])
        self.blocks = nn.ModuleList()
        for width, hidden, count, up in zip(channels, mlp_hidden, level_blocks, up_channels):
            level = nn.ModuleList(SparseConvNeXtBlock3d(width, hidden) for _ in range(count))
            level.append(SubdivisionUpBlock(width, up, pred_subdiv))
            self.blocks.append(level)
        self.output_layer = nn.Linear(up_channels[-1], out_channels)

    def forward(self, x: SparseTensor, guide_subs: Optional[Sequence[torch.Tensor]] = None
                ) -> Tuple[SparseTensor, List[torch.Tensor]]:
        if not self.pred_subdiv and guide_subs is None:
            raise ValueError("this decoder has no subdivision heads of its own; it decodes on "
                             "the lattice another one chose")
        h = x.replace(self.from_latent(x.feats))
        subs: List[torch.Tensor] = []
        for index, level in enumerate(self.blocks):
            for block in level[:-1]:
                h = block(h)
            h, subdiv = level[-1](h, None if guide_subs is None else guide_subs[index])
            if subdiv is not None:
                subs.append(subdiv)
        h = h.replace(self.output_layer(F.layer_norm(h.feats, h.feats.shape[-1:])))
        return h, subs


def dual_grid_fields(feats: torch.Tensor, voxel_margin: float = 0.5
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The 7 channels of a shape read-out: dual-vertex offset within the cell and its margin,
    which of the three axes the surface crosses, and the weight that splits a quad.
    """
    vertices = (1 + 2 * voxel_margin) * torch.sigmoid(feats[..., 0:3]) - voxel_margin
    intersected = feats[..., 3:6] > 0
    split_weight = F.softplus(feats[..., 6:7])
    return vertices, intersected, split_weight


def _geometry(header: Header, prefix: str) -> dict:
    """The decoder the file describes: widths, depths and all, read off its own keys."""
    first, latent = header.shape(prefix + "from_latent.weight")
    out_channels = header.shape(prefix + "output_layer.weight")[0]

    channels, hidden, counts, up_channels = [], [], [], []
    level = 0
    while prefix + f"blocks.{level}.0.conv.weight" in header:
        channels.append(header.shape(prefix + f"blocks.{level}.0.conv.weight")[0])
        hidden.append(header.shape(prefix + f"blocks.{level}.0.mlp.0.weight")[0])
        index = 0
        while prefix + f"blocks.{level}.{index}.conv.weight" in header:
            index += 1
        counts.append(index)
        up_channels.append(header.shape(prefix + f"blocks.{level}.{index}.conv2.weight")[0])
        level += 1

    if not channels:
        raise ValueError(f"{header.path} has no decoder blocks under {prefix}")
    if first != channels[0]:
        raise ValueError(f"{prefix}from_latent reads into {first} and level 0 is "
                         f"{channels[0]} wide")
    pred_subdiv = any(prefix + f"blocks.{lv}.{count}.to_subdiv.weight" in header
                      for lv, count in enumerate(counts))
    return {"latent_channels": latent, "channels": channels, "mlp_hidden": hidden,
            "level_blocks": counts, "up_channels": up_channels,
            "out_channels": out_channels, "pred_subdiv": pred_subdiv}


def _decoder_state(prefix: str):
    """One decoder's subtree of a file that holds several.

    A submanifold conv is stored here as ``[out, x, y, z, in]`` — taps before the channels,
    in the same x-slowest order the neighbour map uses — so a weight is turned to whatever the
    module asks for rather than renamed by hand.
    """

    def state_map(state_dict: dict, model: nn.Module) -> dict:
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        taken = {}
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            want = expected.get(name)
            if want is not None and value.ndim == 5 and tuple(value.shape) != want:
                value = value.permute(0, 4, 1, 2, 3)
            taken[name] = value
        return taken

    return state_map


def load_trellis2_shape_vae(vae_path: str, device: torch.device | str,
                       dtype: Optional[torch.dtype] = None) -> SlatDecoder:
    """The geometry decoder: it chooses its own subdivision and reads out 7 channels."""
    return _load(vae_path, device, "shape decoder", "shape_dec.", dtype)


def load_trellis2_texture_vae(vae_path: str, device: torch.device | str,
                         dtype: Optional[torch.dtype] = None) -> SlatDecoder:
    """The material decoder: 6 channels, and it decodes where the shape decoder went."""
    return _load(vae_path, device, "texture decoder", "txt_dec.", dtype)


def _load(vae_path: str, device: torch.device | str, what: str, prefix: str,
          dtype: Optional[torch.dtype]) -> SlatDecoder:
    return load_codec(vae_path, device, what=f"Trellis2 {what}",
                      construct=lambda header: SlatDecoder(
                          **_geometry(header, prefix)),
                      state_map=_decoder_state(prefix), dtype=dtype)


__all__ = ["SlatDecoder", "SparseConvNeXtBlock3d", "SubdivisionUpBlock", "dual_grid_fields",
           "load_trellis2_shape_vae", "load_trellis2_texture_vae"]
