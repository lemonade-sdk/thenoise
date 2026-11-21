"""Where LTX's rotary phases come from.

Not the usual integer-index rope: a token's coordinates are its patch box's corners in
seconds and pixels, normalized by the canvas the model was trained on, so an angle means
a physical time rather than a rank. The rotation itself is the shared
:func:`thenoise.utils.rope.apply_rope_split_half`; what lives here is the coordinate
construction, and the fact that one frequency range is partitioned across the heads —
so the phases are per-head tensors, not one row broadcast to every head.

Both released builds keep their precision: the frequency vector in float64 (dropping it
moves cos and sin by about one bf16 ulp, which is free to keep and not free to lose) and
the rotation in float32.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from thenoise.utils.positions import grid_from_axes
from thenoise.utils.rope import RopeCache

#: One entry per module: each answers for one modality of one clip at a time.
_ENTRY = "phases"


class AVRope(nn.Module):
    """Fractional-coordinate split-half rope, with the clip's phases built once.

    The three entry points each describe a grid by its geometry rather than by a
    coordinate tensor, so the phases can be cached against that geometry: every step of
    a render asks for the same ones, and rebuilding them costs milliseconds per forward
    at the resolutions these models run. ``_apply`` drops them, since they belong to the
    device they were built on.
    """

    def __init__(self, dim: int, heads: int, *, modality: str,
                 base_num_frames: int = 20, base_height: int = 2048,
                 base_width: int = 2048, sampling_rate: int = 16000,
                 hop_length: int = 160, scale_factors=(8, 32, 32),
                 patch_size: int = 1, patch_size_t: int = 1,
                 causal_offset: int = 1, theta: float = 10000.0,
                 double_precision: bool = True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.modality = modality
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.scale_factors = tuple(scale_factors)
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.causal_offset = causal_offset
        self.theta = theta
        self.double_precision = double_precision
        self._frequencies = RopeCache(self._build)
        self._key = None

    def _apply(self, fn, recurse=True):
        self._frequencies.clear()
        self._key = None
        return super()._apply(fn, recurse)

    # ------------------------------------------------------------ entry points
    def video(self, batch: int, frames: int, height: int, width: int, fps: float,
              device) -> tuple[torch.Tensor, torch.Tensor]:
        """Phases ``[B, H, frames*height*width, head_dim/2]`` for a video patch grid."""
        return self._cached(("video", batch, frames, height, width, float(fps), device),
                            "video", batch, frames, height, width, float(fps), device)

    def audio(self, batch: int, tokens: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Phases for the audio stream's own time grid."""
        return self._cached(("audio", batch, tokens, device),
                            "audio", batch, tokens, device)

    def video_time(self, batch: int, frames: int, height: int, width: int, fps: float,
                   device) -> tuple[torch.Tensor, torch.Tensor]:
        """Phases carrying only where in time each video token sits, its pixels sharing one
        angle. Cross-modality attention compares a video with a sound, and time is the
        one coordinate the two have in common."""
        return self._cached(("video_time", batch, frames, height, width, float(fps), device),
                            "video_time", batch, frames, height, width, float(fps), device)

    def token_positions(self, batch: int, tokens: int,
                        device) -> tuple[torch.Tensor, torch.Tensor]:
        """Phases for plain token rank over ``base_num_frames``: a sequence with no
        physical time to place, which is what the text connectors carry."""
        return self._cached(("positions", batch, tokens, device),
                            "positions", batch, tokens, device)

    def _cached(self, key, kind, *args):
        if key != self._key:
            self._frequencies.store(_ENTRY, kind, *args)
            self._key = key
        return self._frequencies[_ENTRY]

    def _build(self, kind: str, batch: int, *args):
        grid = getattr(self, f"_{kind}_grid")(*args)
        return self._phases(grid, batch)

    # ------------------------------------------------------------ coordinates
    def _video_grid(self, frames: int, height: int, width: int, fps: float,
                    device) -> torch.Tensor:
        """``[tokens, 3]`` of normalized box midpoints, t-major as ``grid_from_axes``."""
        steps = (self.patch_size_t, self.patch_size, self.patch_size)
        axes = [torch.arange(0, end, step, device=device, dtype=torch.float32)
                for end, step in ((frames, steps[0]), (height, steps[1]),
                                  (width, steps[2]))]
        grid = grid_from_axes(axes)
        corners = torch.stack((grid, grid + grid.new_tensor(steps)), dim=-1)
        scale = grid.new_tensor(self.scale_factors).view(1, -1, 1)
        corners = corners * scale
        # Time is measured in seconds rather than in patch ranks, and the offset puts
        # the frames the codec drops at negative times; the clamp lands them on zero.
        corners[..., 0, :] = ((corners[..., 0, :] + self.causal_offset - self.scale_factors[0])
                              .clamp_min(0) / fps)
        return corners.mean(dim=-1) / grid.new_tensor(
            (self.base_num_frames, self.base_height, self.base_width))

    def _video_time_grid(self, frames: int, height: int, width: int, fps: float,
                         device) -> torch.Tensor:
        return self._video_grid(frames, height, width, fps, device)[:, :1]

    def _audio_grid(self, tokens: int, device) -> torch.Tensor:
        """``[tokens, 1]``: each audio token's box in seconds, from the mel hop."""
        grid = torch.arange(0, tokens, self.patch_size_t, device=device,
                            dtype=torch.float32)
        scale = self.scale_factors[0]
        start = (grid * scale + self.causal_offset - scale).clamp_min(0)
        end = ((grid + self.patch_size_t) * scale + self.causal_offset - scale).clamp_min(0)
        corners = torch.stack([start, end], -1) * (self.hop_length / self.sampling_rate)
        return corners[:, None, :].mean(dim=-1) / self.base_num_frames

    def _positions_grid(self, tokens: int, device) -> torch.Tensor:
        grid = torch.arange(tokens, device=device, dtype=torch.float32)
        return grid[:, None] / self.base_num_frames

    # ------------------------------------------------------------ phases
    def _phases(self, grid: torch.Tensor, batch: int):
        """``[tokens, axes]`` of normalized positions -> ``[B, H, tokens, head_dim/2]``."""
        bins = grid.shape[-1] * 2
        precision = torch.float64 if self.double_precision else torch.float32
        freqs = torch.pow(self.theta, torch.linspace(0, 1, self.dim // bins,
                                                    dtype=precision, device=grid.device))
        freqs = (freqs * math.pi / 2).float()
        angles = ((grid.unsqueeze(-1) * 2 - 1) * freqs).transpose(-1, -2).flatten(1)
        pad = self.dim // 2 - angles.shape[-1]
        cos, sin = angles.cos(), angles.sin()
        if pad:
            # Channels the axes ran out of rotate by nothing, written as 1 and 0 rather
            # than as a band the rotation would have to know where to put.
            cos = torch.cat([cos.new_ones((*cos.shape[:-1], pad)), cos], -1)
            sin = torch.cat([sin.new_zeros((*sin.shape[:-1], pad)), sin], -1)
        half = self.dim // 2
        if half % self.heads:
            raise ValueError(f"{self.dim} phases do not divide across {self.heads} heads")
        per_head = half // self.heads
        out = []
        for table in (cos, sin):
            heads = table.reshape(-1, self.heads, per_head).permute(1, 0, 2)
            out.append(heads[None].expand(batch, -1, -1, -1).contiguous())
        return out[0], out[1]
