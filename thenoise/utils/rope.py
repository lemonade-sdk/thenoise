"""Shared rotary-position-embedding (RoPE) machinery.

Two rotation conventions, each with a builder producing the frequency tensors and
an ``apply_*`` that consumes them:

  * **2x2 matrix** (``[cos, -sin, sin, cos]`` per adjacent pair, carried as a
    ``[B, L, dim/2, 2, 2]`` tensor). No complex ops, so ``torch.compile`` can
    codegen the attention path.
  * **split-half** (``cos/sin`` at half the rotated width, applied to the two halves
    of those channels).

``RopeCache`` stores frequencies under simple string names, computing them on
demand through a model-supplied builder.
"""
from __future__ import annotations

import torch
from einops import rearrange


def rope(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """Rotary frequencies for one position axis, as a ``[B, seq, dim/2, 2, 2]`` matrix."""
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)


def matrix_rope(dims: list[int], theta: float):
    """Builder for the 2x2-matrix convention.

    Returns ``pos -> [B, L, dim/2, 2, 2]``, embedding ``pos`` (``[B, L, n_axes]``)
    axis-by-axis with its own ``dims[i]`` and ``theta``.
    """
    def build(pos: torch.Tensor) -> torch.Tensor:
        return torch.cat([rope(pos[..., i], d, theta) for i, d in enumerate(dims)], dim=-3)
    return build


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``freqs`` ``[B, L, dim/2, 2, 2]`` to query/key ``[B, H, L, D]``."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs.unsqueeze(1)  # [B, 1, L, dim/2, 2, 2]
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_out = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dim, negating the second."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_split_half(xq: torch.Tensor, cos: torch.Tensor,
                          sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first ``2 * cos.shape[-1]`` channels of ``xq`` ``[B, H, L, D]``.

    ``cos``/``sin`` carry each angle once rather than duplicated over the rotation's two
    halves, and broadcast against the channels they turn: ``[1, 1, L, rot/2]`` from the
    builders below, or ``[B, H, L, rot/2]`` where a stream hands its heads different
    slices of one frequency range. Channels past the rotation pass through untouched,
    which is where a rope narrower than the head leaves its idle tail.

    The band is doubled here rather than carried doubled in the table, to keep the
    rotation a single elementwise chain on its way into attention. On gfx1151 a ``cat``
    joining two *computed* operands costs 1.6x to 3.4x at 16k tokens (measured on blocks
    of three models, where gfx1150 does not notice), whereas a ``cat`` appending an idle
    tail — one computed operand and a view of the query — is free. The doubling pays for
    itself even where the band is per-head and therefore as wide as the rotation it
    feeds: 5-6% ahead of the concatenation on a two-stream block, measured at both 9k
    and 16k tokens. Carrying the band doubled in the table would buy the same chain at
    the price of phases twice as wide, so the doubling happens on the way past.

    The answer comes back in the query's dtype. Leaving a promotion in place — fp32
    phases beside bf16 weights — hands SDPA operands of mixed dtype, and it answers
    that with its unfused backend rather than by refusing.
    """
    if cos.shape != sin.shape:
        raise ValueError("RoPE needs cos and sin of one shape")
    half = cos.shape[-1]
    if not half or 2 * half > xq.shape[-1]:
        raise ValueError(f"a rotation of {2 * half} channels does not fit a head of "
                         f"{xq.shape[-1]}")
    cos, sin = (torch.cat((band, band), dim=-1) for band in (cos, sin))
    rotated, rest = xq[..., :2 * half], xq[..., 2 * half:]
    rotated = rotated * cos + _rotate_half(rotated) * sin
    if rest.shape[-1]:
        rotated = torch.cat((rotated, rest), dim=-1)
    return rotated.to(xq.dtype)


def split_half_rope_1d(head_dim: int, theta: float = 10000.0):
    """Builder for the split-half convention over a 1D position sequence.

    Returns ``(seq_len, device) -> (cos, sin)``, each ``[1, 1, seq_len, head_dim // 2]``,
    with frequencies following ``theta ** (2i/head_dim)``.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    def build(seq_len: int, device):
        freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq.to(device))
        return freqs.cos()[None, None], freqs.sin()[None, None]

    return build


def split_half_rope_axes(scale: torch.Tensor, angle_scale: float = 1.0):
    """Builder of the split-half convention over any number of axes sharing one frequency.

    Returns ``pos -> (cos, sin)``, each ``[1, 1, L, n_axes * len(scale)]``, for ``pos``
    an ``[L, n_axes]`` coordinate tensor: every axis rotated by the same ``scale``, the
    axes' angles concatenated in coordinate order.

    ``angle_scale`` is how far one coordinate unit travels: 1.0 for coordinates the
    frequencies already expect, ``2 * pi`` for a normalized grid spanning one turn.
    """

    def build(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = (angle_scale * pos[:, :, None] * scale.to(pos)).flatten(1)
        return angles.cos()[None, None], angles.sin()[None, None]

    return build


def split_half_rope_4d(head_dim: int, theta: float = 10000.0):
    """Builder for the split-half convention over a 4-axis ``(r, f, h, w)`` position.

    Returns ``pos -> (cos, sin)``, each ``[1, 1, L, head_dim // 2]``, for ``pos`` an
    ``[L, 4]`` coordinate tensor (no batch axis, as ``utils.positions`` builds them).

    The head dim splits into 4 axes of ``head_dim // 4`` channels, each carrying
    ``head_dim // 8`` frequencies. That fixes which pair :func:`apply_rope_split_half`
    rotates with which angle: ``(j, j + head_dim // 4)`` belongs to axis
    ``j // (head_dim // 8)``.
    """
    if head_dim % 8:
        raise ValueError(
            f"a 4-axis split-half rope splits the head dim into 4 x 2 groups, "
            f"so it needs head_dim divisible by 8, got {head_dim}"
        )
    bands = head_dim // 8

    def build(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Frequencies in the coordinates' own precision.
        scale = 1.0 / theta ** (torch.arange(bands, dtype=pos.dtype, device=pos.device) / bands)
        return split_half_rope_axes(scale)(pos)

    return build


def split_half_rope_3d(
    head_dim: int,
    patch_spatial: int,
    patch_temporal: int,
    h_extrapolation_ratio: float = 1.0,
    w_extrapolation_ratio: float = 1.0,
    t_extrapolation_ratio: float = 1.0,
):
    """Builder for the split-half convention over a ``(T, H, W)`` video grid.

    Returns ``(shape, device) -> (cos, sin)`` where ``shape`` is the raw latent
    ``(B, C, T, H, W)`` and the result is ``[T'*H'*W', 1, 1, head_dim]`` over the
    patchified grid. Frequencies are split across the temporal and spatial axes and
    NTK-scaled (``theta = 10000 * ratio``).
    """
    dim_h = head_dim // 6 * 2
    dim_w = dim_h
    dim_t = head_dim - 2 * dim_h
    h_ntk = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
    w_ntk = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
    t_ntk = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
    h_theta = 10000.0 * h_ntk
    w_theta = 10000.0 * w_ntk
    t_theta = 10000.0 * t_ntk

    def build(shape, device):
        _, _, T, H, W = shape
        T = T // patch_temporal
        H = H // patch_spatial
        W = W // patch_spatial
        h_range = torch.arange(0, dim_h, 2, device=device)[: dim_h // 2].float() / dim_h
        t_range = torch.arange(0, dim_t, 2, device=device)[: dim_t // 2].float() / dim_t
        h_freq = 1.0 / (h_theta**h_range)
        w_freq = 1.0 / (w_theta**h_range)
        t_freq = 1.0 / (t_theta**t_range)
        half_t = torch.outer(torch.arange(T, device=device).float(), t_freq)
        half_h = torch.outer(torch.arange(H, device=device).float(), h_freq)
        half_w = torch.outer(torch.arange(W, device=device).float(), w_freq)
        t_e = half_t[:, None, None, :].expand(T, H, W, -1)
        h_e = half_h[None, :, None, :].expand(T, H, W, -1)
        w_e = half_w[None, None, :, :].expand(T, H, W, -1)
        angles = torch.cat([t_e, h_e, w_e], dim=-1).reshape(T * H * W, -1)
        return angles.cos()[None, None], angles.sin()[None, None]

    return build


class RopeCache:
    """Caches computed rotary frequencies under simple string names.

    Frequencies come from a model-supplied ``build`` callable. ``store`` overwrites
    any previous value for a name, so the cache holds at most one entry per name.
    """

    def __init__(self, build):
        self._build = build
        self._cache: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}

    def store(self, name: str, *args, dtype: torch.dtype | None = None):
        """Compute frequencies for ``*args`` and cache them under ``name``.

        ``dtype`` casts the result; a ``(cos, sin)`` tuple is cast element-wise.
        """
        freqs = self._build(*args)
        if dtype is not None:
            if isinstance(freqs, tuple):
                freqs = tuple(f.to(dtype) for f in freqs)
            else:
                freqs = freqs.to(dtype)
        self._cache[name] = freqs
        return freqs

    def __getitem__(self, name: str):
        try:
            return self._cache[name]
        except KeyError:
            raise KeyError(
                f"RopeCache has no entry for {name!r}; call store({name!r}, ...) first"
            ) from None

    def clear(self) -> None:
        """Drop all cached entries."""
        self._cache.clear()


__all__ = [
    "rope",
    "matrix_rope",
    "apply_rope",
    "apply_rope_split_half",
    "split_half_rope_3d",
    "split_half_rope_4d",
    "split_half_rope_axes",
    "split_half_rope_1d",
    "RopeCache",
]
