"""Shared rotary-position-embedding (RoPE) machinery.

Two rotation conventions are supported, each with a small builder that produces
the frequency tensors and an ``apply_*`` that consumes them:

  * **2x2 matrix** (``[cos, -sin, sin, cos]`` per adjacent pair, carried as a
    ``[B, L, dim/2, 2, 2]`` tensor). No complex ops, so ``torch.compile`` can
    codegen the attention path.
  * **split-half** (precomputed ``cos/sin`` pairs, applied to the two halves of
    the head dim via ``_rotate_half``).

``RopeCache`` is the shared caching layer: it stores frequencies under simple
string names, computing them on demand through a model-supplied builder. Each
name maps to exactly one computed value and ``store`` overwrites the previous
one, so the cache never grows beyond the fixed set of names a caller uses.
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

    Returns a callable ``pos -> [B, L, dim/2, 2, 2]`` that embeds ``pos``
    (``[B, L, n_axes]``) axis-by-axis with its own ``dims[i]`` and ``theta``.
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
    """Split-half rotation: swap the two halves of the last dim, negating the second."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_split_half(xq: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply split-half RoPE to ``xq`` ``[B, L, H, D]`` using ``cos/sin`` ``[L, 1, 1, D]``."""
    cos = cos.transpose(0, 1)  # [1, L, 1, D]
    sin = sin.transpose(0, 1)
    rot_dim = cos.shape[-1]
    xq_rot, xq_pass = xq[..., :rot_dim], xq[..., rot_dim:]
    xq_rot = xq_rot * cos + _rotate_half(xq_rot) * sin
    return torch.cat((xq_rot, xq_pass), dim=-1)


def split_half_rope_3d(
    head_dim: int,
    patch_spatial: int,
    patch_temporal: int,
    h_extrapolation_ratio: float = 1.0,
    w_extrapolation_ratio: float = 1.0,
    t_extrapolation_ratio: float = 1.0,
):
    """Builder for the split-half convention over a ``(T, H, W)`` video grid.

    Returns a callable ``(shape, device) -> (cos, sin)`` where ``shape`` is the
    raw latent ``(B, C, T, H, W)`` and the result is ``[T'*H'*W', 1, 1, head_dim]``
    with ``T', H', W'`` the patchified grid dims (``T//patch_temporal``,
    ``H//patch_spatial``, ``W//patch_spatial``). Frequencies are split across the
    temporal (T) and spatial (H, W) axes and NTK-scaled
    (``theta = 10000 * ratio``).
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
        # Duplicate the (t, h, w) freqs to fill the full head dim.
        angles = torch.cat([t_e, h_e, w_e, t_e, h_e, w_e], dim=-1).reshape(T * H * W, 1, 1, -1)
        return angles.cos(), angles.sin()

    return build


class RopeCache:
    """Caches computed rotary frequencies under simple string names.

    Frequencies are produced by a model-supplied ``build`` callable (see
    ``matrix_rope`` / ``split_half_rope_3d``). ``store`` overwrites any previous
    value for a name, so the cache holds at most one entry per name and never
    grows beyond the fixed set of names a caller uses. The cache owns no
    learnable state and is not an ``nn.Module``.
    """

    def __init__(self, build):
        self._build = build
        self._cache: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}

    def store(self, name: str, *args, dtype: torch.dtype | None = None):
        """Compute frequencies for ``*args`` and cache them under ``name``.

        ``dtype`` casts the result (e.g. to the activation dtype); a ``(cos, sin)``
        tuple result is cast element-wise.
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
        """Drop all cached entries (call before starting a fresh prompt)."""
        self._cache.clear()


__all__ = [
    "rope",
    "matrix_rope",
    "apply_rope",
    "apply_rope_split_half",
    "split_half_rope_3d",
    "RopeCache",
]
