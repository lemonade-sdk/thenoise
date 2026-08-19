"""Flux.2 (Flux Klein) sampling helpers.

The schedule is the official Flux.2 flow-matching schedule: ``linspace(1, 0,
steps+1)`` pushed through a generalized time/SNR shift whose ``mu`` is computed
empirically from the image-token count. Each schedule step is an Euler step of the
flow ODE, so the default solver is Euler.

Also carries the token-position helpers used to pack the 4D latent into the
DiT's 1D token sequence and back:

  * ``prc_img``  ``[B, C, H, W]`` -> ``[B, seq, C]`` tokens + ``[B, seq, 4]`` ids
  * ``prc_txt``  ``[B, L, Ctx]`` text -> unchanged + ``[B, L, 4]`` ids
  * ``scatter_ids`` reconstructs ``[B, C, 1, H, W]`` from tokens + ids
"""

from __future__ import annotations

import math

import torch
from einops import rearrange

__all__ = [
    "get_schedule",
    "compute_empirical_mu",
    "generalized_time_snr_shift",
    "prc_img",
    "prc_txt",
    "scatter_ids",
]


def generalized_time_snr_shift(t: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Generalized time/SNR shift: ``exp(mu) / (exp(mu) + (1/t - 1)^sigma)``."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Empirical shift ``mu`` for a given image token count and step count."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def get_schedule(num_steps: int, image_seq_len: int, flow_shift: float | None = None) -> list[float]:
    """Flux.2 timestep grid ``(num_steps + 1)`` values from 1 -> 0.

    Returns a list of floats (the last is 0.0). The ``i``-th Euler step integrates
    ``x += (t[i+1] - t[i]) * velocity``, so the adapter's ``denoise_step`` returns
    the model velocity directly (no negation).
    """
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if flow_shift is not None:
        timesteps = (timesteps * flow_shift) / (1 + (flow_shift - 1) * timesteps)
    else:
        timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def prc_img(x: torch.Tensor, t_coord: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a 4D latent ``[B, C, H, W]`` (or ``[C, H, W]``) into tokens + ids.

    Returns ``(tokens [B, H*W, C], ids [B, H*W, 4])``. The 4 position coordinates
    are (t, h, w, l) with ``t = l = 1`` for a still image.
    """
    h = x.shape[-2]
    w = x.shape[-1]
    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = rearrange(x, "c h w -> (h w) c") if x.ndim == 3 else rearrange(x, "b c h w -> b (h w) c")
    if x.ndim == 3:  # after rearrange
        x_ids = x_ids.unsqueeze(0).expand(x.shape[0], -1, -1)
    return x, x_ids.to(x.device)


def prc_txt(x: torch.Tensor, t_coord: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach position ids to a text embedding ``[B, L, C]``.

    Returns ``(x, ids [B, L, 4])``; the text tokens are unchanged.
    """
    _l = x.shape[-2]
    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),  # dummy
        "w": torch.arange(1),  # dummy
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    if x.ndim == 3:
        x_ids = x_ids.unsqueeze(0).expand(x.shape[0], -1, -1)
    return x, x_ids.to(x.device)


def _compress_time(t_ids: torch.Tensor) -> torch.Tensor:
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype)
    return t_remap[t_ids]


def scatter_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
    """Reconstruct ``[1, C, T, H, W]`` tensors from ``[B, seq, C]`` tokens + ids.

    Uses the position ids to scatter each token back into its grid slot. Returns a
    list of one tensor per batch element (T == 1 for a still image).
    """
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = _compress_time(t_ids)
        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids
        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
    return x_list
