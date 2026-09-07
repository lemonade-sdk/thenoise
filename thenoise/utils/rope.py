"""Shared 2x2-matrix rotary embedding (Flux.2 and Krea 2).

Both models apply the same per-axis RoPE: frequencies are built as a
``[cos, -sin, sin, cos]`` 2x2 matrix and applied to the query/key heads via
real-arithmetic multiplies (no complex ops, so ``torch.compile`` can codegen the
attention path). Frequencies are computed in fp32 and the rotation runs in fp32
before casting back to the activation dtype.

Frequencies are carried in ``[B, L, dim/2, 2, 2]`` (one matrix per token per
axis, axes concatenated along the ``dim/2`` axis); ``apply_rope`` inserts the
broadcast head dim itself.
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


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``freqs`` ``[B, L, dim/2, 2, 2]`` to query/key ``[B, H, L, D]``."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]  # [B, 1, L, dim/2, 2, 2]
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_out = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


__all__ = ["rope", "apply_rope"]
