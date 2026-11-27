"""Shared RMS normalization used by the DiT models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS normalization with a learnable scale; ``heads`` gives each head its own scale row."""

    def __init__(self, dim: int, eps: float = 1e-5, heads: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((heads, dim) if heads else dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dim() > 1:
            scaled = F.rms_norm(x.float(), (self.dim,), eps=self.eps) * self.weight
            return scaled.to(x.dtype)
        return F.rms_norm(x, (self.dim,), eps=self.eps, weight=self.weight)


__all__ = ["RMSNorm"]
