"""Shared RMS normalization used by the DiT models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS normalization with a learnable scale ``weight``.

    ``weight`` is ones-initialized, so the module starts as plain RMS
    normalization; the variance is computed in fp32 and the normalized result is
    cast back to the input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), eps=self.eps, weight=self.weight)


__all__ = ["RMSNorm"]
