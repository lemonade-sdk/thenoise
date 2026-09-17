"""Shared QK-norm used by the DiT attention blocks.

QK-norm applies RMSNorm to the query and key head tensors before attention.
The value tensor never participates, so it is left to the caller. ``eps`` is a
constructor parameter because the models differ: Flux.2 / Anima use ``1e-6``,
Krea 2 / Z-Image use ``1e-5``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from thenoise.utils.rms_norm import RMSNorm


class QKNorm(nn.Module):
    """RMSNorm on query and key heads; value is untouched.

    ``q`` and ``k`` are ``[B, H, L, D]`` (or any layout whose last dim is the head
    dim, since RMSNorm is per-token). The two norms share no weights.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.query_norm = RMSNorm(dim, eps=eps)
        self.key_norm = RMSNorm(dim, eps=eps)

    def reset_parameters(self) -> None:
        self.query_norm.reset_parameters()
        self.key_norm.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_norm(q), self.key_norm(k)


def qk_norm_key_map(key: str, q_legacy: str = "q_norm", k_legacy: str = "k_norm") -> str:
    """Map a checkpoint's legacy QK-norm keys onto the shared ``QKNorm`` layout.

    The shared module names its two norms ``query_norm``/``key_norm``, but the
    checkpoints store them under the model's original attribute names: Anima and
    Z-Image use ``q_norm``/``k_norm``, Krea 2 uses ``qnorm``/``knorm``. Pass those
    legacy names so ``load_dit``'s ``key_map`` can bridge the two.
    """
    key = key.replace(f".{q_legacy}.", ".qk_norm.query_norm.")
    key = key.replace(f".{k_legacy}.", ".qk_norm.key_norm.")
    return key


__all__ = ["QKNorm", "qk_norm_key_map"]
