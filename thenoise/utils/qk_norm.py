"""QK-norm used by the DiT attention blocks.

RMSNorm on the query and key head tensors before attention; the value tensor never
participates. ``eps`` is a constructor parameter because the models differ.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from thenoise.utils.rms_norm import RMSNorm


class QKNorm(nn.Module):
    """RMSNorm on query and key heads; value is untouched.

    ``q`` and ``k`` are ``[B, H, L, D]``, or any layout whose last dim is the head
    dim. The two norms share no weights.
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

    The shared module names its two norms ``query_norm``/``key_norm``; checkpoints
    store them under the model's original attribute names, which are passed in as
    ``q_legacy``/``k_legacy`` so ``load_dit``'s ``key_map`` can bridge the two.
    """
    key = key.replace(f".{q_legacy}.", ".qk_norm.query_norm.")
    key = key.replace(f".{k_legacy}.", ".qk_norm.key_norm.")
    return key


__all__ = ["QKNorm", "qk_norm_key_map"]
