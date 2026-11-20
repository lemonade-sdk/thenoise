"""QK-norm used by the DiT attention blocks.

A norm on the query and key head tensors before attention; the value tensor never
participates. ``eps`` and the norm class are parameters because the models differ: the
DiTs use RMSNorm on head vectors, the shape codec's LayerNorms carry a bias.
"""
from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from thenoise.utils.rms_norm import RMSNorm


class QKNorm(nn.Module):
    """A norm on query and key; value is untouched. The two norms share no weights.

    The width a norm spans is a fact about the checkpoint: an RMS over one head is not
    an RMS over the concatenated head vector, so normalising before the split into heads
    is a different operation from normalising after it, and the checkpoint's own norm
    shape decides which. Before the split also means the tensors have not yet reached the
    layout :func:`thenoise.utils.attention.attention` asks for.
    """

    def __init__(self, dim: int, eps: float = 1e-5, norm: Type[nn.Module] = RMSNorm) -> None:
        super().__init__()
        self.query_norm = norm(dim, eps=eps)
        self.key_norm = norm(dim, eps=eps)

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
