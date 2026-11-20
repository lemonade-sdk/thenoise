from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from thenoise.dit.quantized import QuantizedLinear


class GELU(nn.Module):
    """One linear, then GELU: the ``net.0`` slot of the diffusers feed-forward."""

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = QuantizedLinear(dim_in, dim_out, bias=bias)
        self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(self.proj(x))


class FeedForward(nn.Module):
    """Two linears with a GELU between, in diffusers' ``net.0``/``net.1``/``net.2`` slots.

    The dropout stays at p=0 so those slot numbers keep matching the saved weights.
    """

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4,
                 bias: bool = True, approximate: str = "tanh"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList(
            [
                GELU(dim, inner_dim, approximate=approximate, bias=bias),
                nn.Dropout(0.0),
                QuantizedLinear(inner_dim, dim_out, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


__all__ = ["FeedForward", "GELU"]
