"""The store a wide-output projection wants its weight kept in.

A projection much wider out than in reads its weight rather than its activation when the
stream is short, and ``F.linear`` reads that weight strided by the row it is stored in.
Storing the same numbers the other way round — ``W`` kept as ``Wᵀ``, with ``W``'s shape —
hands the GEMM the operand it wants. The loader applies it as the weights land, for the
models that ask.

Which ask is written down once each, as ``RELAY_STORES`` in the ``thenoise.dit`` module that
loads that transformer, and ``bench-scripts/block_bench.py`` reads those: a photographed
block has to carry the store a load would leave it in, on the backend the photograph is
taken on.

Not in ``thenoise.dit.quantized``, where the projections live: that module imports the
loader, and the loader is what needs this.
"""
from __future__ import annotations

import torch

#: The only backend these layout penalties were measured on; elsewhere the copies the
#: rules buy would be pure cost.
LAYOUT_PENALTY = torch.version.hip is not None

#: The out-to-in ratio from which a projection's GEMM is reading its weight, measured on
#: gfx1151. The ratio alone does not decide a model: ``moss_image``'s joint block is about
#: 1% worse at these widths, so each model asks for the store at load rather than inheriting
#: it from this bound.
TRANSPOSE_MIN_RATIO = 3.0


def stores_weight_transposed(in_features: int, out_features: int) -> bool:
    """Whether a full-precision projection of these widths wants its weight the other way.

    ``F.linear`` reads the weight as ``[K, N]``, strided by its stored row: with N much
    larger than K and few rows to amortise the read, that read is the GEMM's traffic.
    """
    return LAYOUT_PENALTY and out_features >= TRANSPOSE_MIN_RATIO * in_features


def weight_relayed(weight: torch.Tensor) -> torch.Tensor:
    """``weight``'s own numbers in the store a wide-output projection wants, or ``weight``.

    Shape and values are unchanged, so anything reading ``module.weight`` by shape —
    ``F.linear``, a LoRA delta, a quantizer — sees what it expected, and no checkpoint
    changes. Only a plain row-major floating-point store is relayed: a quantized weight is
    a different tensor in front of a different kernel, and a weight already strided was laid
    out by someone else.

    Moves and dtype casts preserve the resulting strides, so the store is applied once where
    the weights land rather than by a hook on every module door.
    """
    if (weight.dim() != 2 or not weight.is_floating_point()
            or weight.stride(-1) != 1
            or not stores_weight_transposed(weight.shape[-1], weight.shape[-2])):
        return weight
    return weight.t().contiguous().t()


__all__ = ["LAYOUT_PENALTY", "TRANSPOSE_MIN_RATIO", "stores_weight_transposed",
           "weight_relayed"]
