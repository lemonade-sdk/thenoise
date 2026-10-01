"""A linear projection that stays in FP32 inside a lower-precision model.

The DiTs run their trunk in the configured compute dtype (BF16 by default). Most
layers are happy with that and go through
:mod:`thenoise.dit.quantized.QuantizedLinear`, which is also the LoRA target and the
quantization hook. A layer that is *full-precision by contract* — weights calibrated
and exported in FP32, where rounding them is a quality regression rather than a
rounding error — wants the opposite of both of those, so it wants a different module.

Two things defeat "just keep the weight FP32", and this class exists to defeat them
back:

  * ``torch.autocast``: an autocast region around the model casts every ``F.linear``
    to the autocast dtype, so an FP32 weight is rounded on the fly. The GEMM here runs
    with autocast disabled instead. Casting the *activation* to FP32 does not help —
    autocast casts the weight right back down.
  * ``Module.to(dtype)``: the way a dtype cast reaches every parameter is ``_apply``,
    so that is where the dtype is pinned. Device moves still go through untouched,
    which is what the residency manager does on every offload/load.

Both are easy to reintroduce by accident, which is why the precision is a property of
the module rather than of the call site.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fp32Linear(nn.Linear):
    """Linear that runs its GEMM in FP32 no matter what the surrounding model runs in.

    Not quantizable and not a LoRA target by construction (it is not a
    ``QuantizedLinear``) — that is the point, but it also means it is the wrong choice
    for any layer you expect to shrink or adapt.

    ``out_dtype``, when set, casts the result. Leave it ``None`` when the FP32 value is
    the point (it feeds more FP32 math); set it to the trunk dtype when the result is
    handed back to a lower-precision residual path, which would otherwise be promoted
    wholesale by type promotion.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        out_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.out_dtype = out_dtype

    def _apply(self, fn, recurse=True):
        def keep_fp32(tensor):
            if not tensor.is_floating_point():
                return fn(tensor)
            return fn(tensor).to(torch.float32)

        return super()._apply(keep_fp32, recurse=recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = F.linear(
                x.float(), self.weight.float(), None if self.bias is None else self.bias.float()
            )
        return out if self.out_dtype is None else out.to(self.out_dtype)


__all__ = ["Fp32Linear"]
