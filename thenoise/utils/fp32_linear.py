"""A linear projection that stays in FP32 inside a lower-precision model.

A layer whose weights are calibrated and exported in FP32, where rounding them is a
quality regression rather than a rounding error, has to survive the two things that
would round them anyway:

  * ``torch.autocast``, which casts every ``F.linear`` to the autocast dtype — the
    GEMM here runs with autocast disabled;
  * ``Module.to(dtype)``, which reaches every parameter through ``_apply`` — that is
    where the dtype is pinned, while device moves still go through untouched.

Keeping the precision a property of the module makes both hard to break by accident.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fp32Linear(nn.Linear):
    """Linear that runs its GEMM in FP32 no matter what the surrounding model runs in.

    ``out_dtype``, when set, casts the result: leave it ``None`` when the FP32 value
    is the point, set it to the trunk dtype when the result feeds a lower-precision
    residual path that type promotion would otherwise widen.
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
