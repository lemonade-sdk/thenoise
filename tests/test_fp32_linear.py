"""``Fp32Linear``: the two ways a "just keep this layer in FP32" layer loses FP32.

An autocast region and a dtype cast are the whole story, so this needs no weights and
no device — what is under test is that the module, not the call site, holds the line.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from thenoise.utils.fp32_linear import Fp32Linear


def _layer(**overrides) -> Fp32Linear:
    torch.manual_seed(0)
    return Fp32Linear(64, 32, **overrides).eval().requires_grad_(False)


def test_a_trunk_dtype_cast_cannot_round_the_parameters():
    layer = _layer().to(torch.bfloat16)
    assert layer.weight.dtype == torch.float32
    assert layer.bias.dtype == torch.float32


def test_the_gemm_runs_in_fp32_under_an_autocast_region():
    layer, x = _layer(), torch.randn(2, 64, dtype=torch.bfloat16)
    exact = F.linear(x.float(), layer.weight, layer.bias)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = layer(x)

    assert out.dtype == torch.float32
    assert torch.equal(out, exact)

    # The FP32-ness is worth something, not just a different spelling of the same
    # rounding: a bf16 copy of these weights cannot reproduce the FP32 result.
    bf16 = F.linear(x, layer.weight.to(torch.bfloat16), layer.bias.to(torch.bfloat16))
    assert (out - bf16.float()).abs().max() > 1e-3


def test_out_dtype_hands_the_result_back_in_the_trunk_dtype():
    out = _layer(out_dtype=torch.bfloat16)(torch.randn(2, 64, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
