"""``weight_relayed``: the store a wide-output projection wants its weight kept in.

Where a projection is much wider out than in, its GEMM reads the weight rather than the
activation, and ``F.linear`` reads that weight strided by the row it is stored in. The same
numbers stored with the reduction axis contiguous measured faster on gfx1151 at short streams
— block level, bf16, on 17 of the 21 camera blocks the rule reaches, and flat rather than
slower on the rest. What is tested here is the geometry and the invariants the change rests
on, since the rate itself needs that card: the values survive the relay, the shape does not
move, the layout outlives every door a weight comes through, and quantized weights and
cuBLAS are left alone entirely.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import cuda_only
from thenoise.dit import quantized
from thenoise.dit.quantized import (
    TRANSPOSE_MIN_RATIO,
    QuantizedLinear,
    stores_weight_transposed,
    weight_relayed,
)

#: A pair at the engine's ratio, so the boundary is the definition rather than a number here.
IN = 64
OUT = IN * int(TRANSPOSE_MIN_RATIO)


def checkpoint_of(lin: QuantizedLinear) -> dict:
    """What a safetensors file would hold for ``lin``: plain row-major tensors."""
    return {k: v.detach().contiguous().clone() for k, v in lin.state_dict().items()}


# ------------------------------------------------------------------- the rule

def test_the_rule_starts_where_the_measured_ratio_sits():
    assert stores_weight_transposed(IN, OUT)
    assert not stores_weight_transposed(IN, OUT - 1)


def test_a_projection_that_is_not_wider_out_than_in_wants_nothing():
    assert not stores_weight_transposed(IN, IN)
    assert not stores_weight_transposed(IN, IN // 2)


def test_the_gate_is_the_backend_because_only_rocm_was_measured(monkeypatch):
    monkeypatch.setattr(quantized, "LAYOUT_PENALTY", False)
    assert not stores_weight_transposed(IN, 4 * IN)
    assert weight_relayed(torch.empty(4 * IN, IN)).stride(-1) == 1


# ------------------------------------------------------------------- the relay

def test_the_relay_keeps_the_shape_and_every_value():
    weight = torch.randn(OUT, IN)
    relayed = weight_relayed(weight)
    assert relayed.shape == weight.shape
    assert relayed.stride() == (1, OUT)
    torch.testing.assert_close(relayed, weight)


def test_relaying_a_relayed_weight_copies_nothing():
    # Modules are moved and cast more than once, and the second copy would be for free.
    relayed = weight_relayed(torch.randn(OUT, IN))
    assert weight_relayed(relayed) is relayed


def test_a_weight_the_rule_does_not_reach_is_left_alone():
    plain = torch.randn(IN, IN)
    assert weight_relayed(plain) is plain


def test_a_quantized_weight_is_not_a_store_this_rule_reads():
    # int8 storage is a different tensor in front of a different kernel; nothing was
    # measured about it, so nothing is changed about it.
    qdata = torch.randint(-127, 127, (OUT, IN), dtype=torch.int8)
    assert weight_relayed(qdata) is qdata


def test_a_weight_someone_else_laid_out_is_left_alone():
    strided = torch.randn(IN, OUT).t()
    assert weight_relayed(strided) is strided


# ------------------------------------------------------------------ the module

def test_a_module_of_those_widths_is_built_already_relayed():
    assert QuantizedLinear(IN, OUT).weight.stride() == (1, OUT)


def test_a_module_outside_the_rule_keeps_the_plain_store():
    assert QuantizedLinear(IN, IN).weight.stride(-1) == 1


def test_an_assigned_checkpoint_load_relays_without_losing_a_value():
    source = QuantizedLinear(IN, OUT)
    source.weight.data.normal_()
    source.bias.data.normal_()
    state = checkpoint_of(source)
    assert state["weight"].stride(-1) == 1          # a file has one store, the plain one

    fresh = QuantizedLinear(IN, OUT)
    # ``assign=True`` is how the loader lands weights, and it goes around ``_apply``.
    fresh.load_state_dict(state, strict=True, assign=True)
    torch.testing.assert_close(fresh.weight, source.weight.detach())
    assert fresh.weight.stride() == (1, OUT)


def test_a_copied_checkpoint_load_relays_without_losing_a_value():
    source = QuantizedLinear(IN, OUT)
    source.weight.data.normal_()
    fresh = QuantizedLinear(IN, OUT)
    fresh.load_state_dict(checkpoint_of(source), strict=True)
    torch.testing.assert_close(fresh.weight, source.weight.detach())
    assert fresh.weight.stride() == (1, OUT)


def test_a_quantized_weight_survives_a_move_untouched():
    lin = QuantizedLinear(IN, OUT)
    del lin.weight
    qdata = torch.randint(-127, 127, (OUT, IN), dtype=torch.int8)
    lin.register_buffer("weight", qdata)
    lin._relay_store()
    assert lin.weight is qdata and lin.weight.stride(-1) == 1


def test_a_module_written_back_out_hands_over_the_plain_store(tmp_path):
    # The one way the runtime layout could leak: a strided tensor in a state dict reaches
    # safetensors as a refusal to save at all. Round-tripping through a file is the whole
    # reason this guard exists, so the test writes one.
    from safetensors.torch import load_file, save_file

    lin = QuantizedLinear(IN, OUT)
    lin.weight.data.normal_()
    state = lin.state_dict()
    assert state["weight"].stride(-1) == 1
    save_file(state, str(tmp_path / "lin.safetensors"))
    torch.testing.assert_close(load_file(str(tmp_path / "lin.safetensors"))["weight"],
                               lin.weight.detach())


@cuda_only
def test_the_store_survives_the_move_to_the_device():
    lin = QuantizedLinear(IN, OUT).to("cuda")
    assert lin.weight.stride() == (1, OUT)


@cuda_only
def test_a_dtype_cast_does_not_hand_back_the_plain_store():
    lin = QuantizedLinear(IN, OUT).to("cuda").to(torch.bfloat16)
    assert lin.weight.stride() == (1, OUT)
    assert lin.weight.dtype == torch.bfloat16


@cuda_only
def test_a_relayed_projection_answers_the_plain_one():
    lin = QuantizedLinear(IN, OUT).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(512, IN, device="cuda", dtype=torch.bfloat16)
    truth = F.linear(x.float(), lin.weight.float().contiguous(), lin.bias.float())
    torch.testing.assert_close(lin(x).float(), truth, atol=2e-2, rtol=2e-2)
