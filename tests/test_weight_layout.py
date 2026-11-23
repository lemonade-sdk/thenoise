"""``weight_relayed``: the store a wide-output projection wants its weight kept in.

Where a projection is much wider out than in, its GEMM reads the weight rather than the
activation, and ``F.linear`` reads that weight strided by the row it is stored in. A model
asks for the transposed store at load, where its own blocks were measured wanting it. What
is tested here is the plumbing, since the rate needs the card: the values survive the relay,
the shape does not move, the store arrives through the loader and then outlives every move,
and what is not a wide floating-point weight is left alone.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import cuda_only
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils import layout
from thenoise.utils.layout import (
    TRANSPOSE_MIN_RATIO,
    stores_weight_transposed,
    weight_relayed,
)
from thenoise.utils.loader import load_module_state

#: A pair at the engine's ratio, so the boundary is the definition rather than a number here.
IN = 64
OUT = IN * int(TRANSPOSE_MIN_RATIO)


@pytest.fixture(autouse=True)
def rocm_gate(monkeypatch):
    """Hold the backend gate open, because CI has no ROCm and everything below this line
    but the gate itself is backend-independent plumbing."""
    monkeypatch.setattr(layout, "LAYOUT_PENALTY", True)


def checkpoint_of(lin: QuantizedLinear) -> dict:
    """What a safetensors file would hold for ``lin``: plain row-major tensors."""
    return {k: v.detach().contiguous().clone() for k, v in lin.state_dict().items()}


def filled() -> QuantizedLinear:
    """A projection with something to check a load against, bias included."""
    lin = QuantizedLinear(IN, OUT)
    lin.weight.data.normal_()
    lin.bias.data.normal_()
    return lin


def loaded(source: QuantizedLinear) -> QuantizedLinear:
    """A module whose weight arrived through the loader, with the store opted in."""
    fresh = QuantizedLinear(IN, OUT)
    load_module_state(fresh, checkpoint_of(source), "test", relay_stores=True)
    return fresh


# ------------------------------------------------------------------- the rule

def test_the_rule_starts_where_the_measured_ratio_sits():
    assert stores_weight_transposed(IN, OUT)
    assert not stores_weight_transposed(IN, OUT - 1)


def test_a_projection_that_is_not_wider_out_than_in_wants_nothing():
    assert not stores_weight_transposed(IN, IN)
    assert not stores_weight_transposed(IN, IN // 2)


def test_the_gate_is_the_backend_because_only_rocm_was_measured(monkeypatch):
    monkeypatch.setattr(layout, "LAYOUT_PENALTY", False)   # over the autouse fixture
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
    # A load that runs twice would otherwise pay for a second copy of the model.
    relayed = weight_relayed(torch.randn(OUT, IN))
    assert weight_relayed(relayed) is relayed


def test_a_weight_the_rule_does_not_reach_is_left_alone():
    plain = torch.randn(IN, IN)
    assert weight_relayed(plain) is plain


def test_a_quantized_weight_is_not_a_store_this_rule_reads():
    # int8 storage is a different tensor in front of a different kernel.
    qdata = torch.randint(-127, 127, (OUT, IN), dtype=torch.int8)
    assert weight_relayed(qdata) is qdata


def test_a_weight_someone_else_laid_out_is_left_alone():
    strided = torch.randn(IN, OUT).t()
    assert weight_relayed(strided) is strided


# ------------------------------------------------------------------ the doors

def test_a_module_is_built_in_the_store_a_checkpoint_ships():
    # The store is the loader's decision: a projection built at runtime has no measurement.
    assert QuantizedLinear(IN, OUT).weight.stride(-1) == 1


def test_the_load_lands_the_store_a_model_asked_for():
    source = filled()
    state = checkpoint_of(source)
    assert state["weight"].stride(-1) == 1          # a file has one store, the plain one

    fresh = loaded(source)
    torch.testing.assert_close(fresh.weight, source.weight.detach())
    torch.testing.assert_close(fresh.bias, source.bias.detach())
    assert fresh.weight.stride() == (1, OUT)


def test_the_load_keeps_the_plain_store_when_a_model_asks_nothing():
    source = filled()
    fresh = QuantizedLinear(IN, OUT)
    load_module_state(fresh, checkpoint_of(source), "test")
    torch.testing.assert_close(fresh.weight, source.weight.detach())
    assert fresh.weight.stride(-1) == 1


def test_the_relay_loop_leaves_alone_what_is_not_a_weight():
    # A quantized checkpoint's scales travel in the same dict.
    source = filled()
    state = checkpoint_of(source)
    state["scale"] = torch.arange(OUT, dtype=torch.int32)
    fresh = QuantizedLinear(IN, OUT)
    fresh.register_buffer("scale", torch.zeros(OUT, dtype=torch.int32))
    load_module_state(fresh, state, "test", relay_stores=True)
    torch.testing.assert_close(fresh.scale, state["scale"])
    assert fresh.weight.stride() == (1, OUT)


def test_a_module_written_back_out_hands_over_the_plain_store(tmp_path):
    # The one way the runtime layout could leak, since safetensors refuses a strided
    # tensor outright. Writing a file is the test.
    from safetensors.torch import load_file, save_file

    lin = loaded(filled())
    assert lin.weight.stride() == (1, OUT)
    state = lin.state_dict()
    assert state["weight"].stride(-1) == 1
    save_file(state, str(tmp_path / "lin.safetensors"))
    torch.testing.assert_close(load_file(str(tmp_path / "lin.safetensors"))["weight"],
                               lin.weight.detach())


# ------------------------------------------------------------------- the moves

@cuda_only
def test_the_store_survives_the_move_to_the_device():
    lin = loaded(filled()).to("cuda")
    assert lin.weight.stride() == (1, OUT)


@cuda_only
def test_a_dtype_cast_does_not_hand_back_the_plain_store():
    lin = loaded(filled()).to("cuda").to(torch.bfloat16)
    assert lin.weight.stride() == (1, OUT)
    assert lin.weight.dtype == torch.bfloat16


@cuda_only
def test_a_relayed_projection_answers_the_plain_one():
    source = filled()
    lin = loaded(source).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(512, IN, device="cuda", dtype=torch.bfloat16)
    truth = F.linear(x.float(), lin.weight.float().contiguous(), lin.bias.float())
    torch.testing.assert_close(lin(x).float(), truth, atol=2e-2, rtol=2e-2)
