"""``pitch_padded``: the operand layout that costs a full-precision GEMM half its rate.

A GEMM whose activation rows are 32768 bytes apart was measured on gfx1151 at half the rate
the same block charges for its other projections, and sixteen bytes per row recovered it.
The geometry is what is tested here, since the penalty itself needs that card: the pitch
that wants widening, the pitch a producer already widened, the row count below which
widening costs rather than earns, and that a widened operand still answers the same question.
"""
from __future__ import annotations

import torch

from conftest import cuda_only
from thenoise.dit import quantized
from thenoise.dit.quantized import (
    PITCH_MIN_BYTES,
    PITCH_MIN_ROWS,
    PITCH_PAD_ELEMENTS,
    QuantizedLinear,
    pitch_padded,
    pitch_too_wide,
)

WIDTH = PITCH_MIN_BYTES // 2  # bf16 columns at the pitch the penalty is measured on


def _wide(*rows: int) -> torch.Tensor:
    """A bf16 operand of the disease pitch with ``rows`` rows before the last dimension."""
    return torch.zeros(*rows, WIDTH, dtype=torch.bfloat16)


def test_a_row_the_penalty_is_measured_on_is_flagged():
    assert pitch_too_wide(_wide(PITCH_MIN_ROWS))


def test_a_shorter_row_is_left_alone():
    assert not pitch_too_wide(torch.zeros(PITCH_MIN_ROWS, WIDTH - 1, dtype=torch.bfloat16))


def test_a_row_someone_already_widened_is_not_widened_again():
    wide = torch.zeros(PITCH_MIN_ROWS, WIDTH + PITCH_PAD_ELEMENTS, dtype=torch.bfloat16)
    assert not pitch_too_wide(wide[..., :WIDTH])


def test_a_short_stream_is_left_alone_however_wide_its_rows():
    """M=512 was measured losing 14% to widening, where M=4096 gains 16%."""
    assert not pitch_too_wide(_wide(PITCH_MIN_ROWS - 1))


def test_rows_a_batch_dimension_hides_are_still_rows():
    assert pitch_too_wide(_wide(4, PITCH_MIN_ROWS // 4))


def test_a_vector_has_no_row_pitch_to_fix():
    assert not pitch_too_wide(torch.zeros(PITCH_MIN_BYTES, dtype=torch.bfloat16))


def test_a_cpu_operand_is_left_alone_however_wide_its_rows():
    x = _wide(PITCH_MIN_ROWS)
    assert pitch_padded(x) is x


@cuda_only
def test_widening_adds_to_the_row_stride_and_not_to_the_shape():
    x = torch.randn(2, PITCH_MIN_ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    y = pitch_padded(x)
    assert y.shape == x.shape
    assert y.stride(-2) * y.element_size() == PITCH_MIN_BYTES + PITCH_PAD_ELEMENTS * 2
    assert y.stride(-1) == 1
    assert torch.equal(y, x)


@cuda_only
def test_a_short_stream_keeps_its_operand_on_device_too():
    x = torch.randn(512, WIDTH, device="cuda", dtype=torch.bfloat16)
    assert pitch_padded(x) is x


@cuda_only
def test_the_gate_hands_the_operand_back_to_a_backend_without_the_penalty(monkeypatch):
    monkeypatch.setattr(quantized, "LAYOUT_PENALTY", False)
    x = torch.randn(PITCH_MIN_ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    assert pitch_padded(x) is x


@cuda_only
def test_a_wide_bf16_layer_answers_the_same_as_the_gemm_alone():
    """The kernel a widened operand selects rounds differently; it does not disagree."""
    torch.manual_seed(0)
    layer = QuantizedLinear(WIDTH, WIDTH // 2).to("cuda", torch.bfloat16).eval()
    x = torch.randn(PITCH_MIN_ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        padded, plain = layer(x), torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert (padded - plain).abs().max() <= plain.abs().max() * 2e-2
