"""The Qwen3.5 conditioner's conv kernel: FLA's, reached through the seam transformers
leaves for ``causal_conv1d``, and what happens where FLA cannot be reached at all.
"""
from __future__ import annotations

import sys

import pytest
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35

from thenoise.utils.text_encoder import (
    CONV_ACTIVATIONS,
    _PATCH_MARK,
    _fla_conv,
    apply_qwen35_kernels,
)


def test_the_conv_wrapper_hands_the_kernel_the_projection_and_answers_channels_first():
    captured = {}

    def fwd(x, weight, bias, residual, activation=None):
        captured.update(x=x, activation=activation)
        return x * 3, None

    def shipped(*args, **kwargs):
        raise AssertionError("the shipped conv was taken")

    projection = torch.randn(1, 9, 4)  # what the layer's in_proj_qkv writes
    out = _fla_conv(shipped, fwd)(
        projection.transpose(1, 2), torch.randn(4, 4), None, activation="silu"
    )

    assert captured["x"].shape == (1, 9, 4)  # the kernel's own layout...
    assert captured["x"].data_ptr() == projection.data_ptr()  # ...reached without a copy
    assert captured["activation"] == "silu"
    assert out.shape == (1, 4, 9)
    assert torch.equal(out, projection.transpose(1, 2) * 3)


def test_an_activation_the_conv_kernel_does_not_apply_stays_on_the_shipped_conv():
    taken = []

    def fwd(x, weight, bias, residual, activation=None):
        taken.append(activation)
        return x, None

    def shipped(hidden_states, weight, bias=None, activation=None, **kwargs):
        taken.append(("shipped", activation))
        return hidden_states

    conv = _fla_conv(shipped, fwd)
    x, w = torch.randn(1, 4, 9), torch.randn(4, 4)
    assert conv(x, w, None, "gelu") is x
    conv(x, w, None, "swish")

    assert taken == [("shipped", "gelu"), "swish"]
    assert "gelu" not in CONV_ACTIVATIONS


def test_the_conv_kernel_is_installed_once_and_a_moved_seam_costs_only_speed():
    pytest.importorskip("fla.modules.causal_conv1d")

    shipped = qwen35.causal_conv1d_fn
    try:
        assert apply_qwen35_kernels() is True
        installed = qwen35.causal_conv1d_fn
        assert installed is not shipped
        assert getattr(installed, _PATCH_MARK, False) is True
        assert apply_qwen35_kernels() is True
        assert qwen35.causal_conv1d_fn is installed  # not wrapped a second time

        del qwen35.causal_conv1d_fn
        assert apply_qwen35_kernels() is False
    finally:
        qwen35.causal_conv1d_fn = shipped


def test_a_torch_build_without_triton_keeps_the_shipped_conv(monkeypatch):
    monkeypatch.setitem(sys.modules, "fla.modules.causal_conv1d", None)
    assert apply_qwen35_kernels() is False
