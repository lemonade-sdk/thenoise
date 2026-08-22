"""INT8+ConvRot support tests.

These cover the shared ``QuantizedLinear`` module and the generic INT8 loading
helpers. They run on CPU (comfy_kitchen's eager backend) and need no GPU or real
checkpoints.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.int8 import is_int8_checkpoint, load_int8_state_dict

# --------------------------------------------------------------------------- QuantizedLinear


def test_quantized_linear_bf16_forward():
    layer = QuantizedLinear(256, 512, bias=False)
    torch.nn.init.ones_(layer.weight)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (4, 512)
    assert out.dtype == torch.bfloat16
    assert not layer._int8


def test_quantized_linear_int8_forward():
    layer = QuantizedLinear(256, 512, bias=False)
    qweight = torch.randint(-127, 127, (512, 256), dtype=torch.int8)
    scale = torch.rand(512, 1, dtype=torch.float32)
    layer.load_int8(qweight, scale)
    assert layer._int8
    assert layer.weight is None
    assert layer.qweight.dtype == torch.int8
    assert layer.scale.dtype == torch.float32

    x = torch.randn(4, 256, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (4, 512)
    assert out.dtype == torch.bfloat16


def test_quantized_linear_load_int8_frees_bf16_weight():
    layer = QuantizedLinear(256, 512)
    assert layer.weight is not None
    layer.load_int8(torch.zeros(512, 256, dtype=torch.int8), torch.zeros(512, 1))
    assert layer.weight is None  # bf16 weight dropped -> memory savings


def test_quantized_linear_int8_lora_residual():
    layer = QuantizedLinear(256, 512)
    layer.load_int8(
        torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        torch.rand(512, 1),
    )
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    base = layer(x)  # without LoRA

    # LoRA factors: down [r, in], up [out, r]
    down = torch.randn(8, 256, dtype=torch.bfloat16)
    up = torch.randn(512, 8, dtype=torch.bfloat16)
    layer.apply_lora(down, up, alpha=8.0, multiplier=0.5, calc_device=torch.device("cpu"))
    assert layer._lora is True

    out = layer(x)
    # expected residual: x @ down^T @ up^T * (alpha/r * multiplier)
    expected = base + x @ (down * (8.0 / 8 * 0.5)).t() @ up.t()
    assert torch.allclose(out.float(), expected.float(), rtol=1e-2, atol=1e-2)

    layer.clear_lora()
    assert layer._lora is False
    assert torch.allclose(layer(x).float(), base.float(), rtol=1e-2, atol=1e-2)


class _TinyInt8Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = QuantizedLinear(256, 512)
        self.q.load_int8(
            torch.randint(-127, 127, (512, 256), dtype=torch.int8),
            torch.rand(512, 1),
        )
        self.plain = nn.Linear(256, 512)


def _int8_lora_sd():
    return {
        "q.lora_down.weight": torch.randn(8, 256, dtype=torch.bfloat16),
        "q.lora_up.weight": torch.randn(512, 8, dtype=torch.bfloat16),
        "q.alpha": torch.tensor(8.0),
    }


def test_apply_lora_to_model_int8_residual():
    from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model

    model = _TinyInt8Model()
    result = apply_lora_to_model(
        model, [_int8_lora_sd()], [1.0], torch.device("cpu")
    )
    assert model.q._lora is True
    assert result["int8_affected"] == ("q",)

    undo_lora_on_model(model, result, torch.device("cpu"))
    assert model.q._lora is False


def test_apply_lora_to_model_mixed_int8_and_bf16():
    from thenoise.utils.lora import apply_lora_to_model

    model = _TinyInt8Model()
    # LoRA targeting both the int8 q layer and the bf16 plain layer
    sd = {
        "q.lora_down.weight": torch.randn(8, 256, dtype=torch.bfloat16),
        "q.lora_up.weight": torch.randn(512, 8, dtype=torch.bfloat16),
        "q.alpha": torch.tensor(8.0),
        "plain.lora_down.weight": torch.randn(8, 256, dtype=torch.bfloat16),
        "plain.lora_up.weight": torch.randn(512, 8, dtype=torch.bfloat16),
        "plain.alpha": torch.tensor(8.0),
    }
    result = apply_lora_to_model(model, [sd], [1.0], torch.device("cpu"))
    assert model.q._lora is True
    assert model.plain.weight is not None  # bf16 layer mutated normally
    assert "plain.weight" in result["affected_keys"]
    assert result["int8_affected"] == ("q",)



# --------------------------------------------------------------------------- is_int8_checkpoint


def _write(path, tensors):
    from safetensors.torch import save_file
    save_file(tensors, str(path))


def test_is_int8_checkpoint_true(tmp_path):
    p = tmp_path / "int8.safetensors"
    _write(p, {
        "model.diffusion_model.blocks.0.cross_attn.k_proj.weight": torch.zeros(16, 8, dtype=torch.int8),
        "model.diffusion_model.blocks.0.cross_attn.k_proj.weight_scale": torch.zeros(16, 1, dtype=torch.float32),
        "model.diffusion_model.blocks.0.cross_attn.k_proj.comfy_quant": torch.zeros(8, dtype=torch.uint8),
    })
    assert is_int8_checkpoint(str(p)) is True


def test_is_int8_checkpoint_false_for_bf16(tmp_path):
    p = tmp_path / "bf16.safetensors"
    _write(p, {
        "blocks.0.cross_attn.k_proj.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
    })
    assert is_int8_checkpoint(str(p)) is False


# --------------------------------------------------------------------------- load_int8_state_dict


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = QuantizedLinear(256, 512, bias=False)
        self.plain = nn.Linear(256, 512, bias=True)


def _tiny_state_dict():
    return {
        "q.weight": torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        "q.weight_scale": torch.rand(512, 1, dtype=torch.float32),
        "q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    }


def test_load_int8_state_dict_mixed():
    model = _TinyModel()
    load_int8_state_dict(model, _tiny_state_dict())

    # quantized layer switched to INT8
    assert model.q._int8 is True
    assert model.q.weight is None
    assert model.q.qweight.dtype == torch.int8
    assert model.q.scale.dtype == torch.float32

    # bf16 layer assigned normally
    assert model.plain.weight.dtype == torch.bfloat16
    assert model.plain.bias.dtype == torch.bfloat16

    # forward runs end-to-end (bf16 in -> bf16 out)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    assert model.q(x).shape == (4, 512)
    assert model.q(x).dtype == torch.bfloat16


def test_load_int8_state_dict_missing_scale_raises():
    sd = _tiny_state_dict()
    del sd["q.weight_scale"]
    with pytest.raises(RuntimeError, match="missing its .weight_scale"):
        load_int8_state_dict(_TinyModel(), sd)


def test_load_int8_state_dict_orphan_scale_raises():
    sd = _tiny_state_dict()
    sd["plain.weight_scale"] = torch.zeros(512, 1, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="orphan"):
        load_int8_state_dict(_TinyModel(), sd)


# --------------------------------------------------------------------------- real checkpoint (optional)

@pytest.mark.skipif(
    not os.path.exists("models/anima/split_files/diffusion_models/anima-turbo-v1.0-int8convrot.safetensors"),
    reason="real INT8 Anima checkpoint not present",
)
def test_real_int8_checkpoint_is_detected():
    from thenoise.models import resolve
    assert resolve("models/anima/split_files/diffusion_models/anima-turbo-v1.0-int8convrot.safetensors") is not None
    assert is_int8_checkpoint("models/anima/split_files/diffusion_models/anima-turbo-v1.0-int8convrot.safetensors") is True
