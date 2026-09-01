"""Quantized (INT8/FP8) support tests.

These cover the shared ``QuantizedLinear`` module (backed by comfy_kitchen's
``QuantizedTensor``) and the generic quantized loading helpers. They run on CPU
(comfy_kitchen's eager backend) and need no GPU or real checkpoints.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.loader import (
    is_quantized_checkpoint,
    load_quantized_state_dict,
)


def _int8_qt(qweight, scale, convrot=True, groupsize=256):
    """Wrap int8 weight + scale into a TensorWiseINT8Layout QuantizedTensor."""
    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qweight.shape),
        is_weight=True,
        convrot=convrot,
        convrot_groupsize=groupsize,
    )
    return QuantizedTensor(qweight, "TensorWiseINT8Layout", params)


def _fp8_qt(qweight, scale):
    """Wrap an fp8 weight + per-tensor scale into a TensorCoreFP8Layout QuantizedTensor."""
    params = TensorCoreFP8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qweight.shape),
    )
    return QuantizedTensor(qweight, "TensorCoreFP8Layout", params)


# --------------------------------------------------------------------------- QuantizedLinear (INT8 raw-op path)


def test_quantized_linear_bf16_forward():
    layer = QuantizedLinear(256, 512, bias=False)
    torch.nn.init.ones_(layer.weight)
    layer = layer.to(torch.bfloat16)  # models are cast to bf16 by the adapter
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (4, 512)
    assert out.dtype == torch.bfloat16
    assert not layer._quantized


def test_quantized_linear_int8_forward():
    layer = QuantizedLinear(256, 512, bias=False)
    qweight = torch.randint(-127, 127, (512, 256), dtype=torch.int8)
    scale = torch.rand(512, 1, dtype=torch.float32)
    layer.load_quantized(_int8_qt(qweight, scale))
    assert layer._quantized
    assert isinstance(layer.weight, QuantizedTensor)
    assert layer.weight._qdata.dtype == torch.int8
    assert layer.weight.params.scale.dtype == torch.float32

    x = torch.randn(4, 256, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (4, 512)
    assert out.dtype == torch.bfloat16


def test_quantized_linear_load_quantized_frees_bf16_weight():
    layer = QuantizedLinear(256, 512)
    assert isinstance(layer.weight, nn.Parameter)
    layer.load_quantized(_int8_qt(torch.zeros(512, 256, dtype=torch.int8), torch.zeros(512, 1)))
    # bf16 parameter dropped (weight is now the quantized buffer) -> memory savings
    assert isinstance(layer.weight, QuantizedTensor)


def test_quantized_linear_int8_bake_lora():
    layer = QuantizedLinear(256, 512)
    layer.load_quantized(
        _int8_qt(torch.randint(-127, 127, (512, 256), dtype=torch.int8), torch.rand(512, 1))
    )
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    base = layer(x)  # without LoRA
    orig_q = layer.weight._qdata.clone()

    # LoRA factors: down [r, in], up [out, r]. The baked delta is
    # (up @ down) * (alpha/r * multiplier), shaped [out, in].
    down = torch.randn(8, 256, dtype=torch.bfloat16) * 6
    up = torch.randn(512, 8, dtype=torch.bfloat16) * 6
    alpha, multiplier = 8.0, 2.0
    delta = (up @ down) * (alpha / down.size(0) * multiplier)
    layer.bake_lora(delta)
    assert not torch.equal(layer.weight._qdata, orig_q)  # baked into int8 weights

    out = layer(x)
    # The INT8 GEMM also quantizes activations, so an exact match to
    # base + x @ delta^T is impossible; instead verify the LoRA moves the output
    # substantially and lands much closer to the delta expectation than to base.
    err_to_expected = (out.float() - (base + x @ delta.t()).float()).abs().max()
    err_to_base = (out.float() - base.float()).abs().max()
    assert err_to_base > 1.0  # LoRA has a clear effect
    assert err_to_expected < err_to_base  # output follows the delta direction


# --------------------------------------------------------------------------- QuantizedLinear (FP8 fallback path)


def test_quantized_linear_fp8_forward():
    layer = QuantizedLinear(256, 512, bias=False)
    # Proper fp8 E4M3 weight + per-tensor scale.
    qweight = torch.randn(512, 256, dtype=torch.bfloat16)
    scale = torch.tensor(qweight.abs().amax().item() / 448.0, dtype=torch.float32)
    fp8 = qweight.to(torch.float8_e4m3fn)
    layer.load_quantized(_fp8_qt(fp8, scale))
    assert layer._quantized
    assert isinstance(layer.weight, QuantizedTensor)
    assert layer.weight._qdata.dtype == torch.float8_e4m3fn

    x = torch.randn(4, 256, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (4, 512)
    assert out.dtype == torch.bfloat16


def test_quantized_linear_fp8_bake_lora():
    layer = QuantizedLinear(256, 512)
    qweight = torch.randn(512, 256, dtype=torch.bfloat16)
    scale = torch.tensor(qweight.abs().amax().item() / 448.0, dtype=torch.float32)
    layer.load_quantized(_fp8_qt(qweight.to(torch.float8_e4m3fn), scale))
    orig_q = layer.weight._qdata.clone()

    down = torch.randn(8, 256, dtype=torch.bfloat16) * 6
    up = torch.randn(512, 8, dtype=torch.bfloat16) * 6
    delta = (up @ down) * (8.0 / down.size(0) * 2.0)
    layer.bake_lora(delta)
    assert not torch.equal(layer.weight._qdata, orig_q)  # requantized fp8
    assert layer.weight._qdata.dtype == torch.float8_e4m3fn  # profile preserved


class _TinyInt8Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = QuantizedLinear(256, 512)
        self.q.load_quantized(
            _int8_qt(torch.randint(-127, 127, (512, 256), dtype=torch.int8), torch.rand(512, 1))
        )
        self.plain = nn.Linear(256, 512)


def _int8_lora_sd():
    return {
        "q.lora_down.weight": torch.randn(8, 256, dtype=torch.bfloat16),
        "q.lora_up.weight": torch.randn(512, 8, dtype=torch.bfloat16),
        "q.alpha": torch.tensor(8.0),
    }


def test_apply_lora_to_model_int8_bake_and_restore(tmp_path):
    from thenoise.utils.loader import load_dit
    from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model

    # Write an INT8 checkpoint matching ``_TinyInt8Model`` and load it through
    # ``load_dit`` so the raw-key restore map is built on the model.
    qweight = torch.randint(-127, 127, (512, 256), dtype=torch.int8)
    scale = torch.rand(512, 1)
    p = tmp_path / "int8.safetensors"
    _write(p, {
        "q.weight": qweight,
        "q.weight_scale": scale,
        "q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _TinyInt8Model()
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)
    orig_q = model.q.weight._qdata.clone()

    result = apply_lora_to_model(
        model, [_int8_lora_sd()], [1.0], torch.device("cpu"), dit_path=str(p)
    )
    assert result["quantized_affected"] == ("q",)
    assert result["quantized_restore_keys"] == ("q.weight",)
    assert result["dit_path"] == str(p)
    assert not torch.equal(model.q.weight._qdata, orig_q)  # baked into int8 weights

    # Undo reloads the original INT8 weights from disk by raw key.
    undo_lora_on_model(model, result, torch.device("cpu"))
    assert torch.equal(model.q.weight._qdata, orig_q)
    assert torch.equal(model.q.weight.params.scale, scale)


def test_apply_lora_to_model_int8_undo_without_dit_path_raises(tmp_path):
    from thenoise.utils.loader import load_dit
    from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model

    # Load a real checkpoint (restore keys captured), but apply without a
    # dit_path so undo has no file to reload from.
    p = tmp_path / "int8.safetensors"
    _write(p, {
        "q.weight": torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        "q.weight_scale": torch.rand(512, 1),
        "q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _TinyInt8Model()
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)

    result = apply_lora_to_model(model, [_int8_lora_sd()], [1.0], torch.device("cpu"))
    assert result["quantized_restore_keys"] == ("q.weight",)  # captured from the model
    with pytest.raises(RuntimeError, match="no dit_path"):
        undo_lora_on_model(model, result, torch.device("cpu"))


def test_apply_lora_to_model_int8_wrapped_restore(tmp_path):
    from thenoise.utils.loader import load_dit
    from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model

    # Repackaged checkpoint with the generic ``model.diffusion_model.`` prefix;
    # the restore map must strip it so the captured raw key still finds the
    # tensor on undo.
    qweight = torch.randint(-127, 127, (512, 256), dtype=torch.int8)
    scale = torch.rand(512, 1)
    p = tmp_path / "int8_wrapped.safetensors"
    _write(p, {
        "model.diffusion_model.q.weight": qweight,
        "model.diffusion_model.q.weight_scale": scale,
        "model.diffusion_model.q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "model.diffusion_model.plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "model.diffusion_model.plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _TinyInt8Model()
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)
    orig_q = model.q.weight._qdata.clone()

    result = apply_lora_to_model(
        model, [_int8_lora_sd()], [1.0], torch.device("cpu"), dit_path=str(p)
    )
    assert result["quantized_restore_keys"] == ("model.diffusion_model.q.weight",)
    assert not torch.equal(model.q.weight._qdata, orig_q)

    undo_lora_on_model(model, result, torch.device("cpu"))
    assert torch.equal(model.q.weight._qdata, orig_q)


def test_apply_lora_to_model_mixed_int8_and_bf16():
    from thenoise.utils.lora import apply_lora_to_model

    model = _TinyInt8Model()
    orig_q = model.q.weight._qdata.clone()
    orig_plain = model.plain.weight.clone()
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
    assert not torch.equal(model.q.weight._qdata, orig_q)  # int8 layer baked
    assert not torch.equal(model.plain.weight, orig_plain)  # bf16 layer mutated
    assert "plain.weight" in result["affected_keys"]
    assert result["quantized_affected"] == ("q",)


def test_apply_lora_to_model_fp8_bake_and_restore(tmp_path):
    from thenoise.utils.loader import load_dit
    from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model

    # FP8 checkpoint with a per-tensor scale (like comfy-quants fp8_e4m3).
    qweight = torch.randn(512, 256, dtype=torch.bfloat16)
    fp8 = qweight.to(torch.float8_e4m3fn)
    scale = torch.tensor(0.5, dtype=torch.float32)
    p = tmp_path / "fp8.safetensors"
    _write(p, {
        "q.weight": fp8,
        "q.weight_scale": scale,
        "q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _TinyInt8Model()  # same shape layout; q will load as fp8
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)
    assert model.q._quantized
    assert model.q.weight._qdata.dtype == torch.float8_e4m3fn
    orig_q = model.q.weight._qdata.clone()

    result = apply_lora_to_model(
        model, [_int8_lora_sd()], [1.0], torch.device("cpu"), dit_path=str(p)
    )
    assert result["quantized_restore_keys"] == ("q.weight",)
    assert not torch.equal(model.q.weight._qdata, orig_q)  # baked fp8

    undo_lora_on_model(model, result, torch.device("cpu"))
    assert torch.equal(model.q.weight._qdata, orig_q)
    assert torch.equal(model.q.weight.params.scale, scale)


# --------------------------------------------------------------------------- is_quantized_checkpoint


def _write(path, tensors):
    from safetensors.torch import save_file
    save_file(tensors, str(path))


def test_is_quantized_checkpoint_true_int8(tmp_path):
    p = tmp_path / "int8.safetensors"
    _write(p, {
        "model.diffusion_model.blocks.0.cross_attn.k_proj.weight": torch.zeros(16, 8, dtype=torch.int8),
        "model.diffusion_model.blocks.0.cross_attn.k_proj.weight_scale": torch.zeros(16, 1, dtype=torch.float32),
        "model.diffusion_model.blocks.0.cross_attn.k_proj.comfy_quant": torch.zeros(8, dtype=torch.uint8),
    })
    assert is_quantized_checkpoint(str(p)) is True


def test_is_quantized_checkpoint_true_fp8(tmp_path):
    p = tmp_path / "fp8.safetensors"
    _write(p, {
        "blocks.0.cross_attn.k_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
        "blocks.0.cross_attn.k_proj.weight_scale": torch.tensor(0.5, dtype=torch.float32),
        "blocks.0.cross_attn.k_proj.comfy_quant": torch.zeros(8, dtype=torch.uint8),
    })
    assert is_quantized_checkpoint(str(p)) is True


def test_is_quantized_checkpoint_false_for_bf16(tmp_path):
    p = tmp_path / "bf16.safetensors"
    _write(p, {
        "blocks.0.cross_attn.k_proj.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
    })
    assert is_quantized_checkpoint(str(p)) is False


def _comfy_quant(convrot=True, groupsize=256):
    """Build a ``comfy_quant`` marker tensor like ComfyUI's INT8 exporter."""
    import json

    conf = {"convrot": bool(convrot)}
    if convrot:
        conf["convrot_groupsize"] = int(groupsize)
    conf["per_row"] = True
    data = json.dumps(conf).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


def test_comfy_quant_marker_sets_groupsize(tmp_path):
    from thenoise.utils.loader import load_dit

    # A layer quantized with convrot_groupsize=64 must rotate activations with
    # 64 at inference, NOT the default 256, or the images are garbage. The
    # group size is read from the layer's comfy_quant marker tensor.
    p = tmp_path / "int8.safetensors"
    _write(p, {
        "q.weight": torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        "q.weight_scale": torch.rand(512, 1),
        "q.comfy_quant": _comfy_quant(convrot=True, groupsize=64),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _TinyModel()
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)
    assert model.q._quantized is True
    assert model.q.weight.params.convrot is True
    assert model.q.weight.params.convrot_groupsize == 64


def test_comfy_quant_marker_convrot_false(tmp_path):
    from thenoise.utils.loader import load_dit

    # A layer whose in_features were not divisible by the group size is NOT
    # ConvRot-rotated: the marker records convrot=false, and inference must not
    # rotate activations (default is convrot=true).
    p = tmp_path / "int8.safetensors"
    _write(p, {
        "q.weight": torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        "q.weight_scale": torch.rand(512, 1),
        "q.comfy_quant": _comfy_quant(convrot=False),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _TinyModel()
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)
    assert model.q._quantized is True
    assert model.q.weight.params.convrot is False
    assert model.q.weight.params.convrot_groupsize == 256  # irrelevant when convrot=False


def test_comfy_quant_marker_defaults(tmp_path):
    from thenoise.utils.loader import load_dit

    # No marker (or an unparseable one) keeps the default convrot profile.
    p = tmp_path / "int8.safetensors"
    _write(p, {
        "q.weight": torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        "q.weight_scale": torch.rand(512, 1),
        "q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _TinyModel()
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)
    assert model.q._quantized is True
    assert model.q.weight.params.convrot is True
    assert model.q.weight.params.convrot_groupsize == 256


# --------------------------------------------------------------------------- load_quantized_state_dict


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = QuantizedLinear(256, 512, bias=False)
        self.plain = nn.Linear(256, 512, bias=True)


def _tiny_state_dict():
    return {
        "q.weight": torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        "q.weight_scale": torch.rand(512, 1, dtype=torch.float32),
        "q.comfy_quant": _comfy_quant(convrot=True, groupsize=256),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    }


def test_load_quantized_state_dict_mixed():
    model = _TinyModel()
    load_quantized_state_dict(model, _tiny_state_dict())

    # quantized layer switched to INT8
    assert model.q._quantized is True
    assert isinstance(model.q.weight, QuantizedTensor)
    assert model.q.weight._qdata.dtype == torch.int8
    assert model.q.weight.params.scale.dtype == torch.float32

    # bf16 layer assigned normally
    assert model.plain.weight.dtype == torch.bfloat16
    assert model.plain.bias.dtype == torch.bfloat16

    # forward runs end-to-end (bf16 in -> bf16 out)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    assert model.q(x).shape == (4, 512)
    assert model.q(x).dtype == torch.bfloat16


def test_load_quantized_state_dict_fp8():
    model = _TinyModel()
    sd = _tiny_state_dict()
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    sd["q.weight"] = w.to(torch.float8_e4m3fn)
    sd["q.weight_scale"] = torch.tensor(0.5, dtype=torch.float32)
    load_quantized_state_dict(model, sd)

    assert model.q._quantized is True
    assert model.q.weight._qdata.dtype == torch.float8_e4m3fn
    assert model.q.weight.params.scale.shape == ()  # per-tensor scale

    x = torch.randn(4, 256, dtype=torch.bfloat16)
    assert model.q(x).shape == (4, 512)
    assert model.q(x).dtype == torch.bfloat16


def test_load_quantized_state_dict_missing_scale_raises():
    sd = _tiny_state_dict()
    del sd["q.weight_scale"]
    with pytest.raises(RuntimeError, match="missing its .weight_scale"):
        load_quantized_state_dict(_TinyModel(), sd)


def test_load_quantized_state_dict_orphan_scale_raises():
    sd = _tiny_state_dict()
    sd["plain.weight_scale"] = torch.zeros(512, 1, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="orphan"):
        load_quantized_state_dict(_TinyModel(), sd)


# --------------------------------------------------------------------------- load_dit (central quant-aware loader)


def test_load_dit_int8(tmp_path):
    from thenoise.utils.loader import load_dit

    p = tmp_path / "int8.safetensors"
    _write(p, {
        "model.diffusion_model.q.weight": torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        "model.diffusion_model.q.weight_scale": torch.rand(512, 1),
        "model.diffusion_model.q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "model.diffusion_model.plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "model.diffusion_model.plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _TinyModel()
    out = load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)
    assert out is model
    assert model.q._quantized is True
    assert model.q.weight._qdata.dtype == torch.int8
    assert model.plain.weight.dtype == torch.bfloat16


def test_load_dit_fp8(tmp_path):
    from thenoise.utils.loader import load_dit

    p = tmp_path / "fp8.safetensors"
    _write(p, {
        "model.diffusion_model.q.weight": torch.randn(512, 256, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
        "model.diffusion_model.q.weight_scale": torch.tensor(0.5, dtype=torch.float32),
        "model.diffusion_model.q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "model.diffusion_model.plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "model.diffusion_model.plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _TinyModel()
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)
    assert model.q._quantized is True
    assert model.q.weight._qdata.dtype == torch.float8_e4m3fn
    assert model.q(x := torch.randn(4, 256, dtype=torch.bfloat16)).shape == (4, 512)


def test_load_dit_bf16(tmp_path):
    from thenoise.utils.loader import load_dit

    p = tmp_path / "bf16.safetensors"
    _write(p, _bf16_sd())
    model = _TinyModel()
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16)
    assert model.q._quantized is False
    assert model.q.weight is not None  # bf16 layer assigned normally
    assert model.q.weight.dtype == torch.bfloat16
    assert model.plain.weight.dtype == torch.bfloat16


def test_load_dit_key_map(tmp_path):
    from thenoise.utils.loader import load_dit

    # ComfyUI's INT8 exporter stores norm ``scale`` params under ``weight``.
    p = tmp_path / "int8.safetensors"
    _write(p, {
        "q.weight": torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        "q.weight_scale": torch.rand(512, 1),
        "q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "norm.weight": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _ScaleModel()
    key_map = lambda k: k[: -len(".weight")] + ".scale" if k.endswith("norm.weight") else k
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16, key_map=key_map)
    assert model.q._quantized is True
    assert model.norm.scale.dtype == torch.bfloat16


class _ScaleNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class _ScaleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = QuantizedLinear(256, 512)
        self.norm = _ScaleNorm(512)


def test_load_dit_drop_keys_on_bf16(tmp_path):
    from thenoise.utils.loader import load_dit

    p = tmp_path / "bf16.safetensors"
    sd = _bf16_sd()
    sd["last.down.residual"] = torch.randn(16, dtype=torch.bfloat16)
    sd["last.up.residual"] = torch.randn(16, dtype=torch.bfloat16)
    _write(p, sd)
    model = _TinyModel()
    # Without drop_keys the strict load would fail on the unexpected keys.
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16, drop_keys=("last.down", "last.up"))
    assert model.q._quantized is False
    assert model.plain.weight.dtype == torch.bfloat16


def test_load_dit_drop_keys_on_int8(tmp_path):
    from thenoise.utils.loader import load_dit

    p = tmp_path / "int8.safetensors"
    sd = {
        "q.weight": torch.randint(-127, 127, (512, 256), dtype=torch.int8),
        "q.weight_scale": torch.rand(512, 1),
        "q.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
        "last.down.residual": torch.randn(16, dtype=torch.bfloat16),
        "last.up.residual": torch.randn(16, dtype=torch.bfloat16),
    }
    _write(p, sd)
    model = _TinyModel()
    # drop_keys applies to the quantized path too.
    load_dit(model, str(p), device="cpu", dtype=torch.bfloat16, drop_keys=("last.down", "last.up"))
    assert model.q._quantized is True
    assert model.plain.weight.dtype == torch.bfloat16


class _BufModel(nn.Module):
    """Model with an internal buffer not saved in the checkpoint."""

    def __init__(self):
        super().__init__()
        self.plain = nn.Linear(256, 512, bias=True)
        self.register_buffer("rope_seq", torch.zeros(128))


def test_load_dit_expected_missing(tmp_path):
    from thenoise.utils.loader import load_dit

    p = tmp_path / "bf16.safetensors"
    _write(p, {
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    })
    model = _BufModel()
    load_dit(
        model, str(p), device="cpu", dtype=torch.bfloat16,
        expected_missing=("rope_seq",),
    )
    assert model.plain.weight.dtype == torch.bfloat16
    assert model.rope_seq.dtype == torch.float32  # buffer kept, not from checkpoint


def test_load_dit_expected_missing_mismatch_raises(tmp_path):
    from thenoise.utils.loader import load_dit

    p = tmp_path / "bf16.safetensors"
    _write(p, {"plain.weight": torch.randn(512, 256, dtype=torch.bfloat16)})  # plain.bias missing
    model = _BufModel()
    with pytest.raises(RuntimeError, match="missing"):
        load_dit(
            model, str(p), device="cpu", dtype=torch.bfloat16,
            expected_missing=("rope_seq",),  # does not cover plain.bias
        )


def _bf16_sd():
    return {
        "q.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "plain.bias": torch.randn(512, dtype=torch.bfloat16),
    }


# --------------------------------------------------------------------------- real checkpoint (optional)

@pytest.mark.skipif(
    not os.path.exists("models/anima/split_files/diffusion_models/anima-turbo-v1.0-int8convrot.safetensors"),
    reason="real INT8 Anima checkpoint not present",
)
def test_real_int8_checkpoint_is_detected():
    from thenoise.models import resolve
    assert resolve("models/anima/split_files/diffusion_models/anima-turbo-v1.0-int8convrot.safetensors") is not None
    assert is_quantized_checkpoint("models/anima/split_files/diffusion_models/anima-turbo-v1.0-int8convrot.safetensors") is True
