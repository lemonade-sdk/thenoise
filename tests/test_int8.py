"""Quantized (INT8/FP8) support tests.

These cover the shared ``QuantizedLinear`` module (backed by comfy_kitchen's
``QuantizedTensor``) and the generic quantized loading helpers. They run on CPU
(comfy_kitchen's eager backend) and need no GPU or real checkpoints.
"""
from __future__ import annotations

import pytest
import torch
from comfy_kitchen.tensor import QuantizedTensor

from conftest import (
    TinyDiT,
    bf16_tensors,
    comfy_quant,
    fp8_pair,
    fp8_qt,
    int8_checkpoint,
    int8_lora_state_dict,
    int8_pair,
    int8_qt,
    int8_tensors,
    write_safetensors,
    wrapped_fp8_tensor,
    wrapped_int8_tensor,
)
from thenoise.dit.quantized import QuantizedLinear
from thenoise.utils.loader import (
    is_quantized_checkpoint,
    load_dit,
    load_quantized_state_dict,
)

OUT_F, IN_F = 512, 256


# ------------------------------------------------------------- QuantizedLinear


def _bf16_x(rows=4):
    return torch.randn(rows, IN_F, dtype=torch.bfloat16)


def test_quantized_linear_bf16_forward():
    layer = QuantizedLinear(IN_F, OUT_F, bias=False)
    torch.nn.init.ones_(layer.weight)
    layer = layer.to(torch.bfloat16)  # models are cast to bf16 by the adapter
    out = layer(_bf16_x())
    assert out.shape == (4, OUT_F)
    assert out.dtype == torch.bfloat16
    assert not layer._quantized


@pytest.mark.parametrize(
    "qt,stored_dtype",
    [(wrapped_int8_tensor(), torch.int8), (wrapped_fp8_tensor(), torch.float8_e4m3fn)],
    ids=["int8", "fp8"],
)
def test_quantized_linear_load_quantized_and_forward(qt, stored_dtype):
    layer = QuantizedLinear(IN_F, OUT_F, bias=False)
    assert isinstance(layer.weight, torch.nn.Parameter)

    layer.load_quantized(qt)
    assert layer._quantized
    # The bf16 parameter is dropped (weight is now the quantized buffer) -> the
    # memory saving this whole path exists for.
    assert isinstance(layer.weight, QuantizedTensor)
    assert layer.weight._qdata.dtype == stored_dtype

    out = layer(_bf16_x())
    assert out.shape == (4, OUT_F)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("layout", ["int8", "fp8"])
def test_quantized_linear_bake_lora_requantizes_in_place(layout):
    """The delta is baked into the stored low-bit weight, profile preserved."""
    layer = QuantizedLinear(IN_F, OUT_F, bias=False)
    if layout == "int8":
        qweight, scale = int8_pair()
        layer.load_quantized(int8_qt(qweight, scale))
    else:
        qweight, scale = fp8_pair()
        layer.load_quantized(fp8_qt(qweight, scale))
    stored_dtype = layer.weight._qdata.dtype
    orig_q = layer.weight._qdata.clone()

    x = _bf16_x()
    base = layer(x)

    # LoRA factors: down [r, in], up [out, r]; the baked delta is
    # (up @ down) * (alpha/r * multiplier), shaped [out, in].
    down = torch.randn(8, IN_F, dtype=torch.bfloat16) * 6
    up = torch.randn(OUT_F, 8, dtype=torch.bfloat16) * 6
    delta = (up @ down) * (8.0 / down.size(0) * 2.0)
    layer.bake_lora(delta)

    assert not torch.equal(layer.weight._qdata, orig_q)  # requantized in place
    assert layer.weight._qdata.dtype == stored_dtype  # layout profile preserved

    out = layer(x)
    # The INT8 GEMM also quantizes activations, so an exact match to
    # base + x @ delta^T is impossible; instead verify the LoRA moves the output
    # substantially and lands much closer to the delta expectation than to base.
    err_to_expected = (out.float() - (base + x @ delta.t()).float()).abs().max()
    err_to_base = (out.float() - base.float()).abs().max()
    assert err_to_base > 1.0
    assert err_to_expected < err_to_base


# ------------------------------------------------------- quantized LoRA undo


@pytest.mark.parametrize(
    "kwargs",
    [
        {},                                                  # plain INT8
        {"weight_dtype": torch.float8_e4m3fn},                # FP8 (per-tensor scale)
        {"prefix": "model.diffusion_model."},                 # ComfyUI repackage
    ],
    ids=["int8", "fp8", "int8-wrapped"],
)
def test_apply_lora_bakes_quantized_and_undo_reloads_from_disk(kwargs, tmp_path):
    """Undo reloads the ORIGINAL low-bit weights from the checkpoint by raw key.

    The raw key must survive the wrapper-prefix stripping, otherwise a
    repackaged checkpoint restores nothing (the LoRA would stay baked in).
    """
    from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model

    prefix = kwargs.get("prefix", "")
    path, tensors = int8_checkpoint(tmp_path / "dit.safetensors", **kwargs)
    model = TinyDiT()
    load_dit(model, path, device="cpu", dtype=torch.bfloat16)
    assert model.q._quantized
    orig_q = model.q.weight._qdata.clone()
    orig_scale = model.q.weight.params.scale.clone()
    assert torch.equal(orig_scale, tensors[f"{prefix}q.weight_scale"])

    result = apply_lora_to_model(
        model, [int8_lora_state_dict()], [1.0], torch.device("cpu"), dit_path=path
    )
    assert result["quantized_affected"] == ("q",)
    assert result["quantized_restore_keys"] == (f"{prefix}q.weight",)
    assert result["dit_path"] == path
    assert not torch.equal(model.q.weight._qdata, orig_q)

    undo_lora_on_model(model, result, torch.device("cpu"))
    assert torch.equal(model.q.weight._qdata, orig_q)
    assert torch.equal(model.q.weight.params.scale, orig_scale)


def test_apply_lora_undo_without_dit_path_raises(tmp_path):
    from thenoise.utils.lora import apply_lora_to_model, undo_lora_on_model

    # Restore keys are captured at load time, but undo has no file to read them
    # from -> a hard error rather than a silently half-restored model.
    path, _ = int8_checkpoint(tmp_path / "int8.safetensors")
    model = TinyDiT()
    load_dit(model, path, device="cpu", dtype=torch.bfloat16)

    result = apply_lora_to_model(model, [int8_lora_state_dict()], [1.0], torch.device("cpu"))
    assert result["quantized_restore_keys"] == ("q.weight",)
    with pytest.raises(RuntimeError, match="no dit_path"):
        undo_lora_on_model(model, result, torch.device("cpu"))


def test_apply_lora_mixed_quantized_and_bf16_layers():
    from thenoise.utils.lora import apply_lora_to_model

    model = TinyDiT(quantized=True)
    orig_q = model.q.weight._qdata.clone()
    orig_plain = model.plain.weight.clone()

    loras = int8_lora_state_dict("q") | int8_lora_state_dict("plain")
    result = apply_lora_to_model(model, [loras], [1.0], torch.device("cpu"))

    assert not torch.equal(model.q.weight._qdata, orig_q)  # int8 layer baked
    assert not torch.equal(model.plain.weight, orig_plain)  # bf16 layer mutated
    assert "plain.weight" in result["affected_keys"]
    assert result["quantized_affected"] == ("q",)


# ------------------------------------------------------ is_quantized_checkpoint


@pytest.mark.parametrize(
    "tensors,expected",
    [
        # INT8 and FP8 both carry a ``.weight_scale`` next to the low-bit weight.
        (int8_tensors(), True),
        (int8_tensors(weight_dtype=torch.float8_e4m3fn), True),
        # A wrapped INT8 file is still quantized (the prefix is stripped first).
        (int8_tensors(prefix="model.diffusion_model."), True),
        (bf16_tensors(), False),
        # The check is a header-name heuristic: any ``.weight_scale`` is enough to
        # take the quantized load path (which then fails loudly if no low-bit
        # weight owns it, rather than silently loading garbage).
        ({"blocks.0.attn.q_proj.weight_scale": torch.zeros(16, 1)}, True),
    ],
    ids=["int8", "fp8", "int8-wrapped", "bf16", "scale-only"],
)
def test_is_quantized_checkpoint(tmp_path, tensors, expected):
    path = write_safetensors(tmp_path / "ckpt.safetensors", tensors)
    assert is_quantized_checkpoint(path) is expected


# ------------------------------------------------------------ comfy_quant marker


@pytest.mark.parametrize(
    "marker,convrot,groupsize",
    [
        # A layer quantized with convrot_groupsize=64 must rotate activations
        # with 64 at inference, NOT the default 256, or the images are garbage.
        (comfy_quant(convrot=True, groupsize=64), True, 64),
        # A layer whose in_features were not divisible by the group size is NOT
        # ConvRot-rotated: inference must not rotate (default is convrot=true).
        (comfy_quant(convrot=False), False, 256),
        # No marker at all keeps the default profile...
        (None, True, 256),
        # ...and so does an unparseable one.
        (torch.zeros(8, dtype=torch.uint8), True, 256),
    ],
    ids=["groupsize-64", "convrot-off", "no-marker", "unparseable-marker"],
)
def test_comfy_quant_marker_drives_the_inference_profile(tmp_path, marker, convrot, groupsize):
    path, _ = int8_checkpoint(tmp_path / "int8.safetensors", marker=marker)
    model = TinyDiT()
    load_dit(model, path, device="cpu", dtype=torch.bfloat16)

    assert model.q._quantized is True
    assert model.q.weight.params.convrot is convrot
    assert model.q.weight.params.convrot_groupsize == groupsize


# --------------------------------------------------- load_quantized_state_dict


def test_load_quantized_state_dict_mixed():
    model = TinyDiT()
    load_quantized_state_dict(model, int8_tensors())

    # The quantized layer switched to INT8...
    assert model.q._quantized is True
    assert isinstance(model.q.weight, QuantizedTensor)
    assert model.q.weight._qdata.dtype == torch.int8
    assert model.q.weight.params.scale.dtype == torch.float32
    # ...and the full-precision layer was assigned normally.
    assert model.plain.weight.dtype == torch.bfloat16
    assert model.plain.bias.dtype == torch.bfloat16

    # forward runs end-to-end (bf16 in -> bf16 out)
    x = _bf16_x()
    assert model.q(x).shape == (4, OUT_F)
    assert model.q(x).dtype == torch.bfloat16


def test_load_quantized_state_dict_fp8_uses_a_per_tensor_scale():
    model = TinyDiT()
    sd = int8_tensors(weight_dtype=torch.float8_e4m3fn, scale=torch.tensor(0.5))
    load_quantized_state_dict(model, sd)

    assert model.q._quantized is True
    assert model.q.weight._qdata.dtype == torch.float8_e4m3fn
    assert model.q.weight.params.scale.shape == ()  # per-tensor, not per-row
    assert model.q(_bf16_x()).dtype == torch.bfloat16


def test_load_quantized_state_dict_missing_scale_raises():
    sd = int8_tensors()
    del sd["q.weight_scale"]
    with pytest.raises(RuntimeError, match="missing its .weight_scale"):
        load_quantized_state_dict(TinyDiT(), sd)


def test_load_quantized_state_dict_orphan_scale_raises():
    sd = int8_tensors()
    sd["plain.weight_scale"] = torch.zeros(OUT_F, 1, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="orphan"):
        load_quantized_state_dict(TinyDiT(), sd)


# --------------------------------------------------------------- load_dit


def test_load_dit_bf16(tmp_path):
    path = write_safetensors(tmp_path / "bf16.safetensors", bf16_tensors())
    model = load_dit(TinyDiT(), path, device="cpu", dtype=torch.bfloat16)
    assert model.q._quantized is False
    assert model.q.weight.dtype == torch.bfloat16
    assert model.plain.weight.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "kwargs", [{}, {"prefix": "model.diffusion_model."}], ids=["raw-keys", "wrapped-keys"]
)
def test_load_dit_int8_and_fp8(tmp_path, kwargs):
    """Both quantized formats load, and the wrapper prefix is stripped."""
    for weight_dtype in (torch.int8, torch.float8_e4m3fn):
        path, _ = int8_checkpoint(
            tmp_path / f"dit-{weight_dtype}.safetensors", weight_dtype=weight_dtype, **kwargs
        )
        model = load_dit(TinyDiT(), path, device="cpu", dtype=torch.bfloat16)
        assert model.q._quantized is True
        assert model.q.weight._qdata.dtype == weight_dtype
        assert model.plain.weight.dtype == torch.bfloat16
        assert model.q(_bf16_x()).shape == (4, OUT_F)


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "int8"])
def test_load_dit_drop_keys_applies_on_both_paths(tmp_path, quantized):
    """Unexpected leftover keys (e.g. Krea2's unused ``last.*``) are dropped."""
    extra = {
        "last.down.residual": torch.randn(16, dtype=torch.bfloat16),
        "last.up.residual": torch.randn(16, dtype=torch.bfloat16),
    }
    tensors = (
        int8_tensors(extra=extra) if quantized else bf16_tensors(extra=extra)
    )
    path = write_safetensors(tmp_path / "dit.safetensors", tensors)

    model = TinyDiT()
    # Without drop_keys the strict load would fail on the unexpected keys.
    load_dit(model, path, device="cpu", dtype=torch.bfloat16, drop_keys=("last.down", "last.up"))
    assert model.q._quantized is quantized
    assert model.plain.weight.dtype == torch.bfloat16


class _ScaleNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(dim))


class _ScaleModel(torch.nn.Module):
    """A model whose norm parameter is called ``scale``, not ``weight``."""

    def __init__(self):
        super().__init__()
        self.q = QuantizedLinear(IN_F, OUT_F)
        self.norm = _ScaleNorm(OUT_F)


def test_load_dit_key_map(tmp_path):
    """ComfyUI's INT8 exporter stores norm ``scale`` params under ``weight``."""
    path, _ = int8_checkpoint(
        tmp_path / "int8.safetensors",
        extra={"norm.weight": torch.randn(OUT_F, dtype=torch.bfloat16)},
        drop=["plain.weight", "plain.bias"],
    )
    key_map = lambda k: k[: -len(".weight")] + ".scale" if k.endswith("norm.weight") else k
    model = load_dit(_ScaleModel(), path, device="cpu", dtype=torch.bfloat16, key_map=key_map)

    assert model.q._quantized is True
    assert model.norm.scale.dtype == torch.bfloat16


class _WeightNorm(torch.nn.Module):
    """A norm whose parameter is called ``weight`` (the shared RMSNorm layout)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))


class _WeightModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = _WeightNorm(OUT_F)


def test_load_dit_value_map_on_bf16(tmp_path):
    """value_map renames a zero-centered ``scale`` to ``weight`` and shifts by one.

    Mirrors Krea2's reconciliation: the checkpoint stores ``scale`` (effective
    ``weight = scale + 1``); the loader must rename and shift so the runtime
    ``weight`` param is correct. Runs on the BF16 path.
    """
    path = write_safetensors(
        tmp_path / "bf16.safetensors",
        {"norm.scale": torch.full((OUT_F,), 2.0, dtype=torch.bfloat16)},
    )

    def value_map(key, tensor):
        if key.endswith(".scale"):
            return key[: -len(".scale")] + ".weight", tensor + 1.0
        return key, tensor

    model = load_dit(_WeightModel(), path, device="cpu", dtype=torch.bfloat16, value_map=value_map)

    assert torch.equal(model.norm.weight, torch.full((OUT_F,), 3.0, dtype=torch.bfloat16))


class _BufModel(torch.nn.Module):
    """Model with an internal buffer that is deliberately not in the checkpoint."""

    def __init__(self):
        super().__init__()
        self.plain = torch.nn.Linear(IN_F, OUT_F, bias=True)
        self.register_buffer("rope_seq", torch.zeros(128))


def test_load_dit_expected_missing_keeps_the_buffer(tmp_path):
    path = write_safetensors(
        tmp_path / "bf16.safetensors",
        {
            "plain.weight": torch.randn(OUT_F, IN_F, dtype=torch.bfloat16),
            "plain.bias": torch.randn(OUT_F, dtype=torch.bfloat16),
        },
    )
    model = load_dit(
        _BufModel(), path, device="cpu", dtype=torch.bfloat16, expected_missing=("rope_seq",)
    )
    assert model.plain.weight.dtype == torch.bfloat16
    assert model.rope_seq.dtype == torch.float32  # kept, not taken from the checkpoint


def test_load_dit_unexpected_missing_raises(tmp_path):
    path = write_safetensors(
        tmp_path / "bf16.safetensors",
        {"plain.weight": torch.randn(OUT_F, IN_F, dtype=torch.bfloat16)},  # plain.bias missing
    )
    with pytest.raises(RuntimeError, match="missing"):
        load_dit(
            _BufModel(), path, device="cpu", dtype=torch.bfloat16,
            expected_missing=("rope_seq",),  # does not cover plain.bias
        )
