"""Quantized text-encoder loading tests.

Cover the unified ``load_text_encoder_weights`` path (auto-select quantized vs
BF16) plus the ``replace_linears`` structural swap. They use a tiny fake Qwen3-like
model (``nn.Linear`` stacks under a ``.model`` submodule) so no transformers /
real checkpoints are needed. They run on CPU (comfy_kitchen's eager backend).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from thenoise.dit.quantized import replace_linears
from thenoise.utils.loader import load_text_encoder_weights


def _comfy_quant(convrot=True, groupsize=256):
    """Build a ``comfy_quant`` marker tensor like ComfyUI's INT8 exporter."""
    import json

    conf = {"convrot": bool(convrot), "per_row": True}
    if convrot:
        conf["convrot_groupsize"] = int(groupsize)
    data = json.dumps(conf).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64, bias=False)
        self.k_proj = nn.Linear(64, 64, bias=False)


class _FakeQwen(nn.Module):
    """A Qwen3-like full model: a ``.model`` submodule with linears + a buffer."""

    def __init__(self):
        super().__init__()
        self.model = self._make_model()

    @staticmethod
    def _make_model():
        model = nn.Module()
        model.embed_tokens = nn.Embedding(128, 64)
        model.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])
        model.norm = nn.LayerNorm(64)
        model.rotary_emb = nn.Module()
        model.rotary_emb.register_buffer("inv_freq", torch.empty(32))
        return model


def _build_model():
    model = _FakeQwen()
    replace_linears(model)
    return model


def _write(path, tensors):
    from safetensors.torch import save_file

    save_file(tensors, str(path))


def _complete_bf16_state_dict(model):
    """Return a full bf16 state dict so the strict BF16 load passes."""
    return {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- replace_linears


def test_replace_linears_swaps_nn_linear_instances():
    from thenoise.dit.quantized import QuantizedLinear

    model = _FakeQwen()
    # Before the swap the linears are plain nn.Linear.
    assert isinstance(model.model.layers[0].q_proj, nn.Linear)
    assert not isinstance(model.model.layers[0].q_proj, QuantizedLinear)

    replace_linears(model)
    assert isinstance(model.model.layers[0].q_proj, QuantizedLinear)
    assert model.model.layers[0].q_proj.in_features == 64
    assert model.model.layers[0].q_proj.bias is None
    # Non-linear leaves (embeddings / norms / buffers) are left alone.
    assert isinstance(model.model.embed_tokens, nn.Embedding)
    assert isinstance(model.model.norm, nn.LayerNorm)


def test_replace_linears_preserves_bias():
    class _HasBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 16, bias=True)

    m = _HasBias()
    replace_linears(m)
    from thenoise.dit.quantized import QuantizedLinear

    assert isinstance(m.proj, QuantizedLinear)
    assert m.proj.bias is not None


# --------------------------------------------------------------------------- load_text_encoder_weights (quantized)


def test_load_text_encoder_weights_quantized_int8(tmp_path):
    p = tmp_path / "te_int8.safetensors"
    _write(p, {
        "model.layers.0.q_proj.weight": torch.randint(-127, 127, (64, 64), dtype=torch.int8),
        "model.layers.0.q_proj.weight_scale": torch.rand(64, 1),
        "model.layers.0.q_proj.comfy_quant": _comfy_quant(convrot=True, groupsize=64),
        "model.embed_tokens.weight": torch.randn(128, 64, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(64, dtype=torch.bfloat16),
        "model.norm.bias": torch.randn(64, dtype=torch.bfloat16),
        "model.rotary_emb.inv_freq": torch.randn(32, dtype=torch.bfloat16),
    })
    model = _build_model()
    load_text_encoder_weights(model, str(p), device="cpu", dtype=torch.bfloat16)
    # The int8 q_proj switched to its quantized layout, carrying the marker group size.
    assert model.model.layers[0].q_proj._quantized is True
    assert model.model.layers[0].q_proj.weight.params.convrot is True
    assert model.model.layers[0].q_proj.weight.params.convrot_groupsize == 64
    # Un-quantized leaves are materialized as bf16 params / buffers.
    assert model.model.norm.weight.dtype == torch.bfloat16
    assert model.model.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert model.model.rotary_emb.inv_freq.shape == (32,)


def test_load_text_encoder_weights_quantized_fp8(tmp_path):
    p = tmp_path / "te_fp8.safetensors"
    _write(p, {
        "model.layers.0.k_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
        "model.layers.0.k_proj.weight_scale": torch.tensor(0.5, dtype=torch.float32),
        "model.layers.0.k_proj.comfy_quant": torch.zeros(8, dtype=torch.uint8),
        "model.embed_tokens.weight": torch.randn(128, 64, dtype=torch.bfloat16),
    })
    model = _build_model()
    load_text_encoder_weights(model, str(p), device="cpu", dtype=torch.bfloat16)
    assert model.model.layers[0].k_proj._quantized is True


# --------------------------------------------------------------------------- load_text_encoder_weights (BF16)


def test_load_text_encoder_weights_bf16(tmp_path):
    model = _build_model()
    sd = _complete_bf16_state_dict(model)
    p = tmp_path / "te_bf16.safetensors"
    _write(p, sd)

    load_text_encoder_weights(model, str(p), device="cpu", dtype=torch.bfloat16)
    # BF16 checkpoint keeps the QuantizedLinear in its plain (un-quantized) mode.
    assert model.model.layers[0].q_proj._quantized is False
    assert model.model.layers[0].q_proj.weight.dtype == torch.bfloat16
    assert model.model.layers[0].q_proj.weight.shape == (64, 64)


def test_load_text_encoder_weights_drops_tied_lm_head(tmp_path):
    # The tied ``lm_head.weight`` (absent from quantized files, redundant in BF16
    # files) must be dropped so it doesn't trip the strict load. Encoders ``del``
    # their lm_head module (we only consume .model), mirroring that here.
    class _WithLmHead(_FakeQwen):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(64, 128, bias=False)

    model = _WithLmHead()
    replace_linears(model)
    sd = _complete_bf16_state_dict(model)
    # The file carries the redundant tied lm_head.weight; the loader drops it.
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
    del model.lm_head  # unused (we consume .model only)

    p = tmp_path / "te_lmhead.safetensors"
    _write(p, sd)

    load_text_encoder_weights(model, str(p), device="cpu", dtype=torch.bfloat16)
    assert model.model.layers[0].q_proj.weight.shape == (64, 64)
    assert not hasattr(model, "lm_head")


def test_load_text_encoder_weights_model_prefix_key_map(tmp_path):
    # The bare-model layout (like Anima) needs the ``model.`` prefix stripped, passed
    # as a key_map applied on both paths.
    class _Bare(_FakeQwen):
        def __init__(self):
            super().__init__()
            self.layers = self._make_model().layers
            self.embed_tokens = nn.Embedding(128, 64)
            self.norm = nn.LayerNorm(64)

    p = tmp_path / "te_prefix.safetensors"
    _write(p, {
        "model.layers.0.q_proj.weight": torch.randint(-127, 127, (64, 64), dtype=torch.int8),
        "model.layers.0.q_proj.weight_scale": torch.rand(64, 1),
        "model.layers.0.q_proj.comfy_quant": _comfy_quant(),
        "model.embed_tokens.weight": torch.randn(128, 64, dtype=torch.bfloat16),
    })
    model = _Bare()
    replace_linears(model)
    load_text_encoder_weights(
        model,
        str(p),
        device="cpu",
        dtype=torch.bfloat16,
        key_map=lambda k: k[len("model."):] if k.startswith("model.") else k,
    )
    assert model.layers[0].q_proj._quantized is True
    assert model.embed_tokens.weight.shape == (128, 64)
