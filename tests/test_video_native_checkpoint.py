"""Comfy-Org TI2V-5B transformer checkpoint tests (small tensors)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from thenoise.dit.wan_video.checkpoint import (
    _TI2V_CONFIG, _WanWeightModel, load_wan_native_transformer,
)
from thenoise.dit.wan_video.model import WanNativeDiT
from thenoise.utils.wan_weights import wan_dit_key


TINY = dict(num_attention_heads=2, attention_head_dim=12, num_layers=2,
            ffn_dim=64, freq_dim=32, text_dim=32, rope_max_seq_len=64)


def _raw_key(key: str) -> str:
    roots = {
        "condition_embedder.text_embedder.linear_1.": "text_embedding.0.",
        "condition_embedder.text_embedder.linear_2.": "text_embedding.2.",
        "condition_embedder.time_embedder.linear_1.": "time_embedding.0.",
        "condition_embedder.time_embedder.linear_2.": "time_embedding.2.",
        "condition_embedder.time_proj.": "time_projection.1.",
        "proj_out.": "head.head.",
        "scale_shift_table": "head.modulation",
    }
    for dst, src in roots.items():
        if key.startswith(dst):
            return src + key[len(dst):]
    if not key.startswith("blocks."):
        return key
    parts = key.split(".", 2)
    prefix, suffix = ".".join(parts[:2]) + ".", parts[2]
    if suffix == "scale_shift_table":
        return prefix + "modulation"
    if suffix.startswith("norm2."):
        return prefix + "norm3." + suffix[6:]
    for dst, src in (
        ("attn1.to_q.", "self_attn.q."), ("attn1.to_k.", "self_attn.k."),
        ("attn1.to_v.", "self_attn.v."), ("attn1.to_out.0.", "self_attn.o."),
        ("attn1.norm_q.", "self_attn.norm_q."), ("attn1.norm_k.", "self_attn.norm_k."),
        ("attn2.to_q.", "cross_attn.q."), ("attn2.to_k.", "cross_attn.k."),
        ("attn2.to_v.", "cross_attn.v."), ("attn2.to_out.0.", "cross_attn.o."),
        ("attn2.norm_q.", "cross_attn.norm_q."), ("attn2.norm_k.", "cross_attn.norm_k."),
        ("ffn.net.0.proj.", "ffn.0."), ("ffn.net.2.", "ffn.2."),
    ):
        if suffix.startswith(dst):
            return prefix + src + suffix[len(dst):]
    return key


def _write(root, *, missing=None, wrong_shape=None):
    checkpoint = root / "split_files" / "diffusion_models" / "wan2.2_ti2v_5B_fp16.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(**(_TI2V_CONFIG | TINY))
    with torch.random.fork_rng():
        torch.manual_seed(123)
        source = _WanWeightModel(cfg)
    with torch.no_grad():
        for param in source.parameters():
            param.uniform_(-0.1, 0.1)
    state = {}
    for name, tensor in source.state_dict().items():
        raw = _raw_key(name)
        assert wan_dit_key(raw) == name, (name, raw, wan_dit_key(raw))
        state[raw] = tensor.detach().half().contiguous()
    if missing:
        state.pop(missing)
    if wrong_shape:
        state[wrong_shape] = torch.zeros(2, 2, dtype=torch.float16)
    save_file(state, str(checkpoint))
    return state


def test_comfy_checkpoint_dtype_and_forward(tmp_path):
    raw = _write(tmp_path)
    model = load_wan_native_transformer(tmp_path, dtype=torch.float32, _test_config=TINY)
    assert isinstance(model, WanNativeDiT)
    assert not any(p.is_meta for p in model.parameters())
    assert model.patch_embedding.weight.dtype == torch.float32
    assert torch.equal(model.proj_out.weight, raw["head.head.weight"].float())
    x = torch.randn(1, 48, 3, 4, 6)
    ctx = torch.randn(1, 5, 32)
    with torch.no_grad():
        a = model(x, torch.tensor([500.0]), ctx, return_dict=False)[0]
        b = model(x, torch.full((1, 18), 500.0), ctx, return_dict=False)[0]
    assert a.shape == b.shape == x.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_bf16_comfy_precision_exceptions(tmp_path):
    _write(tmp_path)
    model = load_wan_native_transformer(tmp_path, dtype=torch.bfloat16, _test_config=TINY)
    assert model.proj_out.weight.dtype == torch.bfloat16
    assert model.patch_embedding.weight.dtype == torch.float32
    assert model.condition_embedder.time_linear_1.weight.dtype == torch.float32
    assert model.blocks[0].scale_shift_table.dtype == torch.float32
    x = torch.randn(1, 48, 1, 4, 4, dtype=torch.bfloat16)
    context = torch.randn(1, 4, TINY["text_dim"], dtype=torch.bfloat16)
    with torch.no_grad():
        output = model(x, torch.tensor([500.0]), context).sample
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_fp16_compute_dtype_preserves_fp32_patch_embedding(tmp_path):
    _write(tmp_path)
    model = load_wan_native_transformer(tmp_path, dtype=torch.float16, _test_config=TINY)
    assert model.patch_embedding.weight.dtype == torch.float32
    assert model.condition_embedder.time_linear_1.weight.dtype == torch.float32
    assert model.condition_embedder.time_proj.weight.dtype == torch.float16
    assert model.dtype == torch.float16

    x = torch.randn(1, 48, 1, 4, 4, dtype=torch.float32).to(model.dtype)
    context = torch.randn(1, 4, TINY["text_dim"], dtype=torch.float32).to(model.dtype)
    with torch.no_grad():
        output = model(x, torch.full((1, 4), 500.0), context).sample
    assert output.dtype == torch.float16
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_checkpoint_rejects_missing_and_wrong_shapes(tmp_path):
    _write(tmp_path, missing="patch_embedding.weight")
    with pytest.raises(ValueError, match="mismatch"):
        load_wan_native_transformer(tmp_path, _test_config=TINY)
    _write(tmp_path, wrong_shape="patch_embedding.weight")
    with pytest.raises(ValueError, match="shape mismatch"):
        load_wan_native_transformer(tmp_path, _test_config=TINY)


def test_official_5b_topology_is_constructible_on_meta():
    with torch.device("meta"):
        model = _WanWeightModel(SimpleNamespace(**_TI2V_CONFIG))
    assert model.patch_embedding.weight.shape == (3072, 48, 1, 2, 2)
    assert len(model.blocks) == 30
    assert model.blocks[0].ffn.net[0].proj.weight.shape == (14336, 3072)
    assert all(p.is_meta for p in model.parameters())
