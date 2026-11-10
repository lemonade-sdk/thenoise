"""CPU-only checks of independent native sharded Wan transformer loading."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from thenoise.video.native_checkpoint import (
    _WanWeightModel, _config, _rope_tables, load_wan_native_transformer,
)
from thenoise.video.native_dit import WanNativeDiT


def _tiny_config():
    return {
        "_class_name": "WanTransformer3DModel",
        "in_channels": 48, "out_channels": 48,
        "num_attention_heads": 2, "attention_head_dim": 12,
        "num_layers": 2, "ffn_dim": 64, "freq_dim": 32,
        "text_dim": 32, "patch_size": [1, 2, 2],
        "cross_attn_norm": True, "qk_norm": "rms_norm_across_heads",
        "image_dim": None, "added_kv_proj_dim": None,
        "rope_max_seq_len": 64, "eps": 1e-6,
    }


def _write_checkpoint(directory, *, shards=True):
    transformer = directory / "transformer"
    transformer.mkdir()
    cfg = _tiny_config()
    (transformer / "config.json").write_text(json.dumps(cfg))
    with torch.random.fork_rng():
        torch.manual_seed(123)
        source = _WanWeightModel(SimpleNamespace(**{**cfg, "patch_size": (1, 2, 2)})).eval()
    _rope_tables(source.rope, cfg["rope_max_seq_len"])
    # _WanWeightModel is a metadata skeleton; in production every parameter is
    # replaced by checkpoint tensors. Initialize the synthetic fixture only.
    with torch.no_grad():
        for p in source.parameters():
            p.uniform_(-0.05, 0.05)
    # In a real checkpoint the shard index splits layers across file boundaries.
    state = {k: v.detach().clone().contiguous() for k, v in source.state_dict().items()}
    state["blocks.0.scale_shift_table"][0, 0, 0] = 1.0012345
    state["condition_embedder.time_embedder.linear_1.weight"][0, 0] = 1.0012345
    source.load_state_dict(state)
    if shards:
        names = ["diffusion_pytorch_model-00001-of-00002.safetensors",
                 "diffusion_pytorch_model-00002-of-00002.safetensors"]
        groups = [dict(), dict()]
        for i, (key, value) in enumerate(sorted(state.items())):
            groups[i % 2][key] = value
        for filename, group in zip(names, groups):
            save_file(group, transformer / filename)
        weight_map = {key: filename for filename, group in zip(names, groups) for key in group}
        (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
        )
    else:
        save_file(state, transformer / "diffusion_pytorch_model.safetensors")
    return source, state


@pytest.mark.parametrize("shards", [True, False])
def test_independent_checkpoint_loader_preserves_shapes_precision_and_keys(tmp_path, shards):
    source, state = _write_checkpoint(tmp_path, shards=shards)
    model = load_wan_native_transformer(tmp_path, dtype=torch.bfloat16, _test_size=True)
    assert isinstance(model, WanNativeDiT)
    assert model.patch_embedding.weight.dtype == torch.bfloat16
    assert model.condition_embedder.text_linear_1.weight.dtype == torch.bfloat16
    assert model.condition_embedder.time_linear_1.weight.dtype == torch.float32
    assert model.blocks[0].norm2.weight.dtype == torch.float32
    assert model.blocks[0].scale_shift_table.dtype == torch.float32
    assert model.scale_shift_table.dtype == torch.float32
    assert model.blocks[0].scale_shift_table[0, 0, 0].item() == pytest.approx(1.0012345, abs=1e-7)
    assert model.condition_embedder.time_linear_1.weight[0, 0].item() == pytest.approx(1.0012345, abs=1e-7)
    assert torch.equal(model.patch_embedding.weight, state["patch_embedding.weight"].to(torch.bfloat16))
    assert not any(p.is_meta for p in model.parameters())
    assert model.rope.freqs_cos.dtype == torch.float32
    assert torch.equal(model.rope.freqs_cos, source.rope.freqs_cos)


@pytest.mark.parametrize("i2v", [False, True])
def test_loaded_transformer_forward_shape(tmp_path, i2v):
    _write_checkpoint(tmp_path)
    model = load_wan_native_transformer(tmp_path, dtype=torch.float32, _test_size=True)
    latents = torch.randn(1, 48, 3, 4, 6)
    context = torch.randn(1, 5, 32)
    timestep = torch.full((1, 18), 500.0) if i2v else torch.tensor([500.0])
    if i2v:
        timestep[:, :6] = 0
    with torch.no_grad():
        out = model(latents, timestep, context, return_dict=False)[0]
    assert out.shape == latents.shape
    assert torch.isfinite(out).all()


def test_rejects_wrong_variant_without_loading_tensors(tmp_path):
    _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="config mismatches"):
        load_wan_native_transformer(tmp_path)
    cfg_path = tmp_path / "transformer" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["in_channels"] = 16
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="unsupported Wan"):
        load_wan_native_transformer(tmp_path, _test_size=True)


def test_rejects_missing_or_unexpected_key_before_loading(tmp_path):
    _write_checkpoint(tmp_path)
    idx = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    mapping = json.loads(idx.read_text())
    mapping["weight_map"].pop("patch_embedding.weight")
    idx.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="key mismatch"):
        load_wan_native_transformer(tmp_path, _test_size=True)


def test_rejects_missing_shard_and_path_traversal(tmp_path):
    _write_checkpoint(tmp_path)
    idx = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    mapping = json.loads(idx.read_text())
    entry = mapping["weight_map"]["patch_embedding.weight"]
    (tmp_path / "transformer" / entry).unlink()
    with pytest.raises(FileNotFoundError, match="missing Wan transformer shard"):
        load_wan_native_transformer(tmp_path, _test_size=True)
    mapping["weight_map"]["patch_embedding.weight"] = "../../unsafe.safetensors"
    idx.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="index entry"):
        load_wan_native_transformer(tmp_path, _test_size=True)


def test_rejects_wrong_shard_header(tmp_path):
    _write_checkpoint(tmp_path)
    idx = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    mapping = json.loads(idx.read_text())
    a, b = sorted(set(mapping["weight_map"].values()))
    first_key = next(k for k, v in mapping["weight_map"].items() if v == a)
    mapping["weight_map"][first_key] = b
    idx.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="keys differ"):
        load_wan_native_transformer(tmp_path, _test_size=True)


def test_official_ti2v_5b_config_meta_topology(tmp_path):
    from thenoise.video.native_checkpoint import _TI2V_CONFIG
    root = tmp_path / "transformer"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "_class_name": "WanTransformer3DModel", **_TI2V_CONFIG,
        "patch_size": [1, 2, 2], "rope_max_seq_len": 1024,
        "eps": 1e-6,
    }))
    config = _config(tmp_path, test_size=False)
    assert config.attention_head_dim == 128
    with torch.device("meta"):
        skeleton = _WanWeightModel(config)
    sd = skeleton.state_dict()
    assert all(t.is_meta for t in sd.values())


def test_rejects_corrupt_tensor_shape_before_assignment(tmp_path):
    _write_checkpoint(tmp_path, shards=False)
    path = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"
    from safetensors.torch import load_file
    state = load_file(path)
    state["patch_embedding.weight"] = torch.zeros(2, 2)
    save_file(state, path)
    with pytest.raises(ValueError, match="tensor shape mismatch"):
        load_wan_native_transformer(tmp_path, _test_size=True)


def test_cancel_between_native_weight_shards(tmp_path):
    _write_checkpoint(tmp_path, shards=True)
    calls = []

    def stop_after_first():
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        load_wan_native_transformer(tmp_path, _test_size=True, check_cancel=stop_after_first)
    assert len(calls) == 2
