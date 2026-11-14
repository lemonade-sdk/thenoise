"""Comfy-Org Wan transformer checkpoint tests (small tensors)."""
from __future__ import annotations

import re

import pytest
import torch
from safetensors.torch import save_file

from thenoise.dit.wan_video.checkpoint import load_wan_native_transformer
from thenoise.dit.wan_video.model import Wan22Params, WanNativeDiT, WanNativeRoPE, detect_params
from thenoise.utils.wan_weights import wan_dit_key

#: Small enough to build in a test, and still legal geometry for `detect_params`:
#: one 128-wide head, two blocks.
TINY = Wan22Params(in_channels=48, out_channels=48, dim=128, num_layers=2,
                   ffn_dim=256, freq_dim=32, text_dim=32, patch_size=(1, 2, 2))


def _raw_key(key: str) -> str:
    """The inverse of `wan_dit_key`, to write a checkpoint the way Comfy-Org does."""
    roots = {
        "condition_embedder.text_linear_1.": "text_embedding.0.",
        "condition_embedder.text_linear_2.": "text_embedding.2.",
        "condition_embedder.time_linear_1.": "time_embedding.0.",
        "condition_embedder.time_linear_2.": "time_embedding.2.",
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
    # ComfyUI norm2 is the FFN norm, norm3 the cross-attention one.
    if suffix.startswith("norm2."):
        return prefix + "norm3." + suffix[6:]
    for dst, src in (
        ("attn1.to_q.", "self_attn.q."), ("attn1.to_k.", "self_attn.k."),
        ("attn1.to_v.", "self_attn.v."), ("attn1.to_out.", "self_attn.o."),
        ("attn1.norm_q.", "self_attn.norm_q."), ("attn1.norm_k.", "self_attn.norm_k."),
        ("attn2.to_q.", "cross_attn.q."), ("attn2.to_k.", "cross_attn.k."),
        ("attn2.to_v.", "cross_attn.v."), ("attn2.to_out.", "cross_attn.o."),
        ("attn2.norm_q.", "cross_attn.norm_q."), ("attn2.norm_k.", "cross_attn.norm_k."),
        ("ffn_in.", "ffn.0."), ("ffn_out.", "ffn.2."),
    ):
        if suffix.startswith(dst):
            return prefix + src + suffix[len(dst):]
    return key


_MATMUL_WEIGHTS = re.compile(r"^blocks\.\d+\.(?:self|cross)_attn\.[qkvo]\.weight$"
                             r"|^blocks\.\d+\.ffn\.[02]\.weight$")


def _write(root, *, name="wan2.2_ti2v_5B_fp16.safetensors", missing=None, wrong_shape=None,
           mixed=False):
    checkpoint = root / "split_files" / "diffusion_models" / name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng():
        torch.manual_seed(123)
        source = WanNativeDiT(TINY)
    with torch.no_grad():
        for param in source.parameters():
            param.uniform_(-0.1, 0.1)
    state = {}
    for module_name, tensor in source.state_dict().items():
        raw = _raw_key(module_name)
        assert wan_dit_key(raw) == module_name, (module_name, raw, wan_dit_key(raw))
        precision = torch.float16
        if mixed:
            precision = torch.float16 if _MATMUL_WEIGHTS.match(raw) else torch.float32
        state[raw] = tensor.detach().to(precision).contiguous()
    if missing:
        state.pop(missing)
    if wrong_shape:
        state[wrong_shape] = torch.zeros(2, 2, dtype=torch.float16)
    save_file(state, str(checkpoint))
    return state, checkpoint


def test_params_are_read_from_the_header(tmp_path):
    _, path = _write(tmp_path)
    assert detect_params(str(path)) == TINY


def test_a_wrong_checkpoint_is_named_rather_than_guessed(tmp_path):
    state, path = _write(tmp_path)
    state["patch_embedding.weight"] = state["patch_embedding.weight"][:, :3].contiguous()
    save_file(state, str(path))
    with pytest.raises(ValueError, match="no Wan variant"):
        detect_params(str(path))


def test_a_file_that_is_not_a_wan_transformer_is_refused(tmp_path):
    save_file({"layers.0.weight": torch.zeros(2, 2)}, str(tmp_path / "other.safetensors"))
    with pytest.raises(ValueError, match="patch_embedding"):
        detect_params(str(tmp_path / "other.safetensors"))


def test_wan_rope_uses_shared_builder_and_caches_by_geometry():
    from thenoise.utils.rope import RopeCache

    rope = WanNativeRoPE(TINY.head_dim, TINY.patch_size)
    assert isinstance(rope._frequencies, RopeCache)

    input_video = torch.zeros(1, 48, 3, 4, 6)
    first = rope(input_video)
    assert rope(input_video) is first
    assert first.shape == (1, 18, TINY.head_dim // 2, 2, 2)

    head_dim = TINY.head_dim
    hw = 2 * (head_dim // 6)
    cos_axes, sin_axes = [], []
    for part in (head_dim - 2 * hw, hw, hw):
        inv_freq = 1 / 10000 ** (torch.arange(0, part, 2, dtype=torch.float64) / part)
        angles = torch.outer(torch.arange(64, dtype=torch.float64), inv_freq)
        cos_axes.append(angles.cos().float())
        sin_axes.append(angles.sin().float())
    cos = torch.cat([cos_axes[0][:3, None, None].expand(3, 2, 3, -1),
                     cos_axes[1][None, :2, None].expand(3, 2, 3, -1),
                     cos_axes[2][None, None, :3].expand(3, 2, 3, -1)], dim=-1)
    sin = torch.cat([sin_axes[0][:3, None, None].expand(3, 2, 3, -1),
                     sin_axes[1][None, :2, None].expand(3, 2, 3, -1),
                     sin_axes[2][None, None, :3].expand(3, 2, 3, -1)], dim=-1)
    expected = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(first.shape)
    torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-5)

    rope.to("cpu")
    second = rope(input_video)
    assert second is not first
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert rope(torch.zeros(1, 48, 1, 4, 4)).shape == (1, 4, TINY.head_dim // 2, 2, 2)


def test_comfy_checkpoint_dtype_and_forward(tmp_path):
    raw, path = _write(tmp_path)
    model = load_wan_native_transformer(str(path), "cpu", dtype=torch.float32)
    assert isinstance(model, WanNativeDiT)
    assert not any(p.is_meta for p in model.parameters())
    assert model.patch_embedding.weight.dtype == torch.float32
    assert torch.equal(model.proj_out.weight, raw["head.head.weight"].float())
    x = torch.randn(1, 48, 3, 4, 6)
    ctx = torch.randn(1, 5, TINY.text_dim)
    with torch.no_grad():
        a = model(x, torch.tensor([500.0]), ctx, return_dict=False)[0]
        b = model(x, torch.full((1, 18), 500.0), ctx, return_dict=False)[0]
    assert a.shape == b.shape == x.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    # One timestep per token, all equal, conditions the model exactly like one per batch.
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_bf16_comfy_precision_exceptions(tmp_path):
    _, path = _write(tmp_path)
    model = load_wan_native_transformer(str(path), "cpu", dtype=torch.bfloat16)
    assert model.proj_out.weight.dtype == torch.bfloat16
    assert model.patch_embedding.weight.dtype == torch.float32
    assert model.condition_embedder.time_linear_1.weight.dtype == torch.float32
    assert model.blocks[0].scale_shift_table.dtype == torch.float32
    x = torch.randn(1, 48, 1, 4, 4, dtype=torch.bfloat16)
    context = torch.randn(1, 4, TINY.text_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model(x, torch.tensor([500.0]), context).sample
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_fp16_compute_dtype_preserves_fp32_patch_embedding(tmp_path):
    _, path = _write(tmp_path)
    model = load_wan_native_transformer(str(path), "cpu", dtype=torch.float16)
    assert model.patch_embedding.weight.dtype == torch.float32
    assert model.condition_embedder.time_linear_1.weight.dtype == torch.float32
    assert model.condition_embedder.time_proj.weight.dtype == torch.float16
    assert model.dtype == torch.float16

    x = torch.randn(1, 48, 1, 4, 4, dtype=torch.float32).to(model.dtype)
    context = torch.randn(1, 4, TINY.text_dim, dtype=torch.float32).to(model.dtype)
    with torch.no_grad():
        output = model(x, torch.full((1, 4), 500.0), context).sample
    assert output.dtype == torch.float16
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_a_plain_checkpoint_runs_at_its_own_precision(tmp_path):
    """D9: the file's dtype is the compute dtype unless the caller overrides it."""
    _, path = _write(tmp_path)
    assert load_wan_native_transformer(str(path), "cpu").dtype == torch.float16


def test_a_mixed_checkpoint_loads_at_one_trunk_dtype(tmp_path):
    """A Linear that is Half in its weight and Float in its bias cannot run.

    ``Wan2_2-TI2V-5B-Turbo_fp16.safetensors`` is written like this.
    """
    _, path = _write(tmp_path, mixed=True)
    model = load_wan_native_transformer(str(path), "cpu")
    assert model.dtype == torch.float16
    for module in model.modules():
        weight, bias = getattr(module, "weight", None), getattr(module, "bias", None)
        if torch.is_tensor(weight) and torch.is_tensor(bias):
            assert bias.dtype == weight.dtype, module
    assert model.patch_embedding.weight.dtype == torch.float32
    assert model.blocks[0].norm2.weight.dtype == torch.float32

    x = torch.randn(1, 48, 1, 4, 4, dtype=torch.float16)
    context = torch.randn(1, 4, TINY.text_dim, dtype=torch.float16)
    with torch.no_grad():
        output = model(x, torch.tensor([500.0]), context).sample
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_the_file_it_is_given_is_the_file_it_reads(tmp_path):
    """Weights may be renamed freely; nothing is looked up by filename."""
    _, path = _write(tmp_path)
    renamed = tmp_path / "whatever-i-called-this.safetensors"
    path.rename(renamed)
    model = load_wan_native_transformer(str(renamed), "cpu")
    assert model.params == TINY


def test_checkpoint_rejects_missing_and_wrong_shapes(tmp_path):
    # Not patch_embedding: a structural key is refused earlier, by detect_params.
    _write(tmp_path, missing="blocks.0.self_attn.q.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_wan_native_transformer(str(_checkpoint(tmp_path)), "cpu")
    _write(tmp_path, wrong_shape="blocks.0.self_attn.q.weight")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_wan_native_transformer(str(_checkpoint(tmp_path)), "cpu")


def _checkpoint(root):
    return root / "split_files" / "diffusion_models" / "wan2.2_ti2v_5B_fp16.safetensors"


def test_official_5b_topology_is_constructible_on_meta():
    params = Wan22Params(in_channels=48, out_channels=48, dim=3072, num_layers=30,
                         ffn_dim=14336, freq_dim=256, text_dim=4096, patch_size=(1, 2, 2))
    with torch.device("meta"):
        model = WanNativeDiT(params)
    assert params.heads == 24
    assert model.patch_embedding.weight.shape == (3072, 48, 1, 2, 2)
    assert len(model.blocks) == 30
    assert model.blocks[0].ffn_in.weight.shape == (14336, 3072)
    assert all(p.is_meta for p in model.parameters())
