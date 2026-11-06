"""Qwen-Image 2.1 adapter specifics that need no weights."""
from __future__ import annotations

import torch
import pytest
from PIL import Image
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.registry import registry
from conftest import comfy_quant, write_safetensors

from thenoise.dit.qwen_image21.models import (
    Attention,
    QwenImage21Params,
    QwenImage21Transformer2DModel,
)
from thenoise.models import QwenImage21Model
import thenoise.models.qwen_image21 as qwen21_adapter
from thenoise.models.base import Conditioning, DiffusionModel
from thenoise.models.config import EncodePromptArgs, SamplingParams
from thenoise.memory import MemoryManager, _weight_bytes
from thenoise.upscale import LatentTranscodeNet, Qwen21TranscodeUpscaler
from thenoise.upscale import qwen21_transcode


class FakeVAE:
    """Just the surface the transcoder is allowed to reach for."""

    z_dim = 4

    def decode_features(self, latents, *, blocks=1):
        b, _, h, w = latents.shape
        return torch.zeros(b, 6, 2 * h, 2 * w, dtype=latents.dtype)


def _bare(**attrs):
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(QwenImage21Model)
    model.device = "cpu"
    model.dtype = torch.float32
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def _tiny_qwen21_dit():
    return QwenImage21Transformer2DModel(
        QwenImage21Params(
            in_channels=2,
            out_channels=2,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=1,
            context_in_dim=4,
            mlp_ratio=2,
            axes_dims_rope=(2, 2, 4),
        )
    ).eval().requires_grad_(False)


def _quantize_projection(module, *, convrot=True, groupsize=256, scale_value=0.01):
    rows, cols = module.weight.shape
    qdata = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols)
    qdata = ((qdata % 255) - 127).to(torch.int8)
    scale = torch.full((rows, 1), scale_value, dtype=torch.float32)
    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=(rows, cols),
        is_weight=True,
        convrot=convrot,
        convrot_groupsize=groupsize,
    )
    module.load_quantized(QuantizedTensor(qdata, "TensorWiseINT8Layout", params))


def _quantize_all_changed_projections(dit, **kwargs):
    for block in dit.transformer_blocks:
        projections = [
            block.attn.to_q,
            block.attn.to_k,
            block.attn.to_v,
            block.attn.to_out[0],
            block.img_mlp.gate_up,
            block.img_mlp.out,
        ]
        for projection in projections:
            _quantize_projection(projection, **kwargs)


def _assert_qkv_aliases(attn):
    fused = attn.to_qkv.weight
    assert isinstance(fused, QuantizedTensor)
    for module in (attn.to_q, attn.to_k, attn.to_v):
        weight = module.weight
        assert weight._qdata.untyped_storage()._cdata == fused._qdata.untyped_storage()._cdata
        assert weight.params.scale.untyped_storage()._cdata == fused.params.scale.untyped_storage()._cdata


def _assert_qkv_storage_aliases(attn):
    fused = attn.to_qkv.weight
    if isinstance(fused, QuantizedTensor):
        return _assert_qkv_aliases(attn)
    for module in (attn.to_q, attn.to_k, attn.to_v):
        assert module.weight.untyped_storage()._cdata == fused.untyped_storage()._cdata


def _qwen_qkv_lora(up_scale, seed):
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for index, name in enumerate(("to_q", "to_k", "to_v")):
        prefix = f"transformer_blocks.0.attn.{name}"
        state[f"{prefix}.lora_A.weight"] = torch.randn(
            1, 8, generator=generator
        ) * 0.2
        state[f"{prefix}.lora_B.weight"] = torch.randn(
            8, 1, generator=generator
        ) * up_scale
    return state


def _mock_tested_t2i_cache_runtime(
    monkeypatch,
    *,
    arch="gfx1151",
    version="2.14.0+rocm10.1.0rc2",
    hip="7.16.26382",
    backend="hip",
    kitchen_version="0.2.40",
):
    """Make the runtime qualification deterministic without requiring a GPU."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName=f"{arch}:sramecc+:xnack-"),
    )
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(registry, "get_capable_backend", lambda op: backend)
    monkeypatch.setattr(qwen21_adapter, "package_version", lambda package: kitchen_version)


def _bare_lora_adapter(dit, lora_dir, *, dit_path=None, device="cpu"):
    return _bare(
        dit=dit,
        lora_dir=str(lora_dir),
        dit_path=dit_path,
        device=device,
        _active_lora_result=None,
        _active_lora_spec=None,
        _lora_state_unfit=False,
    )


def _assert_fused_projection_matches_named(attn, x):
    with torch.no_grad():
        named = torch.cat([module(x) for module in (attn.to_q, attn.to_k, attn.to_v)], -1)
        joined = attn.to_qkv(x)
    torch.testing.assert_close(joined, named, atol=1e-6, rtol=1e-6)


def test_the_vision_tokens_get_an_rgb_composite_not_a_dropped_alpha():
    """A transparent reference must not reach Qwen3-VL as its raw RGB."""
    transparent_red = Image.new("RGBA", (32, 32), (255, 0, 0, 0))

    images = _bare()._encoder_images(
        EncodePromptArgs(prompt="p", image=transparent_red)
    )

    assert [img.mode for img in images] == ["RGB"]
    assert images[0].getpixel((0, 0)) == (255, 255, 255)


def test_every_reference_in_a_list_is_composited():
    images = _bare()._encoder_images(
        EncodePromptArgs(
            prompt="p",
            image=[Image.new("RGBA", (32, 32), (0, 0, 255, 0)), Image.new("RGB", (32, 32), "red")],
        )
    )

    assert images[0].getpixel((0, 0)) == (255, 255, 255)  # transparent -> white
    assert images[1].getpixel((0, 0)) == (255, 0, 0)  # opaque -> untouched


def test_the_vision_half_and_the_latent_half_are_fitted_by_the_same_rule():
    """The reference latent is spliced into the vision tokens' place, so both halves
    have to agree on shape."""
    refs = [Image.new("RGB", (200, 100), "gray"), Image.new("RGB", (60, 300), "gray")]

    images = _bare()._encoder_images(EncodePromptArgs(prompt="p", image=refs))

    assert [img.size for img in images] == [
        QwenImage21Model.REFERENCE_SIZING.target_size(*img.size) for img in refs
    ]


def test_the_adapter_does_not_override_the_shared_decode():
    """The alpha drop-out used to be a ``decode`` override; it must not come back."""
    assert "decode" not in QwenImage21Model.__dict__
    assert QwenImage21Model.decode is DiffusionModel.decode


def test_text_only_prefix_cache_matches_uncached_sampling_steps(monkeypatch):
    """The causal text prefix is fixed at t=0 and can be reused exactly for T2I."""
    # Simulate a qualified accelerator while keeping this exactness test on CPU.
    monkeypatch.setattr(
        QwenImage21Model,
        "_text_prefix_cache_runtime_reason",
        staticmethod(lambda device: None),
    )
    torch.manual_seed(3)
    dit = _tiny_qwen21_dit()
    adapter = _bare(
        dit=dit, _seqs={}, _kv_caches=None, KV_CACHE_WITHOUT_REFERENCE=True
    )
    x = torch.randn(1, 2, 2, 3)
    cond = Conditioning(cond=torch.randn(1, 4, 4))
    params = SamplingParams(
        height=32,
        width=48,
        steps=3,
        seed=0,
        guidance_scale=1.0,
        sampler="euler",
        kv_cache=True,
    )

    adapter.prepare_latent(x, cond, params)
    cache = adapter.kv_cache("cond")
    assert cache is not None
    assert cache.cached_slice == "prefix"

    with torch.no_grad():
        # First step fills the complete sequence. At the next timestep the cached
        # path projects/updates only target tokens while the reference recomputes
        # the fixed text prefix.
        adapter.denoise_step(x, torch.tensor([0.8]), cond, 1.0, 0)
        assert cache.filled
        cached = adapter.denoise_step(x, torch.tensor([0.4]), cond, 1.0, 1)
        plain = dit(x, torch.tensor([0.4]), adapter._seqs["cond"])

    assert torch.equal(cached, plain)


def test_unsupported_checkpoint_falls_back_from_explicit_text_prefix_cache(caplog):
    dit = _tiny_qwen21_dit()
    adapter = _bare(
        dit=dit,
        _seqs={},
        _kv_caches=None,
        KV_CACHE_WITHOUT_REFERENCE=False,
        KV_CACHE_WITHOUT_REFERENCE_REASON="requires the validated INT8-ConvRot profile",
    )
    x = torch.randn(1, 2, 2, 3)
    cond = Conditioning(cond=torch.randn(1, 4, 4))
    params = SamplingParams(
        height=32,
        width=48,
        steps=2,
        seed=0,
        guidance_scale=1.0,
        sampler="euler",
        kv_cache=True,
    )

    adapter.prepare_latent(x, cond, params)

    assert adapter._kv_caches is None
    assert "validated INT8-ConvRot" in caplog.text


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"arch": "gfx1150"}, "gfx1151"),
        ({"version": "2.13.0+rocm10.1.0rc2"}, "PyTorch 2.14.0"),
        ({"version": "2.14.0+rocm6.3"}, "ROCm 10.1"),
        ({"hip": "6.3.42134"}, "HIP 7.16"),
        ({"kitchen_version": "0.2.41"}, "comfy-kitchen 0.2.40"),
        ({"backend": "triton"}, "HIP int8_linear backend"),
    ],
)
def test_text_prefix_cache_runtime_requires_the_measured_stack(monkeypatch, overrides, reason):
    _mock_tested_t2i_cache_runtime(monkeypatch, **overrides)

    assert QwenImage21Model._text_prefix_cache_runtime_reason("cuda:0") is not None
    assert reason in QwenImage21Model._text_prefix_cache_runtime_reason("cuda:0")


def test_text_prefix_cache_runtime_accepts_only_the_measured_stack(monkeypatch):
    _mock_tested_t2i_cache_runtime(monkeypatch)

    assert QwenImage21Model._text_prefix_cache_runtime_reason("cuda:0") is None
    assert "ROCm GPU" in QwenImage21Model._text_prefix_cache_runtime_reason("cpu")


def test_runtime_lora_disables_only_the_t2i_prefix_cache(monkeypatch):
    # Keep tensors on CPU while isolating the runtime-LoRA branch qualification.
    monkeypatch.setattr(
        QwenImage21Model,
        "_text_prefix_cache_runtime_reason",
        staticmethod(lambda device: None),
    )
    dit = _tiny_qwen21_dit()
    adapter = _bare(
        dit=dit,
        _seqs={},
        _kv_caches=None,
        KV_CACHE_WITHOUT_REFERENCE=True,
        _text_prefix_cache_profile_supported=True,
    )
    projection = dit.transformer_blocks[0].attn.to_q
    _quantize_projection(projection, convrot=False)
    projection.set_runtime_lora(
        torch.ones(1, dit.dim), torch.ones(projection.out_features, 1)
    )

    x = torch.randn(1, 2, 2, 3)
    cond = Conditioning(cond=torch.randn(1, 4, 4))
    params = SamplingParams(
        height=32, width=48, steps=2, seed=0, guidance_scale=1.0,
        sampler="euler", kv_cache=True,
    )

    assert not adapter.resolve_text_prefix_cache()
    adapter.prepare_latent(x, cond, params)
    assert adapter._kv_caches is None

    # Reference editing keeps its previously validated cache semantics even when
    # a runtime branch is active.
    adapter.prepare_latent(
        x, cond, params, ref=[torch.randn(1, 2, 2, 2)],
        ref_method="index_timestep_zero",
    )
    assert adapter.kv_cache("cond") is not None


def test_fused_qkv_projections_match_separate_projections():
    """Joining projection rows preserves each Q/K/V output within float precision."""
    torch.manual_seed(4)
    dit = _tiny_qwen21_dit()
    x = torch.randn(1, 2, 2, 3)
    seq = dit.build_sequence(x, torch.randn(1, 4, 4))
    t = torch.tensor([0.6])
    attn = dit.transformer_blocks[0].attn
    attn.to_q.set_runtime_lora(
        torch.randn(2, dit.dim), torch.randn(attn.inner_dim, 2)
    )

    with torch.no_grad():
        separate = dit(x, t, seq)
        assert dit.fuse_qkv_projections()
        fused = dit(x, t, seq)
        dit.unfuse_qkv_projections()
        restored = dit(x, t, seq)

    assert torch.allclose(fused, separate, atol=1e-6, rtol=1e-6)
    assert torch.equal(restored, separate)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_int8_fused_qkv_projection_matches_separate_rows(device):
    """The validated quantized path joins rows without changing any output bit."""
    torch.manual_seed(6)
    attn = Attention(256, 1, 256)
    for index, projection in enumerate((attn.to_q, attn.to_k, attn.to_v)):
        qdata = torch.randint(-127, 128, (256, 256), dtype=torch.int8)
        scale = torch.full((256, 1), 0.01, dtype=torch.float32)
        params = TensorWiseINT8Layout.Params(
            scale=scale,
            orig_dtype=torch.bfloat16,
            orig_shape=(256, 256),
            is_weight=True,
            convrot=True,
            convrot_groupsize=256,
        )
        projection.load_quantized(
            QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
        )
    attn.to(device)
    x = torch.randn(1, 128, 256, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        separate = torch.cat([module(x) for module in (attn.to_q, attn.to_k, attn.to_v)], -1)
        assert attn.fuse_qkv_projections()
        fused = attn.to_qkv(x)
    if device == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(fused, separate)


def test_qkv_fusion_falls_back_atomically_for_mixed_quantized_layouts():
    from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout

    dit = _tiny_qwen21_dit()
    first = dit.transformer_blocks[0].attn
    second = dit.transformer_blocks[1].attn
    for projection in (first.to_q, first.to_k, first.to_v):
        _quantize_projection(projection)
    rows, cols = second.to_q.weight.shape
    second.to_q.load_quantized(
        QuantizedTensor(
            torch.zeros(rows, cols, dtype=torch.float8_e4m3fn),
            "TensorCoreFP8Layout",
            TensorCoreFP8Layout.Params(
                scale=torch.ones(1),
                orig_dtype=torch.bfloat16,
                orig_shape=(rows, cols),
            ),
        )
    )
    original = [
        (attn, tuple(module.weight for module in (attn.to_q, attn.to_k, attn.to_v)))
        for attn in (first, second)
    ]

    assert not dit.fuse_qkv_projections()
    for attn, weights in original:
        assert attn.to_qkv is None
        assert all(
            module.weight is weight
            for module, weight in zip((attn.to_q, attn.to_k, attn.to_v), weights)
        )


def test_qkv_fusion_falls_back_for_inconsistent_quantized_compute_dtypes():
    from dataclasses import replace

    attn = Attention(8, 1, 8)
    for projection in (attn.to_q, attn.to_k, attn.to_v):
        _quantize_projection(projection)
    weight = attn.to_k.weight
    attn.to_k.load_quantized(
        QuantizedTensor(
            weight._qdata,
            "TensorWiseINT8Layout",
            replace(weight.params, orig_dtype=torch.float16),
        )
    )

    assert not attn.fuse_qkv_projections()
    assert attn.to_qkv is None
    assert all(
        isinstance(projection.weight, QuantizedTensor)
        for projection in (attn.to_q, attn.to_k, attn.to_v)
    )


def test_qkv_fusion_falls_back_for_transposed_quantized_weights():
    from dataclasses import replace

    attn = Attention(8, 1, 8)
    for projection in (attn.to_q, attn.to_k, attn.to_v):
        _quantize_projection(projection)
    weight = attn.to_k.weight
    attn.to_k.load_quantized(
        QuantizedTensor(
            weight._qdata,
            "TensorWiseINT8Layout",
            replace(weight.params, transposed=True),
        )
    )

    assert not attn.fuse_qkv_projections()
    assert attn.to_qkv is None


def test_lora_switches_rebuild_bf16_qkv_fusion(tmp_path):
    dit = _tiny_qwen21_dit()
    for block in dit.transformer_blocks:
        for projection in (block.attn.to_q, block.attn.to_k, block.attn.to_v):
            torch.nn.init.zeros_(projection.weight)
    assert dit.fuse_qkv_projections()
    attn = dit.transformer_blocks[0].attn
    original = [module.weight.detach().clone() for module in (attn.to_q, attn.to_k, attn.to_v)]
    write_safetensors(tmp_path / "style_a.safetensors", _qwen_qkv_lora(0.5, 10))
    write_safetensors(tmp_path / "style_b.safetensors", _qwen_qkv_lora(1.25, 11))
    model = _bare_lora_adapter(dit, tmp_path)
    x = torch.randn(2, 5, dit.dim)

    model.switch_loras(["style_a.safetensors"], dit)
    assert attn.to_qkv is not None
    _assert_fused_projection_matches_named(attn, x)
    style_a = attn.to_qkv.weight.detach().clone()

    model.switch_loras(["style_b.safetensors"], dit)
    assert attn.to_qkv is not None
    _assert_fused_projection_matches_named(attn, x)
    assert not torch.equal(style_a, attn.to_qkv.weight)

    model.switch_loras(None, dit)
    assert attn.to_qkv is not None
    _assert_fused_projection_matches_named(attn, x)
    for projection, weight in zip((attn.to_q, attn.to_k, attn.to_v), original):
        assert torch.equal(projection.weight, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA/ROCm device is required")
def test_lora_switching_after_cpu_gpu_offload_preserves_fusion(tmp_path):
    dit = _tiny_qwen21_dit()
    for block in dit.transformer_blocks:
        for projection in (block.attn.to_q, block.attn.to_k, block.attn.to_v):
            torch.nn.init.zeros_(projection.weight)
    assert dit.fuse_qkv_projections()
    attn = dit.transformer_blocks[0].attn
    original = [module.weight.detach().clone() for module in (attn.to_q, attn.to_k, attn.to_v)]
    write_safetensors(tmp_path / "style_a.safetensors", _qwen_qkv_lora(0.5, 20))
    write_safetensors(tmp_path / "style_b.safetensors", _qwen_qkv_lora(1.25, 21))
    model = _bare_lora_adapter(dit, tmp_path, device="cuda")
    memory = MemoryManager("cuda", "cpu")
    memory.register("dit", dit)

    with memory.stage("dit", pixels=0):
        model.switch_loras(["style_a.safetensors"], dit)
        _assert_qkv_storage_aliases(attn)
        _assert_fused_projection_matches_named(
            attn, torch.randn(2, 5, dit.dim, device="cuda")
        )
    memory.offload("dit")
    assert attn.to_q.weight.device.type == "cpu"
    _assert_qkv_storage_aliases(attn)
    model.switch_loras(["style_b.safetensors"], dit)
    _assert_qkv_storage_aliases(attn)
    _assert_fused_projection_matches_named(attn, torch.randn(2, 5, dit.dim))

    with memory.stage("dit", pixels=0):
        model.switch_loras(None, dit)
        _assert_qkv_storage_aliases(attn)
        _assert_fused_projection_matches_named(
            attn, torch.randn(2, 5, dit.dim, device="cuda")
        )
        for projection, weight in zip((attn.to_q, attn.to_k, attn.to_v), original):
            assert torch.equal(projection.weight, weight.to("cuda"))
    memory.offload("dit")


def test_quantized_runtime_and_baked_lora_switches_restore_fused_qkv(tmp_path):
    from thenoise.utils.loader import build_quantized_restore_map

    dit = _tiny_qwen21_dit()
    for block in dit.transformer_blocks:
        for projection in (block.attn.to_q, block.attn.to_k, block.attn.to_v):
            _quantize_projection(projection, convrot=False, scale_value=0.01)
    assert dit.fuse_qkv_projections()

    checkpoint = tmp_path / "tiny_qwen.safetensors"
    checkpoint_state = {}
    first_attn = dit.transformer_blocks[0].attn
    original = []
    for name, projection in zip(
        ("to_q", "to_k", "to_v"),
        (first_attn.to_q, first_attn.to_k, first_attn.to_v),
    ):
        weight = projection.weight
        original.append((weight._qdata.clone(), weight.params.scale.clone()))
        prefix = f"transformer_blocks.0.attn.{name}"
        checkpoint_state[f"{prefix}.weight"] = weight._qdata.clone()
        checkpoint_state[f"{prefix}.weight_scale"] = weight.params.scale.clone()
        checkpoint_state[f"{prefix}.comfy_quant"] = comfy_quant(convrot=False)
    write_safetensors(checkpoint, checkpoint_state)
    dit._quantized_restore_map = build_quantized_restore_map(str(checkpoint))

    write_safetensors(tmp_path / "runtime.safetensors", _qwen_qkv_lora(1e-5, 12))
    write_safetensors(tmp_path / "baked.safetensors", _qwen_qkv_lora(1.0, 13))
    model = _bare_lora_adapter(dit, tmp_path, dit_path=str(checkpoint))
    x = torch.randn(2, 5, dit.dim, dtype=torch.bfloat16)

    model.switch_loras(["runtime.safetensors"], dit)
    assert all(module._lora_down is not None for module in (
        first_attn.to_q, first_attn.to_k, first_attn.to_v
    ))
    _assert_qkv_aliases(first_attn)
    _assert_fused_projection_matches_named(first_attn, x)

    model.switch_loras(["baked.safetensors"], dit)
    assert all(module._lora_down is None for module in (
        first_attn.to_q, first_attn.to_k, first_attn.to_v
    ))
    _assert_qkv_aliases(first_attn)
    _assert_fused_projection_matches_named(first_attn, x)

    model.switch_loras(None, dit)
    _assert_qkv_aliases(first_attn)
    for projection, (qdata, scale) in zip(
        (first_attn.to_q, first_attn.to_k, first_attn.to_v), original
    ):
        assert torch.equal(projection.weight._qdata, qdata)
        assert torch.equal(projection.weight.params.scale, scale)
    _assert_fused_projection_matches_named(first_attn, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA/ROCm device is required")
def test_fused_qkv_aliases_and_memory_accounting_survive_offload_cycles():
    """CPU/GPU transfers preserve the canonical int8 payload and scale views."""
    attn = Attention(8, 1, 8)
    for index, projection in enumerate((attn.to_q, attn.to_k, attn.to_v)):
        _quantize_projection(projection, scale_value=0.01)
    assert attn.fuse_qkv_projections()

    qkv_modules = torch.nn.ModuleList(
        [attn.to_q, attn.to_k, attn.to_v, attn.to_qkv]
    )
    qkv_weight = attn.to_qkv.weight
    expected = qkv_weight._qdata.nbytes + qkv_weight.params.scale.nbytes
    assert _weight_bytes(qkv_modules) == expected

    manager = MemoryManager("cuda", "cpu")
    manager.register("dit", attn)
    assert manager.weight_bytes("dit") == _weight_bytes(attn)
    for _ in range(2):
        with manager.stage("dit", pixels=0):
            torch.cuda.synchronize()
            _assert_qkv_aliases(attn)
            assert _weight_bytes(qkv_modules) == expected
        manager.offload("dit")
        _assert_qkv_aliases(attn)
        assert _weight_bytes(qkv_modules) == expected
    with manager.stage("dit", pixels=0):
        torch.cuda.synchronize()
        _assert_qkv_aliases(attn)
        assert _weight_bytes(qkv_modules) == expected
    manager.offload("dit")


@pytest.mark.parametrize(
    "bad_profile",
    ["no_convrot", "other_group", "per_tensor_scale", "transposed", "fp8", "bf16"],
)
def test_text_prefix_cache_rejects_unvalidated_projection_profiles(bad_profile):
    dit = _tiny_qwen21_dit()
    _quantize_all_changed_projections(dit)
    assert QwenImage21Model._supports_text_prefix_cache(dit)
    projection = dit.transformer_blocks[0].attn.to_q

    if bad_profile == "no_convrot":
        _quantize_projection(projection, convrot=False)
    elif bad_profile == "other_group":
        _quantize_projection(projection, groupsize=128)
    elif bad_profile == "per_tensor_scale":
        rows, cols = projection.weight.shape
        params = TensorWiseINT8Layout.Params(
            scale=torch.ones(1),
            orig_dtype=torch.bfloat16,
            orig_shape=(rows, cols),
            is_weight=True,
            convrot=True,
            convrot_groupsize=256,
        )
        projection.load_quantized(
            QuantizedTensor(torch.zeros(rows, cols, dtype=torch.int8), "TensorWiseINT8Layout", params)
        )
    elif bad_profile == "transposed":
        from dataclasses import replace

        weight = projection.weight
        projection.load_quantized(
            QuantizedTensor(
                weight._qdata,
                "TensorWiseINT8Layout",
                replace(weight.params, transposed=True),
            )
        )
    elif bad_profile == "fp8":
        from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout

        rows, cols = projection.weight.shape
        params = TensorCoreFP8Layout.Params(
            scale=torch.ones(1),
            orig_dtype=torch.bfloat16,
            orig_shape=(rows, cols),
        )
        projection.load_quantized(
            QuantizedTensor(
                torch.zeros(rows, cols, dtype=torch.float8_e4m3fn),
                "TensorCoreFP8Layout",
                params,
            )
        )
    else:
        rows, cols = projection.weight.shape
        projection.weight = torch.nn.Parameter(torch.zeros(rows, cols))
        projection._quantized = False

    assert not QwenImage21Model._supports_text_prefix_cache(dit)


def test_reference_prefix_cache_keeps_cfg_branches_exact():
    """Editing references and conditional/unconditional caches stay independent."""
    torch.manual_seed(5)
    dit = _tiny_qwen21_dit()
    adapter = _bare(dit=dit, _seqs={}, _kv_caches=None)
    x = torch.randn(1, 2, 2, 3)
    reference = torch.randn(1, 2, 2, 2)
    cond = Conditioning(
        cond=torch.randn(1, 4, 4),
        null=torch.randn(1, 3, 4),
    )
    params = SamplingParams(
        height=32,
        width=48,
        steps=3,
        seed=0,
        guidance_scale=2.0,
        sampler="euler",
        kv_cache=True,
    )

    adapter.prepare_latent(
        x, cond, params, ref=[reference], ref_method="index_timestep_zero"
    )
    assert set(adapter._kv_caches) == {"cond", "uncond"}

    with torch.no_grad():
        adapter.denoise_step(x, torch.tensor([0.8]), cond, 2.0, 0)
        assert all(cache.filled for cache in adapter._kv_caches.values())
        cached = adapter.denoise_step(x, torch.tensor([0.4]), cond, 2.0, 1)

        t = torch.tensor([0.4])
        pos = dit(x, t, adapter._seqs["cond"])
        neg = dit(x, t, adapter._seqs["uncond"])
        plain = neg + 2.0 * (pos - neg)

    assert torch.equal(cached, plain)


def test_the_adapter_upscales_with_the_trained_transcoder(monkeypatch):
    """The advertised 2x is the bridge, conditioned on this model's own VAE."""
    loaded = []

    def fake_load(device, dtype):
        loaded.append((device, dtype))
        return LatentTranscodeNet(feature_channels=6, latent_channels=4, width=4, depth=1)

    monkeypatch.setattr(qwen21_transcode, "_load_net", fake_load)
    model = _bare(_upscaler=None, vae=FakeVAE())

    upscaler = model.get_upscaler()

    assert isinstance(upscaler, Qwen21TranscodeUpscaler)
    assert upscaler.vae is model.vae  # conditioned on the model's own decoder
    assert upscaler.scale == QwenImage21Model.UPSCALE_SCALE
    assert loaded == [(torch.device("cpu"), torch.float32)]
    assert model.get_upscaler() is upscaler  # built once, then cached
