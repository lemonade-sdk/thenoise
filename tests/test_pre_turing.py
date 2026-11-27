"""Capability-based fallbacks for older NVIDIA GPUs (no GPU needed)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from thenoise.utils import attention, device


@pytest.mark.parametrize("where, hip", [("cpu", None), ("mps", None), ("cuda:0", "6.4")])
def test_non_nvidia_never_queries_cuda(monkeypatch, where, hip):
    def forbidden(*args):
        raise AssertionError("CUDA queried for another backend")

    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    assert device.nvidia_capability(where) is None
    assert device.int_mm_supported(torch.device(where))
    assert device.compile_supported(where)


def test_int_mm_and_compile_support(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    caps = {0: (6, 0), 1: (8, 9), 2: (7, 5)}
    def capability(dev):
        return caps[dev.index if dev.index is not None else torch.cuda.current_device()]
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert not device.int_mm_supported(torch.device("cuda:0"))
    assert device.int_mm_supported(torch.device("cuda:2"))
    assert not device.compile_supported("cuda:0")
    assert device.compile_supported("cuda:1")
    assert device.nvidia_capability("cuda") == (8, 9)


def _long_bf16(seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (1, 2, 4096, 64)          # 4096 x 4096 = 2^24: the long path
    return [torch.randn(shape, generator=g).to(torch.bfloat16) for _ in range(3)]


@pytest.fixture
def pre_ampere(monkeypatch):
    """Route CPU tensors through the sub-sm_80 path."""
    monkeypatch.setattr(attention, "_bf16_needs_fp16", lambda q: q.dtype is torch.bfloat16)


def test_bf16_attention_fallback_runs_fp16_and_matches(pre_ampere):
    q, k, v = _long_bf16()
    hq, hk, hv, back = attention._memory_efficient_dtype(q, k, v)
    assert hq.dtype is torch.float16 and back is torch.bfloat16
    got = attention.attention(
        [q, k, v],
        attn_params=attention.AttentionParams())
    want = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    want = want.transpose(1, 2).reshape(1, 4096, -1)
    assert got.dtype is torch.bfloat16
    torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)


def test_bf16_attention_fallback_short_sequences_untouched(pre_ampere):
    q = torch.randn(1, 2, 64, 64).to(torch.bfloat16)
    assert attention._memory_efficient_dtype(q, q, q)[3] is None


@pytest.mark.parametrize("fallback", [True, False], ids=["fallback", "current_hardware_graph_unchanged"])
def test_attention_compiles_fullgraph(monkeypatch, fallback):
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    monkeypatch.setattr(attention, "_bf16_needs_fp16", lambda q: fallback)
    q, k, v = _long_bf16(1)
    params = attention.AttentionParams()
    torch._dynamo.reset()
    try:
        compiled = torch.compile(attention.attention, fullgraph=True)
        got = compiled(q, k, v, attn_params=params)
        want = attention.attention(q, k, v, attn_params=params)
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
        if not fallback:
            original = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 4096, -1)
            torch.testing.assert_close(got, original, atol=2e-2, rtol=2e-2)
    finally:
        torch._dynamo.reset()


def test_bf16_attention_fallback_never_overflows_fp16(pre_ampere):
    q, k, v = _long_bf16(2)
    q[0, 0, :8] = 1e5
    v[0, 1, :8] = -7e4
    hq, hk, hv, _ = attention._memory_efficient_dtype(q, k, v)
    assert all(torch.isfinite(t).all() for t in (hq, hk, hv))
    assert torch.tensor(attention._FP16_MAX, dtype=torch.bfloat16).item() == attention._FP16_MAX


def test_mask_untouched_without_fallback(monkeypatch):
    seen = {}

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0):
        seen["mask"] = attn_mask
        return torch.zeros_like(q)

    monkeypatch.setattr(attention.F, "scaled_dot_product_attention", sdpa)
    q = torch.randn(1, 2, 64, 32).to(torch.bfloat16)
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
    attention.attention([q, q, q], attn_params=attention.AttentionParams(attention_mask=mask))
    assert seen["mask"] is mask                      # same object, same dtype and precision


def test_fallback_mask_keeps_meaning(pre_ampere):
    q, k, v = _long_bf16(3)
    mask = torch.zeros(1, 1, 4096, 4096, dtype=torch.float32)
    mask[..., 2048:] = float("-inf")
    mask[..., 1024:2048] = -1e30
    got = attention.attention([q, k, v], attn_params=attention.AttentionParams(attention_mask=mask))
    want = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask)
    want = want.transpose(1, 2).reshape(1, 4096, -1)
    assert torch.isfinite(got.float()).all()
    torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("where, hip", [("cpu", None), ("cuda:0", "6.4"), ("cuda:0", None)],
                         ids=["cpu", "rocm", "ampere"])
def test_no_shape_guard_when_fallback_off(monkeypatch, where, hip):
    class Q:
        dtype = torch.bfloat16
        device = torch.device(where)
        is_cuda = device.type == "cuda"
        @property
        def shape(self):
            raise AssertionError("shape read")

    def capability(dev):
        if hip or where == "cpu":
            raise AssertionError("capability queried")
        return (8, 0)

    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    q = Q()
    out = attention._memory_efficient_dtype(q, q, q)
    assert all(t is q for t in out[:3]) and out[3] is None


def test_fallback_mask_conversion():
    mask = torch.tensor([float("-inf"), -1e30, 0, 1e30])
    got = attention._fp16_mask(mask)
    assert got.dtype == torch.float16
    assert torch.isneginf(got[:2]).all()
    assert got[2] == 0 and got[3] == attention._FP16_MAX


@pytest.mark.parametrize("command, module, handler", [
    (["upscale", "--pixel-upscaler", "u", "--input", "i"], "upscale_cli", "run_upscale"),
    (["controlmap"], "controlmap_cli", "run_controlmap"),
])
def test_non_diffusion_commands_do_not_probe_compile_support(monkeypatch, command, module, handler):
    import importlib
    from thenoise import __main__

    def forbidden(*args):
        raise AssertionError("early compile probe")

    called = []
    monkeypatch.setattr(__main__, "_eager_without_triton", forbidden)
    monkeypatch.setattr(importlib.import_module("thenoise." + module), handler,
                        lambda args: called.append(args.command))
    __main__.main(command)
    assert called == [command[0]]
