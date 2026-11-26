"""Capability-based fallbacks for older NVIDIA GPUs (no GPU needed)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from thenoise.utils import attention, device, device_patches


def test_kernel_backends_by_capability(monkeypatch):
    import comfy_kitchen
    off = []
    monkeypatch.setattr(comfy_kitchen, "disable_backend", off.append)
    monkeypatch.setattr(device_patches, "_cuda_capability", lambda d: None)
    assert device_patches._kernel_backends("cpu") is False
    monkeypatch.setattr(device_patches, "_cuda_capability", lambda d: (8, 6))
    assert device_patches._kernel_backends("cuda:0") is False
    monkeypatch.setattr(device_patches, "_cuda_capability", lambda d: (7, 0))
    assert device_patches._kernel_backends("cuda:0") is True and off == ["cuda"]
    off.clear()
    monkeypatch.setattr(device_patches, "_cuda_capability", lambda d: (6, 0))
    assert device_patches._kernel_backends("cuda:0") is True and off == ["cuda", "triton"]


def test_int_mm_and_compile_support(monkeypatch):
    assert device.int_mm_supported(torch.device("cpu"))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(device, "_capability", lambda i: (6, 0))
    assert not device.int_mm_supported(torch.device("cuda", 0))
    monkeypatch.setattr(device, "_capability", lambda i: (7, 5))
    assert device.int_mm_supported(torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    caps = {0: (6, 0), 1: (6, 1)}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: caps[i])
    assert not device.compile_supported()
    caps[1] = (7, 5)
    assert device.compile_supported()


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


def test_bf16_attention_fallback_compiles_fullgraph(pre_ampere, monkeypatch):
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    q, k, v = _long_bf16(1)

    def fn(q, k, v):
        hq, hk, hv, back = attention._memory_efficient_dtype(q, k, v)
        return F.scaled_dot_product_attention(hq, hk, hv).to(back)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v), fn(q, k, v))
