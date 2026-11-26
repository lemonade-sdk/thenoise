"""Pre-Turing NVIDIA fallbacks and the sub-sm_80 attention dtype (no GPU needed)."""
from __future__ import annotations

import torch

from thenoise.utils import attention
from thenoise.utils import device_patches as dp


def test_pre_turing_patch(monkeypatch):
    monkeypatch.setattr(dp, "_cuda_capability", lambda device: None)
    assert dp._pre_turing_fallbacks("cpu") is False
    monkeypatch.setattr(dp, "_cuda_capability", lambda device: (7, 5))
    assert dp._pre_turing_fallbacks("cuda:0") is False

    stances = []
    monkeypatch.setattr(torch.compiler, "set_stance", lambda stance, **kw: stances.append(stance))
    monkeypatch.setattr(torch, "_int_mm", torch._int_mm)
    monkeypatch.setattr(dp, "_cuda_capability", lambda device: (6, 0))
    from comfy_kitchen.registry import registry
    monkeypatch.setattr(registry, "is_available", registry.is_available)

    assert dp._pre_turing_fallbacks("cuda:0") is True
    assert stances == ["force_eager"]
    assert getattr(registry.is_available, dp._MARK, False)
    assert registry.is_available("eager")
    assert getattr(torch._int_mm, dp._MARK, False)


def test_attention_dtype_unchanged_off_cuda():
    q = torch.randn(1, 2, 8, 4, dtype=torch.bfloat16)
    out = attention._memory_efficient_dtype(q, q, q)
    assert out[0] is q and out[3] is None
