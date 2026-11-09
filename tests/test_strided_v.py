"""Only the qualified Qwen T2I path may skip the V layout conversion."""

import torch
from types import SimpleNamespace
from thenoise.dit.kvcache import KVCache
from thenoise.dit.qwen_image21 import models as qwen21
from thenoise.utils.attention import uniform_layout


def test_strided_cache_value_storage_and_layout_override():
    cache = KVCache("cond", "prefix")
    cache.strided_values = True
    bufs = cache.allocate("block", (1, 2, 7, 4), torch.float32, torch.device("cpu"))
    assert bufs.v.shape == (1, 2, 7, 4)
    assert bufs.v.stride() == (56, 4, 8, 1)
    bufs.v.copy_(torch.arange(56, dtype=torch.float32).reshape(1, 2, 7, 4))
    q = k = torch.randn(1, 2, 7, 4)
    _, _, direct = uniform_layout(q, k, bufs.v, allow_strided_v=True)
    _, _, copied = uniform_layout(q, k, bufs.v)
    assert direct is bufs.v
    assert copied.is_contiguous()
    assert torch.equal(direct, copied)
    torch.testing.assert_close(
        torch.nn.functional.scaled_dot_product_attention(q, k, direct),
        torch.nn.functional.scaled_dot_product_attention(q, k, copied),
    )
    ordinary = KVCache("cond", "prefix")
    assert ordinary.allocate("block", (1, 2, 7, 4), torch.float32,
                             torch.device("cpu")).v.is_contiguous()


def test_only_target_segment_skips_value_copy(monkeypatch):
    seen = []

    def capture(qkv, *, attn_params):
        q, k, v = qkv
        seen.append((q.shape[2], k.shape[2], attn_params.allow_strided_v))
        return torch.zeros(q.shape[0], q.shape[2], q.shape[1] * q.shape[3])

    monkeypatch.setattr(qwen21, "eager_sdpa_attention", capture)
    q = torch.randn(1, 1, 7, 4)
    plan = qwen21.AttentionPlan(
        mode="off",
        segments=((0, 3, torch.ones(1, 1, 3, 3, dtype=torch.bool)),
                  (3, 7, None)),
        strided_v=True,
        target_len=4,
    )
    plan.run(q, q, q)
    assert seen == [(3, 3, False), (4, 7, True)]


def test_strided_v_gate_rejects_untested_shapes_and_devices(monkeypatch):
    from thenoise.models import QwenImage21Model

    class Dit:
        transformer_blocks = [SimpleNamespace(attn=SimpleNamespace(to_qkv=object()))]

        def modules(self):
            return []

    model = object.__new__(QwenImage21Model)
    model.device = "cuda"
    model.dit = Dit()
    monkeypatch.setattr(torch.version, "hip", "7.16.0")
    monkeypatch.setattr(QwenImage21Model, "_supports_text_prefix_cache",
                        staticmethod(lambda dit: True))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev: SimpleNamespace(gcnArchName="gfx1151:sramecc+"))
    assert model._use_strided_v(3072) and model._use_strided_v(4096)
    assert not model._use_strided_v(16384)
    model.dit.transformer_blocks[0].attn.to_qkv = None
    assert not model._use_strided_v(4096)
    model.dit.transformer_blocks[0].attn.to_qkv = object()
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev: SimpleNamespace(gcnArchName="sm_120"))
    assert not model._use_strided_v(4096)
