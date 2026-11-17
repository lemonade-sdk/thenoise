"""CPU-only tests for 22B LTX block residency hooks (no Diffusers needed)."""
from __future__ import annotations

import torch
import pytest
from torch import nn

from thenoise.dit.ltx_video.offload import LTXBlockOffload, _Weights


class _Block(nn.Module):
    def __init__(self, index, events):
        super().__init__()
        self.layer = nn.Linear(3, 3, bias=False)
        self.register_buffer("int8_mock", torch.arange(3, dtype=torch.int8))
        self.index = index
        self.events = events

    def forward(self, x):
        self.events.append(("block", self.index, self.layer.weight.device.type))
        return self.layer(x)


class _DiT(nn.Module):
    def __init__(self, events, fail=False):
        super().__init__()
        self.root = nn.Linear(3, 3)
        self.top = nn.Parameter(torch.ones(3))
        self.register_buffer("constant", torch.ones(3))
        self.transformer_blocks = nn.ModuleList([_Block(i, events) for i in range(3)])
        self.events = events
        self.fail = fail

    def forward(self, x):
        x = self.root(x) + self.top + self.constant
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if self.fail and i == 1:
                raise RuntimeError("forced forward failure")
        return x


def _capture(model):
    return {name: tensor for name, tensor in (*model.named_parameters(), *model.named_buffers())}


def test_transfer_master_restore_on_failed_partial_upload():
    module = nn.Module()
    module.register_parameter("p", nn.Parameter(torch.ones(2), requires_grad=False))
    # Simulate an OOM after the first parameter has been temporarily replaced.
    module.register_buffer("bad", torch.ones(2))
    group = _Weights(module.modules())
    before = _capture(module)
    def explode(_):
        raise RuntimeError("transfer failed")
    module.bad.to = explode
    try:
        with pytest.raises(RuntimeError, match="transfer failed"):
            group.upload(torch.device("cpu"))
    finally:
        del module.bad.to
    after = _capture(module)
    assert all(after[name] is before[name] for name in before)


def test_offloader_restores_original_objects_every_step_cpu_simulation(monkeypatch):
    # Emulate the CUDA transfer at the tensor copy seam, including final restore,
    # while keeping small weights on the CPU for this unit test.
    events = []
    dit = _DiT(events).eval().requires_grad_(False)
    original = _capture(dit)
    original_parameter_to = nn.Parameter.to
    original_tensor_to = torch.Tensor.to
    copies = []
    def stub_param_to(self, device, *args, **kwargs):
        copies.append(("parameter", str(device)))
        return original_parameter_to(self, "cpu", *args, **kwargs).clone()
    def stub_tensor_to(self, device, *args, **kwargs):
        copies.append(("buffer", str(device)))
        return original_tensor_to(self, "cpu", *args, **kwargs).clone()
    monkeypatch.setattr(nn.Parameter, "to", stub_param_to)
    monkeypatch.setattr(torch.Tensor, "to", stub_tensor_to)
    offload = LTXBlockOffload(dit, "cuda:0", "cpu")
    x = torch.randn(2, 3)
    with torch.no_grad():
        a = dit(x)
        b = dit(x)
    assert torch.isfinite(a).all() and torch.equal(a, b)
    assert [i for kind, i, device in events] == [0, 1, 2] * 2
    assert all(device == "cpu" for _, _, device in events)
    assert any(device == "cuda:0" for _, device in copies)
    after = _capture(dit)
    assert all(after[name] is original[name] for name in original)
    offload.close()
    assert all(after[name] is original[name] for name in original)


def test_offloader_cleans_up_after_exception(monkeypatch):
    dit = _DiT([], fail=True).eval().requires_grad_(False)
    before = _capture(dit)
    original_parameter_to = nn.Parameter.to
    original_tensor_to = torch.Tensor.to
    monkeypatch.setattr(nn.Parameter, "to", lambda self, device: original_parameter_to(self, "cpu").clone())
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device: original_tensor_to(self, "cpu").clone())
    guard = LTXBlockOffload(dit, "cuda:0", "cpu")
    with pytest.raises(RuntimeError, match="forced forward failure"):
        dit(torch.randn(1, 3))
    after = _capture(dit)
    assert all(after[name] is before[name] for name in before)
    guard.close()


def test_offloader_rejects_unfrozen_and_non_cpu_master():
    dit = _DiT([])
    with pytest.raises(ValueError, match="frozen inference"):
        LTXBlockOffload(dit, "cuda:0", "cpu")
    with pytest.raises(ValueError, match="CPU master"):
        LTXBlockOffload(dit.eval().requires_grad_(False), "cuda:0", "cuda:1")
