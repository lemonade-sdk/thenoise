# SPDX-License-Identifier: Apache-2.0
"""Inference-only, one-block-at-a-time residency for the native LTX DiT.

TheNoise's VideoModel/MemoryManager still own *stages*. This module only makes
one staged 22B transformer fit on the compute device without Diffusers hooks.
CPU checkpoint tensors remain the masters: each active group gets device copies
and is then restored by reference. No costly GPU -> CPU transfer per block.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn


class _Weights:
    """Direct parameters/buffers of a module group, with CPU masters held live.

    Direct dictionaries are intentional: nn.Module.to() mutates the underlying
    Parameter in place in some PyTorch versions, invalidating the CPU master.
    Restoring the original object also preserves quantized buffer subclasses.
    """

    def __init__(self, modules: Iterator[nn.Module]):
        self._parameters: list[tuple[nn.Module, str, nn.Parameter]] = []
        self._buffers: list[tuple[nn.Module, str, torch.Tensor]] = []
        for module in modules:
            for key, value in module._parameters.items():
                if value is not None:
                    if value.is_meta:
                        raise ValueError(f"cannot offload uninitialized meta parameter {key}")
                    if value.requires_grad:
                        raise ValueError("LTX block offloading requires frozen inference weights")
                    self._parameters.append((module, key, value))
            for key, value in module._buffers.items():
                if value is not None:
                    if value.is_meta:
                        raise ValueError(f"cannot offload uninitialized meta buffer {key}")
                    self._buffers.append((module, key, value))
        self._active = False

    def upload(self, device: torch.device) -> None:
        if self._active:
            raise RuntimeError("LTX weight group was already uploaded")
        # If a later transfer fails (including CUDA OOM), restore every original.
        # The CPU masters are never mutated by this path.
        try:
            for module, key, parameter in self._parameters:
                device_param = parameter.to(device)
                module._parameters[key] = nn.Parameter(device_param, requires_grad=False)
            for module, key, buffer in self._buffers:
                module._buffers[key] = buffer.to(device)
            self._active = True
        except BaseException:
            self.restore()
            raise

    def restore(self) -> None:
        # Restoration of Python references never needs another allocation or a
        # GPU -> CPU copy, even when the GPU is out of memory or a forward fails.
        for module, key, parameter in self._parameters:
            module._parameters[key] = parameter
        for module, key, buffer in self._buffers:
            module._buffers[key] = buffer
        self._active = False


def _nonblock_modules(model: nn.Module, blocks: nn.ModuleList) -> Iterator[nn.Module]:
    yield model
    for child in model.children():
        if child is not blocks:
            yield from child.modules()


class LTXBlockOffload:
    """Attach exception-safe hooks to an LTX DiT; keep every block on CPU at rest.

    The pre/post hooks are deliberately synchronous and one-block-at-a-time.
    No CUDA stream, prefetch threads, independent cache or device guessing.
    The caller registers a tiny resident marker as ``dit`` with MemoryManager
    so that its normal stage handling never uploads the entire 22B module.
    """

    def __init__(self, model: nn.Module, onload_device: str | torch.device,
                 offload_device: str | torch.device):
        self._model = model
        self._onload = torch.device(onload_device)
        self._offload = torch.device(offload_device)
        if self._onload == self._offload:
            raise ValueError("block offloading needs different onload/offload devices")
        if self._offload.type != "cpu":
            raise ValueError("LTX block offloading currently needs CPU master weights")
        blocks = getattr(model, "transformer_blocks", None)
        if not isinstance(blocks, nn.ModuleList) or not blocks:
            raise ValueError("expected nonempty LTX transformer_blocks ModuleList")
        # Enforce the invariant before installing a single hook: no CUDA master
        # should be retained inadvertently for a 22B model.
        for tensor in (*model.parameters(), *model.buffers()):
            if tensor is not None and tensor.device != self._offload:
                raise ValueError("LTX block offload requires weights resident on CPU before activation")
        self._root = _Weights(_nonblock_modules(model, blocks))
        self._blocks = [_Weights(block.modules()) for block in blocks]
        self._handles: list[Any] = []
        try:
            self._handles.append(model.register_forward_pre_hook(self._enter))
            self._handles.append(model.register_forward_hook(self._leave, always_call=True))
            for block, weights in zip(blocks, self._blocks, strict=True):
                # Bind weight group at registration time to avoid loop-closure drift.
                self._handles.append(block.register_forward_pre_hook(
                    lambda _module, _args, w=weights: w.upload(self._onload)))
                self._handles.append(block.register_forward_hook(
                    lambda _module, _args, _out, w=weights: w.restore(), always_call=True))
        except BaseException:
            self.close()
            raise

    def _enter(self, _module: nn.Module, _args: tuple) -> None:
        self._root.upload(self._onload)

    def _leave(self, _module: nn.Module, _args: tuple, _output: Any) -> None:
        # On a failed nested block, its always_call cleanup runs first. Restore
        # any partially active block defensively before the root as well.
        for weights in self._blocks:
            weights.restore()
        self._root.restore()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for weights in self._blocks:
            weights.restore()
        self._root.restore()


__all__ = ["LTXBlockOffload"]
