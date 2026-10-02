"""Process-wide inference policy: the grad-mode boundary, frozen weights, the lock.

The engine never trains. This module states that once, so no kernel, adapter,
loader or postprocessing helper has to declare it for itself.

Two orthogonal knobs, one primitive each:

* ``inference()`` — **grad mode**. Wrapped work builds no autograd graph and
  produces inference tensors. This is the engine's only grad mechanism.
* ``freeze(module)`` — **module/weight state**: ``eval()`` behaviour (Dropout,
  BatchNorm running stats) plus ``requires_grad_(False)`` weights, declared once at
  load time.

The mode is set at exactly three boundaries, one per public entry point:
``PipelineController.generate`` / ``.edit``, ``PixelUpscaleController.upscale`` and
``Runtime.load``. Anything else runs inside one of them and inherits the mode —
including the pipeline cache, whose inference tensors are written and read inside
the same boundary. The two helpers that mutate weights in place (``utils.lora``,
``upscale.sesqui_net``) open their own boundary.
"""
from __future__ import annotations

import threading
from typing import Any, TypeVar

import torch
from torch import nn

# One process-wide lock shared by every inference entry point. The generate
# pipeline and the standalone pixel upscale controller mutate the same on-device
# upscaler pool and model state, so the lock must be process-global.
inference_lock = threading.Lock()


def inference():
    """The engine's only grad-mode primitive (``torch.inference_mode``).

    Use as ``with inference():`` at the three entry points listed in the module
    docstring, or as ``@inference()`` on a function that mutates weights in place.
    Beyond ``no_grad`` it also drops the autograd version counters, so a tensor
    produced here cannot be smuggled into a graph later.

    The single indirection keeps the mechanism switchable in one place.
    """
    return torch.inference_mode()


M = TypeVar("M", bound=Any)


def freeze(module: M) -> M:
    """Put ``module`` in eval mode with frozen weights, returning it.

    The engine's only load-time primitive. Accepts the plain wrappers the memory
    manager also takes by freezing the ``nn.Module`` attributes they hold.
    """
    if isinstance(module, nn.Module):
        module.eval().requires_grad_(False)
        return module
    for value in getattr(module, "__dict__", {}).values():
        if isinstance(value, nn.Module):
            value.eval().requires_grad_(False)
    return module


__all__ = ["inference", "freeze", "inference_lock"]
