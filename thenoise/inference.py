"""Process-wide inference policy: the grad-mode boundary, frozen weights, the lock.

The engine never trains. Nothing here is optional configuration: this module is
the single place that expresses that fact, so that no kernel, adapter, loader or
postprocessing helper has to declare it for itself.

Two orthogonal knobs, one primitive each
----------------------------------------
* ``inference()`` — **grad mode**. Everything it wraps builds no autograd graph
  and produces inference tensors. This is the only grad-mechanism in the engine;
  ``torch.no_grad`` is deliberately not used anywhere else (see the guard test in
  ``tests/test_inference_policy.py``).
* ``freeze(module)`` — **module/weight state**: ``eval()`` behaviour (Dropout,
  BatchNorm — e.g. the Flux.2 VAE's ``bn`` running stats) plus
  ``requires_grad_(False)`` weights. Not a grad-mode mechanism: ``inference()``
  leaves ``Parameter.requires_grad`` set, so a weight that must never be
  differentiable says so here, once, at load time.

Where the mode is set
---------------------
Exactly three boundaries, one per public entry point:

  * ``PipelineController.generate`` / ``.edit`` — every stage, the LoRA switch,
    the upscale-and-refine, and the whole post-decode tail (notch filter, pixel
    upscaler, resize, postprocess, PIL).
  * ``PixelUpscaleController.upscale`` — the model-free ``/upscale`` path.
  * ``Runtime.load`` — weight construction, inits and requantization.

Anything else that needs "no grad" is inside one of those and must NOT add its
own context. Two helpers are the documented exception, and both are in-place
*weight mutation* rather than an inference declaration: ``utils.lora``
(apply/undo) and ``upscale.sesqui_net`` (pixel-shuffle init), which are also
called directly from tests without a boundary. They use ``inference()`` too, so
the mechanism stays single-source.

Because every entry point sets the mode, the mode is inherited by everything the
engine does with tensors — including the pipeline cache, whose inference tensors
are written and read inside the same boundary.
"""
from __future__ import annotations

import threading
from typing import Any, TypeVar

import torch
from torch import nn

# One process-wide lock shared by every inference entry point.
#
# It must be process-global: the generate pipeline and the standalone pixel
# upscale controller are separate objects but they mutate the same on-device
# upscaler pool and model state, so a lock owned by either one alone would not
# prevent cross-controller races.
inference_lock = threading.Lock()


def inference():
    """The engine's only grad-mode primitive (``torch.inference_mode``).

    Use as ``with inference():`` at the three entry points listed in the module
    docstring, or as ``@inference()`` on a function that mutates weights in
    place. Stricter than ``torch.no_grad()``: it also drops the autograd version
    counters, so a tensor produced here cannot be smuggled into a graph later.

    One indirection on purpose — if a torch/ROCm build ever trips on inference
    mode (e.g. inside a ``torch.compile`` region), it is switched here and
    nowhere else.
    """
    return torch.inference_mode()


M = TypeVar("M", bound=Any)


def freeze(module: M) -> M:
    """Put ``module`` in eval mode with frozen weights, returning it.

    The engine's only load-time primitive. Called from ``MemoryManager.register``
    (which covers every adapter's ``dit`` / ``text_encoder`` / ``vae``) and at the
    exits of the upscaler loaders that the memory manager never sees.

    Accepts the plain wrappers the memory manager also takes (``Qwen3Embedder``)
    by freezing the ``nn.Module`` attributes they hold.
    """
    if isinstance(module, nn.Module):
        module.eval().requires_grad_(False)
        return module
    for value in getattr(module, "__dict__", {}).values():
        if isinstance(value, nn.Module):
            value.eval().requires_grad_(False)
    return module


__all__ = ["inference", "freeze", "inference_lock"]
