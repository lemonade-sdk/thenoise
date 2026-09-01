"""thenoise: diffusion engine targeting ROCm.

This is deliberately a *focused engine* (a few models, a small explicit API surface),
not a full framework like ComfyUI. The compute backend is PyTorch on ROCm; the model
implementations live in this package (``thenoise.dit.*``), shared model components in
``thenoise.vae`` / ``thenoise.utils``.
"""
from __future__ import annotations

__version__ = "0.5.0"
