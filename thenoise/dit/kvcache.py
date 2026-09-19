"""Shared step-invariant prefix KV cache for editing diffusion models.

Many editing DiTs (Flux.2 Klein, Qwen-Image 2.1, ...) append a set of *reference*
tokens whose AdaLN modulation is made step-independent (the ``index_timestep_zero``
/ ``zero_cond_t`` trick). Their K/V in every attention block then only change
through attention over the *changing* image tokens, so freezing them for the rest
of a sampling run is a small, controllable approximation — this is exactly what
ComfyUI's ``FluxKVCache`` node does. The cache is filled on the first denoise step
(which still runs the full sequence, so it is exact) and read on every later step.

This module holds only the *store* and the fill/read protocol:

  * ``KVSegment`` — one attention block's cached prefix K/V (post-RoPE).
  * ``KVCache``   — fill-once / read-many per-block store, scoped to one run.

Which tokens form the cached slice, how the modulation is made step-independent,
and whether the cache is exact (causal prefix) or approximate (bidirectional
attention) are all *model* concerns and deliberately live outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

import torch

__all__ = ["KVSegment", "KVCache"]


@dataclass(frozen=True)
class KVSegment:
    """One attention block's cached prefix K/V, post-RoPE.

    Shapes are ``[B, H, L, D]`` (the layout this engine's SDPA helper consumes),
    matching the model's attention projection exactly. The cache stores *keys as
    attention sees them* (after RoPE), so a future quantized / CPU-staged cache
    does not need to re-apply rope on read.
    """

    k: torch.Tensor
    v: torch.Tensor


class KVCache:
    """Fill-once / read-many prefix K/V store for one whole DiT, keyed per block.

    Lifetime: one sampling run. The owning model adapter creates a fresh cache in
    ``prepare_latent`` (one per conditioning branch, so CFG never mixes keys) and
    drops it in ``finalize_latent``.

    The ``filled`` flag is set by the model after a fill pass (``set_filled``) and
    read once, *before* the block loop, to choose fill vs read mode — so a mid-loop
    fill can never flip the mode under the loop's feet.
    """

    def __init__(self, label: str = ""):
        self._label = label
        self._segments: dict[Hashable, KVSegment] = {}
        self._filled = False

    @property
    def label(self) -> str:
        return self._label

    @property
    def filled(self) -> bool:
        """True once the cache has been filled for a full run (all blocks)."""
        return self._filled

    def segment(self, key: Hashable) -> Optional[KVSegment]:
        """The cached K/V for ``key``, or ``None`` if this block was not stored."""
        return self._segments.get(key)

    def store(self, key: Hashable, seg: KVSegment) -> None:
        """Record ``key``'s cached K/V (caller must detach/clone before storing)."""
        self._segments[key] = seg

    def set_filled(self) -> None:
        self._filled = True

    def clear(self) -> None:
        self._segments.clear()
        self._filled = False

    def nbytes(self) -> int:
        """Total cached bytes across all blocks (k + v)."""
        total = 0
        for seg in self._segments.values():
            total += seg.k.numel() * seg.k.element_size()
            total += seg.v.numel() * seg.v.element_size()
        return total
