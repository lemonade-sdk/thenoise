"""Shared step-invariant prefix KV cache for editing diffusion models.

Many editing DiTs (Flux.2 Klein, Qwen-Image 2.1, ...) append a set of *reference*
tokens whose AdaLN modulation is made step-independent (the ``index_timestep_zero``
/ ``zero_cond_t`` trick). Their K/V in every attention block then only change
through attention over the *changing* image tokens, so freezing them for the rest
of a sampling run is a small, controllable approximation — this is exactly what
ComfyUI's ``FluxKVCache`` node does. The cache is filled on the first denoise step
(which still runs the full sequence, so it is exact) and read on every later step.

The storage is one pair of preallocated K/V buffers per attention block, sized to
the block's full sequence length for the whole run. Each step rewrites only the
leading tokens it recomputed, leaving the cached reference suffix in place; that is
the ``KVCache.update`` pattern from ``torchtune``, chosen over concatenating a
cached prefix because it keeps ``torch.cat`` out of the compiled blocks entirely —
a concatenation whose split point depends on the (dynamic) token count is something
Inductor's tiling analysis cannot handle.

Which tokens form the cached slice, how the modulation is made step-independent,
and whether the cache is exact (causal prefix) or approximate (bidirectional
attention) are all *model* concerns and deliberately live outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

import torch

__all__ = ["KVBuffers", "KVCache"]


@dataclass(frozen=True)
class KVBuffers:
    """One attention block's K/V buffers for a run: ``[B, H, L_total, D]``.

    The layout is exactly what the model's attention projection produces (post-RoPE
    keys), so a future quantized / CPU-staged cache does not need to re-apply rope.
    ``capacity`` is ``L_total``, fixed when the cache is filled: later steps write
    fewer tokens than the buffer holds, and the tail stays cached.
    """

    k: torch.Tensor
    v: torch.Tensor

    @property
    def capacity(self) -> int:
        return self.k.shape[2]


class KVCache:
    """Per-block K/V buffer store for one whole DiT, scoped to one sampling run.

    Lifetime: one run. The owning model adapter creates a fresh cache in
    ``prepare_latent`` (one per conditioning branch, so CFG never mixes keys — the
    branches can have different token counts) and drops it in ``finalize_latent``.

    The ``filled`` flag is set by the model after a fill pass (``set_filled``) and
    read once, *before* the block loop, to choose fill vs read — so a mid-loop fill
    can never flip the mode under the loop's feet. Note the *write* itself needs no
    mode: a fill writes the whole buffer, a read writes its leading prefix.
    """

    def __init__(self, label: str = ""):
        self._label = label
        self._buffers: dict[Hashable, KVBuffers] = {}
        self._filled = False

    @property
    def label(self) -> str:
        return self._label

    @property
    def filled(self) -> bool:
        """True once the cache has been filled for a full run (all blocks)."""
        return self._filled

    def allocate(self, key: Hashable, shape: tuple[int, ...], dtype: torch.dtype,
                 device: torch.device) -> KVBuffers:
        """Create (or reuse) ``key``'s buffers at their final length; fill step only.

        Every position is written by the incoming K/V on the fill step, so the
        buffers start uninitialized rather than paying a zero-fill of the whole run.
        """
        got = self._buffers.get(key)
        if got is None or got.capacity != shape[2]:
            got = KVBuffers(torch.empty(shape, dtype=dtype, device=device),
                            torch.empty(shape, dtype=dtype, device=device))
            self._buffers[key] = got
        return got

    def get(self, key: Hashable, need: int) -> KVBuffers:
        """``key``'s buffers, which must have room for the ``need`` tokens being written.

        Raises rather than letting a too-small buffer be overrun by the copy in the
        attention block (which would corrupt memory silently).
        """
        got = self._buffers.get(key)
        if got is None:
            raise KeyError(f"{self._label or 'kv'} cache has no buffers for {key!r};"
                           " a run must fill the cache before reading it")
        if got.capacity < need:
            raise RuntimeError(f"{self._label or 'kv'} cache buffer for {key!r} holds "
                               f"{got.capacity} tokens but {need} are being written;"
                               " the cache was filled for a different sequence length")
        return got

    def set_filled(self) -> None:
        self._filled = True

    def clear(self) -> None:
        self._buffers.clear()
        self._filled = False

    def nbytes(self) -> int:
        """Total cached bytes across all blocks (k + v)."""
        return sum(buf.k.numel() * buf.k.element_size() + buf.v.numel() * buf.v.element_size()
                   for buf in self._buffers.values())
