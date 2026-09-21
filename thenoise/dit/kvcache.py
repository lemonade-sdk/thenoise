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
Inductor's tiling analysis cannot handle. It also fixes the token layout a model
must use: the frozen slice has to be contiguous with the tokens a step writes, so
a model attending ``text, target, references`` (Flux.2 Klein, Qwen-Image) keeps the
recomputed tokens at the front of the buffer and one attending ``text + references,
target`` (Qwen-Image 2.1) at the back — ``KVBuffers.cached_slice`` picks the end, and
the fill step writes the whole buffer either way.

Which tokens form the cached slice, how the modulation is made step-independent,
and whether the cache is exact (causal prefix) or approximate (bidirectional
attention) are all *model* concerns and deliberately live outside this module.
Everything a model needs to implement them is here, so the fiddly parts are shared
rather than re-derived per architecture: :func:`cache_mode` (fill / read / off,
decided once per forward), :meth:`KVCache.buffers` (a block's buffers for that
mode, with the token axis already declared dynamic) and :func:`attend` (the prefix
refresh plus the SDPA — the ``cat``-free tail every cached block ends with).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

import torch

from thenoise.utils.attention import attention as sdpa_attention
from thenoise.utils.dynamo import mark_token_axis

__all__ = ["KVBuffers", "KVCache", "attend", "cache_mode"]


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
    cached_slice: str = "suffix"

    @property
    def capacity(self) -> int:
        return self.k.shape[2]

    def write_offset(self, n_tokens: int) -> int:
        """Where a step's ``n_tokens`` recomputed tokens land (0 for a whole-buffer fill)."""
        return self.capacity - n_tokens if self.cached_slice == "prefix" else 0

    def write(self, k: torch.Tensor, v: torch.Tensor, offset: int = 0) -> None:
        """Copy this step's ``k``/``v`` into the buffer at ``offset``, leaving the rest."""
        self.k.narrow(2, offset, k.shape[2]).copy_(k)
        self.v.narrow(2, offset, v.shape[2]).copy_(v)


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

    def __init__(self, label: str = "", cached_slice: str = "suffix"):
        if cached_slice not in ("suffix", "prefix"):
            raise ValueError(
                f"unknown cached_slice {cached_slice!r}; expected 'suffix' or 'prefix'"
            )
        self._label = label
        self.cached_slice = cached_slice
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
                            torch.empty(shape, dtype=dtype, device=device),
                            self.cached_slice)
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

    def buffers(self, key: Hashable, mode: str, shape: tuple[int, ...],
                dtype: torch.dtype, device: torch.device) -> KVBuffers:
        """``key``'s buffers for one block, for this forward's ``mode`` (see :func:`cache_mode`).

        ``fill`` allocates (or reuses) at the full sequence length, ``read`` looks
        the buffers up and checks they have room for the tokens this step writes.
        Either way the token axis is declared dynamic here, so one kernel covers
        every token count the block is later replayed with.
        """
        got = self.allocate(key, shape, dtype, device) if mode == "fill" else self.get(key, shape[2])
        mark_token_axis(got.k, got.v, dim=2)
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


def cache_mode(kv: Optional[KVCache], refs_present: bool) -> str:
    """``"fill"`` / ``"read"`` / ``"off"`` for one forward; call it once, before the blocks.

    Deciding here rather than inside the loop means a mid-loop fill can never flip
    the mode under the loop's feet. ``off`` is the plain (uncached) path: either
    there is no cache, or a run started without reference tokens so there is
    nothing to freeze.
    """
    if kv is None:
        return "off"
    if refs_present:
        return "read" if kv.filled else "fill"
    return "read" if kv.filled else "off"


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           bufs: Optional[KVBuffers] = None) -> torch.Tensor:
    """SDPA over ``q``/``k``/``v``, refreshed through ``bufs`` when the run has a cache.

    ``bufs`` are this block's run-long buffers (``[B, H, L_total, D]``, same layout
    as the arguments): the step's ``L`` tokens are copied into their own positions
    (see :meth:`KVBuffers.write_offset`) and the attention then reads the *whole*
    buffer, so the frozen slice survives the copy while the queries stay at ``L``.
    On the fill step the copy covers the entire buffer, references included, so
    filling and reading are literally the same two copies — no mode flag, no branch,
    and above all no ``torch.cat``: a concatenation whose split point is a dynamic
    token count is what Inductor's tiling analysis chokes on. ``bufs=None`` is the
    plain path.

    Returns the attention output for the *queries*, token-major ``[B, L, H*D]``.
    """
    if bufs is not None:
        bufs.write(k, v, bufs.write_offset(k.shape[2]))
        k, v = bufs.k, bufs.v
    return sdpa_attention([q, k, v])
