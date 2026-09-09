"""Shared batch-sequence preparation for variable-length token streams.

Z-Image, Krea 2 and Qwen-Image all feed variable-length text/image token
sequences into attention. Each implements a variant of the same three steps:

  1. trim invalid tokens (or replace them with a learned pad token),
  2. right-pad the batch to a fixed length (optionally first padding each
     sequence to a multiple of a constant, for stable compiled-kernel shapes),
  3. build a ``[B, L]`` validity mask (``True = attend``) for SDPA key padding.

This module centralizes the batch-padding and mask math. The genuinely
model-specific parts (trim-vs-replace, pad-to-multiple scope, RoPE batch
splitting) stay in the models.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def pad_len_to_multiple(length: int, multiple: int) -> int:
    """Next multiple of ``multiple`` at or above ``length``."""
    return ((length + multiple - 1) // multiple) * multiple


def pad_to_batch(
    seqs: list[Tensor],
    positions: Optional[list[Tensor]] = None,
    *,
    pad_value: float = 0.0,
    pad_token: Optional[Tensor] = None,
    replace_mask: Optional[list[Tensor]] = None,
) -> tuple[Tensor, Optional[Tensor], list[int]]:
    """Right-pad a batch of variable-length sequences to the batch max.

    ``seqs``        : list of ``[L_i, ...]`` per-sample tensors.
    ``positions``   : optional list of ``[L_i, n_axes]`` position ids, padded in
                      lockstep with ``seqs``.
    ``pad_token``   : if given (with ``replace_mask``), every position marked
                      ``True`` in ``replace_mask[i]`` is swapped for
                      ``pad_token`` (Z-Image's learned padding tokens). When
                      omitted, pad positions are zero-filled (the trim path).
    ``replace_mask``: list of ``[L_i]`` bool, ``True = pad``. Ignored unless
                      ``pad_token`` is set.

    Returns ``(feats, positions, item_seqlens)``: ``feats`` is
    ``[B, L_max, ...]``, ``positions`` is ``[B, L_max, n_axes]`` (or ``None``),
    and ``item_seqlens`` are the padded lengths used to build the validity mask.
    """
    item_seqlens = [len(f) for f in seqs]

    if pad_token is not None:
        # Replace pad positions (True in ``replace_mask``) with the pad token.
        # Only Z-Image uses this path (learned padding tokens); the trim path
        # (krea2/qwen) passes no pad_token, so pad positions stay zero.
        feats = [
            torch.where(m.unsqueeze(-1), pad_token.to(f), f)
            for f, m in zip(seqs, replace_mask)
        ]
    else:
        feats = seqs

    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=pad_value)
    if positions is not None:
        positions = torch.nn.utils.rnn.pad_sequence(positions, batch_first=True, padding_value=0.0)
        positions = positions[:, : feats.shape[1]]
    return feats, positions, item_seqlens


def make_key_padding_mask(
    item_seqlens: list[int], device: torch.device, *, always: bool = False
) -> Optional[Tensor]:
    """``[B, L]`` bool validity mask (``True = attend``).

    Returns ``None`` when every sequence has the same length (the equal-length
    fast path, used by Z-Image whose blocks accept ``None``). Pass ``always=True``
    when the caller needs a real tensor even for uniform lengths (Krea 2 and
    Qwen-Image consume the mask downstream, e.g. ``torch.cat`` / ``mask.sum()``).
    """
    max_seqlen = max(item_seqlens)
    if not always and all(seq == max_seqlen for seq in item_seqlens):
        return None
    mask = torch.zeros((len(item_seqlens), max_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(item_seqlens):
        mask[i, :seq_len] = 1
    return mask


__all__ = ["pad_len_to_multiple", "pad_to_batch", "make_key_padding_mask"]
