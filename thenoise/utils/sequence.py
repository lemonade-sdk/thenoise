"""Shared batch-sequence preparation for variable-length token streams.

Z-Image, Krea 2 and Qwen-Image all feed variable-length text/image token
sequences into attention. Each implements a variant of the same three steps:

  1. trim invalid tokens, or fill a model-level pad slot with something of its own
     (``pad_to_length``),
  2. right-pad the batch to the longest item (``pad_to_batch``), optionally after
     having first padded each sequence to a multiple of a constant, for stable
     compiled-kernel shapes,
  3. build a ``[B, L]`` validity mask (``True = attend``) for SDPA key padding.

This module centralizes the padding and mask math. The genuinely model-specific
parts (pad-to-multiple scope, RoPE batch splitting) stay in the models.

Padding a sequence to a multiple of a constant leaves slots that hold no token, and
the two families disagree about what happens to them: Z-Image fills them with a
learned token that attention *does* read (``make_key_padding_mask`` -> a valid
prefix over the padded length), Ming-Image zero-fills them and masks them out
(``alignment_padding_mask`` -> holes inside the sequence).
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
) -> tuple[Tensor, Optional[Tensor], list[int]]:
    """Right-pad a batch of variable-length sequences to the batch max.

    ``seqs``        : list of ``[L_i, ...]`` per-sample tensors.
    ``positions``   : optional list of ``[L_i, n_axes]`` position ids, padded in
                      lockstep with ``seqs``.

    New slots are zero-filled: filling a pad slot with something else (a learned
    embedding, say) is per-item work, and belongs in :func:`pad_to_length`, which
    runs BEFORE this — so the model knows which slots are its own alignment padding
    and which are only here because another item in the batch is longer.

    Returns ``(feats, positions, item_seqlens)``: ``feats`` is
    ``[B, L_max, ...]``, ``positions`` is ``[B, L_max, n_axes]`` (or ``None``),
    and ``item_seqlens`` are the lengths given in, used to build the validity mask.
    """
    item_seqlens = [len(f) for f in seqs]

    feats = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)
    if positions is not None:
        positions = torch.nn.utils.rnn.pad_sequence(positions, batch_first=True, padding_value=0.0)
        positions = positions[:, : feats.shape[1]]
    return feats, positions, item_seqlens


def pad_to_length(
    seqs: list[Tensor],
    lengths: list[int],
    *,
    pad_token: Optional[Tensor] = None,
) -> list[Tensor]:
    """Pad every ``[L_i, ...]`` sequence out to ``lengths[i]`` (never trims).

    ``pad_token`` (a ``[1, ...]`` learned embedding) fills the new slots when given;
    omitted, the slots are zeros. Those are the two alignment-padding conventions of
    the Lumina models: Z-Image fills its pad-to-a-multiple slots with a learned token
    and attends to them, Ming-Image fills them with zeros and masks them out.

    Unlike :func:`pad_to_batch` this grows each sequence to its OWN length rather than
    to the batch max, which is what makes the alignment pads exist before the batch
    step (and therefore be describable by the ``(valid, padded)`` length pairs the
    caller hands to :func:`alignment_padding_mask`).
    """
    out = []
    for seq, length in zip(seqs, lengths):
        pad = length - len(seq)
        if pad < 0:
            raise ValueError(f"cannot pad a {len(seq)}-token sequence to {length}")
        if pad == 0:
            out.append(seq)
            continue
        if pad_token is None:
            filler = torch.zeros((pad,) + tuple(seq.shape[1:]), dtype=seq.dtype, device=seq.device)
        else:
            # ``pad_token`` is a single ``[1, ...]`` embedding; tile it to ``pad`` rows.
            filler = pad_token.to(seq).repeat((pad,) + (1,) * (seq.dim() - 1))
        out.append(torch.cat([seq, filler], dim=0))
    return out


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


def alignment_padding_mask(
    item_segments: list[list[tuple[int, int]]],
    device: torch.device,
    *,
    always: bool = False,
) -> Optional[Tensor]:
    """``[B, L]`` bool validity mask (``True = attend``) with holes at the alignment pads.

    ``item_segments[i]`` describes batch item ``i``'s token stream as the sequence of
    ``(valid, padded)`` length pairs that make it up, in stream order — one pair for a
    single stream (its image tokens, or its caption tokens), two for a unified
    ``[image, caption]`` stream. Each segment contributes ``valid`` attended tokens
    followed by ``padded - valid`` alignment pads that are NOT attended, so the mask
    has holes INSIDE the sequence rather than a single valid prefix.

    This is Ming-Image's ``zero_masked`` convention (vendor ``mask_out_alignment_padding``
    / ComfyUI's ``masked_pad_multiple``): padding the stream to a multiple of 32 keeps
    the kernel shapes tidy, but a pad slot holds a zero, not a token, so attention has
    to skip it. Z-Image pads with a LEARNED token and attends to it, so it stays on
    :func:`make_key_padding_mask` (whose semantics — a valid prefix of the padded
    length — this reduces to when no segment is padded).

    Returns ``None`` when nothing would be masked (every item already the batch max
    with no padded segment), so the caller keeps the mask-free fast path.
    """
    lengths = [sum(padded for _, padded in segs) for segs in item_segments]
    max_len = max(lengths)
    if not always and all(
        length == max_len and all(valid == padded for valid, padded in segs)
        for length, segs in zip(lengths, item_segments)
    ):
        return None

    mask = torch.zeros((len(item_segments), max_len), dtype=torch.bool, device=device)
    for row, segs in enumerate(item_segments):
        start = 0
        for valid, padded in segs:
            if valid:
                mask[row, start : start + valid] = True
            start += padded
    return mask


__all__ = [
    "pad_len_to_multiple",
    "pad_to_batch",
    "pad_to_length",
    "make_key_padding_mask",
    "alignment_padding_mask",
]
