"""Kitchen-fused, strip-wise convolution for the AutoencoderKL-family VAEs.

Each ``GroupNorm -> SiLU -> 3x3 conv`` of a decode is one kernel —
``comfy_kitchen.group_norm_silu_pad3d`` normalises, activates and pads,
``fp16_conv3d`` folds the conv in accumulating FP32 — and the image is walked in strips of
rows with one halo row each side so no conv sees a synthetic edge.

GroupNorm is global, so a strip cannot normalise itself: ``group_norm_stats3d`` takes the
statistics once for the frame and every strip uses them. That is the only state a strip
needs from outside itself, and what makes a stripped decode match an unstripped one rather
than approximate it.

Strips let a residual block update its input in place, so a decoder allocates no other
full-size buffer: one, the size of its largest feature map, each level writing over the
front while reading the previous level from the back (:func:`region`).

1x1s go to :func:`pointwise` — a GEMM over the channel rows channels-last already lays
out — because MIOpen costs more for a channels-last 1x1 than that does. And
:func:`strip_apply` is the walk without the kernels: any chain of torch convolutions, given
only how many of them there are to read context for.

Opt-in per VAE (``fused=True``) and only where :func:`fused_available` agrees.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from comfy_kitchen import fp16_conv3d, group_norm_silu_pad3d, group_norm_stats3d
from comfy_kitchen.registry import registry

__all__ = [
    "STRIP_ELEMS",
    "FusedResnetMixin",
    "fused_available",
    "group_norm_only",
    "image_elems",
    "interleaved_view",
    "pointwise",
    "region",
    "strip_apply",
    "strip_conv",
    "strip_down",
]

#: Elements of one activation held per strip. A conv backend sizes its workspace from the
#: tensor it is handed, so this is what keeps that workspace off the peak.
STRIP_ELEMS = 2**24


def fused_available(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the fused path is worth taking here.

    FP16 only on ROCm: a BF16 conv there falls back to MIOpen, which runs a
    ``channels_last`` conv through NCHW and gives the gain back. CUDA keeps BF16, where
    cuDNN's NHWC conv is the faster one either way.
    """
    hip = torch.version.hip is not None
    backend = "hip" if hip else "cuda"
    if device.type != "cuda" or dtype not in ((torch.float16,) if hip else (torch.float16, torch.bfloat16)):
        return False
    # Without the kitchen conv an FP16 conv on ROCm is MIOpen's again.
    ops = ["group_norm_stats3d", "group_norm_silu_pad3d", *(("fp16_conv3d",) if hip else ())]
    return registry.is_available(backend) and all(
        registry.get_constraints(backend, op) is not None for op in ops
    )


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors can address the same bytes."""
    a0, b0 = a.data_ptr(), b.data_ptr()
    return a0 < b0 + b.numel() * b.element_size() and b0 < a0 + a.numel() * a.element_size()


def _strip_step(x: torch.Tensor, out_elems: Optional[int] = None) -> int:
    """Rows per strip, so neither ``x`` nor an output of ``out_elems`` elements is whole."""
    budget = x.numel() if out_elems is None else max(x.numel(), out_elems)
    return -(-x.shape[-2] // -(-budget // STRIP_ELEMS))


def _strip_rows(
    fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    out_elems: int,
    *,
    scale: int = 1,
    halo: int = 1,
    edge_padded: bool = True,
    layout: Optional[torch.memory_format] = None,
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """Walk ``x`` in strips of rows, ``fn(strip, top, bottom, dst)`` each, into one result.

    A strip is read ``halo`` rows beyond what it keeps and returns either ``dst`` — a kernel
    writing in place, which moves nothing — or the rows it computed. Its first kept row is
    ``scale * (a - lo) + top - halo``, ``top`` counting the zero rows standing in for context
    the image does not have: one at an image edge for a kernel that pads only there, all of
    them for a chain that pads itself and answers one row per row handed it.

    ``out`` may be the tail of the buffer ``x`` lives in, since strips write further forward
    than they read. When the two share rows, output is written one strip late — the next
    strip reads this one's last ``halo`` rows as its own halo — which costs the kernel its
    in-place write, so the deferral is taken only where they really overlap.
    """
    size = x.shape[-2]
    step = _strip_step(x, out_elems)
    if out is None and step >= size:  # one strip: no second buffer to stage the rows through
        edge = 1 if edge_padded else halo
        return fn(x, edge, edge, None)
    put = torch.Tensor.add_ if accumulate else torch.Tensor.copy_
    hold = halo if out is not None and _overlaps(out, x) else 0
    if hold:
        # A strip cannot hold back more rows than it made, and below the halo width the
        # next one reads rows this one had already written.
        step = max(step, hold)
    held: Optional[torch.Tensor] = None

    for a in range(0, size, step):
        b = min(size, a + step)
        lo, hi = max(0, a - halo), min(size, b + halo)
        top = int(lo == 0) if edge_padded else halo
        bottom = int(hi == size) if edge_padded else halo
        dst = None if hold or out is None else out.narrow(-2, scale * a, scale * (b - a))
        rows = fn(x.narrow(-2, lo, hi - lo), top, bottom, dst)
        if rows is dst:
            continue
        if out is None:
            # The caller states a layout because the wrong one costs an order of magnitude,
            # not a rounding; with none asked, the strip's own is the only honest answer.
            shape = (x.shape[0], rows.shape[1], size * scale, x.shape[-1] * scale)
            out = (
                rows.new_empty(shape)
                if layout is None
                else torch.empty(shape, dtype=rows.dtype, device=rows.device, memory_format=layout)
            )
        kept = rows.narrow(-2, scale * (a - lo) + top - halo, scale * (b - a))
        live = scale * max(0, b - a - hold)
        if held is not None:
            put(out.narrow(-2, scale * a - held.shape[-2], held.shape[-2]), held)
        put(out.narrow(-2, scale * a, live), kept.narrow(-2, 0, live))
        held = kept.narrow(-2, live, kept.shape[-2] - live).clone() if hold else None
    if held is not None:
        put(out.narrow(-2, size * scale - held.shape[-2], held.shape[-2]), held)
    return out


def image_elems(x: torch.Tensor, channels: int, scale: int = 1) -> int:
    """What an image like ``x`` would hold with ``channels`` and ``scale`` times the side."""
    return x.shape[0] * channels * x.shape[-2] * x.shape[-1] * scale * scale


def strip_apply(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    scale: int = 1,
    halo: int = 1,
    out_elems: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """``fn`` over whole rows of ``x``, in strips, assuming nothing about what ``fn`` is.

    ``halo`` is the number of 3x3 convolutions inside ``fn``, each padding by itself and so
    reading a synthetic edge at a strip's inner border; that many rows more than the strip
    keeps makes the kept rows what the whole image would have given. ``out_elems`` states the
    result's size when ``fn`` changes the channel count (:func:`image_elems`). ``out`` and
    ``accumulate`` are :func:`_strip_rows`', which is what makes ``out=x`` exact like any
    other use.
    """
    return _strip_rows(
        lambda rows, top, bottom, dst: fn(rows),
        x,
        x.numel() * scale * scale if out_elems is None else out_elems,
        scale=scale,
        halo=halo,
        edge_padded=False,
        out=out,
        accumulate=accumulate,
    )


def group_norm_only(x: torch.Tensor, norm: torch.nn.GroupNorm) -> torch.Tensor:
    """``norm(x)`` with no activation and no pad: the one stage wanting neither, which
    torch's GroupNorm does by reading a channels-last image strided.
    """
    return group_norm_silu_pad3d(
        x.unsqueeze(2), norm.weight, norm.bias, norm.num_groups, norm.eps,
        (0, 0, 0, 0, 0), False, zero_pad=True,
    ).squeeze(2)


def pointwise(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """A 1x1 conv as a GEMM over the channel row of a channels-last image.

    The channel rows are already the GEMM's rows, so neither end needs a copy; MIOpen takes
    a channels-last 1x1 the way it takes any conv, which costs far more than the matrix
    product it is.
    """
    b, c, h, w = x.shape
    y = F.linear(x.permute(0, 2, 3, 1).reshape(b, h * w, c), conv.weight.flatten(1), conv.bias)
    return y.view(b, h, w, -1).permute(0, 3, 1, 2)


def _conv_weight(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The 5D form of a 3x3 weight when the kitchen conv is to take it, else as it is.

    ROCm and channels-last only; that kernel reads NHWC, so a 5D weight doubles as the
    answer to "is the kitchen conv in charge here". Its CUDA twin accumulates in FP16 behind
    a knob this engine does not expose, so there cuDNN keeps the conv, and FP32 with it.
    """
    if (
        x.dtype is torch.float16
        and torch.version.hip is not None
        and x.is_contiguous(memory_format=torch.channels_last)
    ):
        return weight.unsqueeze(2).contiguous(memory_format=torch.channels_last_3d)
    return weight


def _padded_conv(
    padded: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dst: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The conv of a strip the fused kernel has already padded (a 5D weight says which).

    Given ``dst`` the kitchen conv writes its rows there — a strip that padded by reading a
    halo has exactly ``dst``'s rows left afterwards, so a level's buffer takes the write with
    no staging copy. That kernel declines a row-strided ``dst`` and one that also holds the
    residual to add, so callers hand it neither; a 2D weight has no hook and returns rows.
    """
    if weight.ndim == 5:
        if dst is not None:
            fp16_conv3d(padded, weight, bias, None, 1, out=dst.unsqueeze(2))
            return dst
        return fp16_conv3d(padded, weight, bias, None, 1).squeeze(2)
    return F.conv2d(padded.squeeze(2), weight, bias)


def strip_conv(
    conv: torch.nn.Conv2d,
    x: torch.Tensor,
    norm: Optional[torch.nn.GroupNorm] = None,
    *,
    scale: int = 1,
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """``conv(silu(norm(x)))`` in row strips, after a nearest upsample when ``scale > 1``.

    ``norm`` is ``None`` for the convolutions that stand alone — the upsampler's — which
    still want the strip driver and the fused pad. ``out`` and ``accumulate`` are
    :func:`_strip_rows`'; the result is channels-last, as every later fused call wants.
    """
    groups, eps = (1, 0.0) if norm is None else (norm.num_groups, norm.eps)
    gamma = None if norm is None else norm.weight
    beta = None if norm is None else norm.bias
    stats = None if gamma is None else group_norm_stats3d(x.unsqueeze(2), groups, eps)
    weight = _conv_weight(conv.weight, x)
    # Only a conv that owns its destination can write it: an upsample's has two rows
    # more than the strip keeps, and an accumulating block must add what it held.
    into_dst = scale == 1

    def one(rows: torch.Tensor, top: int, bottom: int, dst: torch.Tensor) -> torch.Tensor:
        if scale > 1:
            rows = F.interpolate(rows, scale_factor=scale, mode="nearest")
        padded = group_norm_silu_pad3d(
            rows.unsqueeze(2), gamma, beta, groups, eps,
            (1, 1, top, bottom, 0), gamma is not None, zero_pad=True, stats=stats,
        )
        return _padded_conv(padded, weight, conv.bias, dst if into_dst and not accumulate else None)

    kept = x.shape[0] * conv.out_channels * x.shape[2] * x.shape[3] * scale * scale
    return _strip_rows(one, x, kept, scale=scale, layout=torch.channels_last,
                       out=out, accumulate=accumulate)


def strip_down(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """A stride-2 conv with its ``(0, 1, 0, 1)`` pad, in strips.

    Output rows ``[a, b)`` read input rows ``[2a, 2b + 1)``, so a strip needs one input row
    more than it produces rather than a halo either side — and no norm, so no frame-wide
    statistics either. Only the image's own last row is short one, so a strip ending inside
    it must not pad at its bottom: the pads are told where the image ends.
    """
    rows = x.shape[2] // 2
    step = _strip_step(x, x.shape[0] * conv.out_channels * rows * x.shape[3])
    out = x.new_empty((x.shape[0], conv.out_channels, rows, x.shape[3] // 2))
    weight = _conv_weight(conv.weight, x)
    for a in range(0, rows, step):
        b = min(rows, a + step)
        hi = min(x.shape[2], 2 * b + 1)
        strip = x.narrow(2, 2 * a, hi - 2 * a)
        end = int(hi == x.shape[2])
        # Stride 2 spends the one input row read ahead of the strip, leaving exactly the
        # rows this strip of the output is.
        dst = out.narrow(2, a, b - a)
        if weight.ndim == 5:
            padded = group_norm_silu_pad3d(
                strip.unsqueeze(2), None, None, 1, 0.0,
                (0, 1, 0, end, 0), False, zero_pad=True,
            )
            fp16_conv3d(padded, weight, conv.bias, None, 2, out=dst.unsqueeze(2))
        else:
            dst.copy_(
                F.conv2d(F.pad(strip, (0, 1, 0, end)), conv.weight, conv.bias, stride=2)
            )
    return out


def region(buf: torch.Tensor, end: int, channels: int, h: int, w: int) -> torch.Tensor:
    """The channels-last ``[1, channels, h, w]`` tensor ending at element ``end`` of ``buf``.

    A decoder level takes its scratch from the front of what is left and its residual
    stream from the very end of one buffer, so no level allocates again.
    """
    n = channels * h * w
    return buf.narrow(0, end - n, n).view(1, h, w, channels).permute(0, 3, 1, 2)


def interleaved_view(x: torch.Tensor, channels: int, part: int) -> torch.Tensor:
    """``x``'s storage as two ``channels``-wide images alternating per row slot.

    ``part`` 0 takes the front half of every row slot and 1 the back; the pair covers ``x``
    exactly and each is a row-strided NHWC image. A block halving ``x``'s channels keeps its
    hidden state in one half and its result in the other, so it allocates nothing.
    """
    b, c, h, w = x.shape
    flat = x.permute(0, 2, 3, 1).reshape(-1)
    return flat.as_strided(
        (1, channels, h, w),
        (h * w * c, 1, w * c, channels),
        flat.storage_offset() + part * w * channels,
    )


class FusedResnetMixin:
    """The fused forward of a ``norm1 -> conv1 -> norm2 -> conv2`` residual block.

    Mixed into a block keeping the AutoencoderKL attribute names — ``norm1``/``norm2``
    affine GroupNorm, ``conv1``/``conv2`` 3x3 zero-padded, ``nin_shortcut`` the 1x1 that
    exists only when the widths differ — whose own ``forward`` stays the reference.
    """

    def halves_in_place(self, x: torch.Tensor) -> bool:
        """True when this block's ``C -> C/2`` reduction can run inside ``x`` itself."""
        return (
            x.shape[0] == 1
            and self.in_channels == 2 * self.out_channels
            and x.is_contiguous(memory_format=torch.channels_last)
        )

    def fused_forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x + conv2(silu(norm2(conv1(silu(norm1(x))))))``, with ``x`` consumed.

        ``hidden`` is the scratch the first conv writes to; a level hands every block the
        same region, since only one block's hidden state is live at a time.
        """
        if x.stride(1) != 1:  # NCHW, as it comes out of the mid-block attention
            x = x.contiguous(memory_format=torch.channels_last)
        if self.in_channels == self.out_channels:
            h = strip_conv(self.conv1, x, self.norm1, out=hidden)
            return strip_conv(self.conv2, h, self.norm2, out=x, accumulate=True)
        if self.halves_in_place(x):
            return self._halve_in_place(x)
        h = strip_conv(self.conv1, x, self.norm1)
        return strip_conv(self.conv2, h, self.norm2, out=pointwise(self.nin_shortcut, x),
                          accumulate=True)

    def _halve_in_place(self, x: torch.Tensor) -> torch.Tensor:
        """The ``C -> C/2`` block with hidden state and result sharing ``x``'s storage.

        Writing a strip of either half overwrites rows the next strip still reads as its
        halo, so a strip holds its last row back and the next one writes that row first.
        """
        h = interleaved_view(x, self.out_channels, 0)
        out = interleaved_view(x, self.out_channels, 1)
        stats = group_norm_stats3d(x.unsqueeze(2), self.norm1.num_groups, self.norm1.eps)
        gamma, beta = self.norm1.weight, self.norm1.bias
        weight = _conv_weight(self.conv1.weight, x)
        shortcut = self.nin_shortcut
        rows = x.shape[2]
        step = _strip_step(x)
        held: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        def normed(strip: torch.Tensor, top: int, bottom: int) -> torch.Tensor:
            return group_norm_silu_pad3d(
                strip.unsqueeze(2), gamma, beta, self.norm1.num_groups, self.norm1.eps,
                (1, 1, top, bottom, 0), True, zero_pad=True, stats=stats,
            )

        for a in range(0, rows, step):
            b = min(rows, a + step)
            lo, hi = max(0, a - 1), min(rows, b + 1)
            top = int(lo == 0)
            h_rows = _padded_conv(normed(x.narrow(2, lo, hi - lo), top, int(hi == rows)), weight, self.conv1.bias)
            # The conv's row offset, as in :func:`_strip_rows`.
            off = a - lo + top - 1
            out_rows = pointwise(shortcut, x.narrow(2, a, b - a))
            if held is not None:
                h.narrow(2, a - 1, 1).copy_(held[0])
                out.narrow(2, a - 1, 1).copy_(held[1])
            h.narrow(2, a, b - a - 1).copy_(h_rows.narrow(2, off, b - a - 1))
            out.narrow(2, a, b - a - 1).copy_(out_rows.narrow(2, 0, b - a - 1))
            held = (
                h_rows.narrow(2, off + b - a - 1, 1).clone(),
                out_rows.narrow(2, b - a - 1, 1).clone(),
            )
        h.narrow(2, rows - 1, 1).copy_(held[0])
        out.narrow(2, rows - 1, 1).copy_(held[1])
        return strip_conv(self.conv2, h, self.norm2, out=out, accumulate=True)
