#!/usr/bin/env python
"""Does a VAE run faster in channels_last? Layout is a first-class cost on ROCm.

The DiT learned this lesson with attention: the fused SDPA kernels pick their path from
the input strides, and a packed ``[B, L, H, D]`` straggler cost an order of magnitude
(see ``utils.attention.uniform_layout``). Conv has the same trap from the other side —
MIOpen's kernels are NHWC-native, so handed NCHW it wraps them in layout conversions.
``vae_kernels.py`` caught that happening: ``batched_transpose_*`` around Qwen-Image's
convs (≈35% of its conv time) and ``Im2d2Col_v2`` staging a 9x-expanded copy for the
Flux AE's GEMM-path convs (≈30% of its conv time and most of its peak memory).

This probe runs the same case in both layouts in one process and prints, per codec/op:

* milliseconds, GPU time and achieved TFLOPS in each layout, and the category table
  underneath — conv, matmul, norm, activation, elementwise, resample, copies — so a win
  that is really GroupNorm getting slower in NHWC cannot hide behind the headline;
* how much of the conv time is *staging* — kernels whose name says transpose or im2col,
  i.e. MIOpen moving the activations instead of computing on them — and the conv TFLOPS
  once that is subtracted, which is the number to compare against the ~28 TFLOPS the
  ``features`` rows prove this machine can do;
* peak allocated/reserved, since an im2col buffer is nine times its input;
* the largest relative difference between the two outputs, because a layout change that
  silently changes pixels is not a layout change.

Layout is applied by wrapping the bench's own helpers (weights, pixels and latents are
built in the requested format), so the codec itself is untouched — what is measured is
what the engine would get if it built its codec ``channels_last``.

``MIOPEN_FIND_MODE`` belongs in this experiment too and cannot be set from here (MIOpen
reads it when the ROCm libraries load), so compare across processes:

    MIOPEN_FIND_MODE=1 .venv/bin/python bench-scripts/probes/vae_layout.py --res 1024x1024
    .venv/bin/python bench-scripts/diff_bench.py a.json b.json    # for the bench itself

Usage
-----
    .venv/bin/python bench-scripts/probes/vae_layout.py --res 1024x1024 --models qwen_image,flux2
    .venv/bin/python bench-scripts/probes/vae_layout.py --res 2048x2048 --models flux2 --kernels
    .venv/bin/python bench-scripts/probes/vae_layout.py --models qwen_image --ops encode,decode
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import vae_bench as vb  # noqa: E402
from vae_kernels import category, op_key  # noqa: E402  (same join rules as the op probe)

# Kernels whose only job is to move activations. MIOpen wraps its NHWC convs in
# transposes and stages a widened im2col copy for GEMM-path convs; both are billed
# inside ``aten::convolution``'s device time, so they have to come back out before
# anyone calls the remainder a conv's speed. ``Im2d2Col`` is spelled with a 2d in it, and
# a regex that says only ``im2col`` matches none of it — 28% of a flux decode, missed.
CONV_STAGE = re.compile(r"transpose|im2col|im2d2col|col2im|nhwc|nchw|memcpy",
                        re.IGNORECASE)


def describe(t) -> str:
    """Which memory format an output actually ended up in (there is no attribute)."""
    if not isinstance(t, torch.Tensor) or t.dim() != 4:
        return "-"
    if t.is_contiguous(memory_format=torch.channels_last):
        return "nhwc"
    if t.is_contiguous():
        return "nchw"
    return f"strided{tuple(t.stride())}"


class Layout:
    """Build bench cases in a chosen memory format by wrapping the bench's own helpers."""

    def __init__(self, channels_last: bool):
        self.mf = torch.channels_last if channels_last else torch.contiguous_format

    def __enter__(self):
        self.new_vae, self.pixels, self.randn = vb.new_vae, vb.rand_pixels, vb.randn
        mf = self.mf

        def new_vae(ctx, name, cls, *a, **kw):
            vae = self.new_vae(ctx, name, cls, *a, **kw)
            if mf != torch.channels_last:
                return vae
            try:
                return vae.to(memory_format=mf)
            except RuntimeError as exc:          # a rank-5 parameter: no channels_last
                raise SystemExit(f"{name} cannot be built channels_last: {exc}") from exc

        def pixels(ctx, res, channels):
            return self.pixels(ctx, res, channels).contiguous(memory_format=mf)

        def randn(ctx, *shape):
            return self.randn(ctx, *shape).contiguous(memory_format=mf)

        vb.new_vae, vb.rand_pixels, vb.randn = new_vae, pixels, randn
        return self

    def __exit__(self, *exc):
        vb.new_vae, vb.rand_pixels, vb.randn = self.new_vae, self.pixels, self.randn
        return False


def build(fx, op, res, device, dtype):
    """The case, with its VAE detached so the caller can free things in order."""
    captured: dict = {}
    base = vb.vae_case

    def patched(ctx, vae, name, r, o, note=""):
        captured["vae"] = vae
        return base(ctx, vae, name, r, o, note)

    vb.vae_case = patched
    try:
        case = fx.build(vb.Ctx(device=device, dtype=dtype), res, op)
    finally:
        vb.vae_case = base
    return case, captured["vae"]


def run_case(fx, op, res, device, dtype, channels_last, groups, iters):
    """Time one layout, then split its GPU time by category and count its FLOPs."""
    gpu = device == "cuda" and torch.cuda.is_available()
    sync = torch.cuda.synchronize if gpu else (lambda: None)
    with Layout(channels_last):
        case, vae = build(fx, op, res, device, dtype)
        with torch.no_grad():
            for _ in range(3):                      # per-shape conv search, allocator
                out = case.run()
            out_layout, sample = describe(out), out.detach().float().cpu()
            sync()
            if gpu:
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(groups):
                t0 = time.perf_counter()
                for _ in range(iters):
                    case.run()
                sync()
                times.append((time.perf_counter() - t0) * 1000 / iters)
            times.sort()
            # Same protocol as the bench: peak over the timed calls, after warmup.
            peak = torch.cuda.max_memory_allocated() / 2**30 if gpu else 0.0
            peak_res = torch.cuda.max_memory_reserved() / 2**30 if gpu else 0.0
            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if gpu else [])
            with profile(activities=acts, with_stack=False) as prof:
                for _ in range(iters):
                    case.run()
                sync()
            with FlopCounterMode(display=False) as counter:
                case.run()
            sync()

    # Time per category from the aten rows (kineto bills a conv's kernel to
    # ``aten::convolution``), staging from the kernel names, which live *inside* those
    # rows, and FLOPs per category from the flop counter's own op names. Three
    # instruments, three spellings: normalise with ``op_key``/``category`` and join late.
    cat: dict[str, float] = {}
    stage = 0.0
    kernels: dict[str, list] = {}
    rows = prof.key_averages()
    # A CPU run records no device activity at all: fall back to CPU wall per aten op and
    # leave the staging column blank rather than inventing it.
    on_device = any(float(getattr(r, "self_device_time_total", 0.0)) > 0 for r in rows)
    for row in rows:                       # kernel rows: names, so staging lives here
        if row.key.startswith(("aten::", "cuda", "hip")):
            continue
        us = float(getattr(row, "self_device_time_total", 0.0))
        if us <= 0:
            continue
        seen = kernels.setdefault(row.key, [0.0, 0])
        seen[0] += us
        seen[1] += int(row.count)
        if CONV_STAGE.search(row.key):
            stage += us
    for row in rows:                       # aten rows: time per category
        if not row.key.startswith("aten::"):
            continue
        us = float(getattr(row, "self_device_time_total" if on_device
                           else "self_cpu_time_total", 0.0))
        if us <= 0:
            continue
        c = category(row.key)
        cat[c] = cat.get(c, 0.0) + us

    flops: dict[str, float] = {}
    for k, f in counter.get_flop_counts()["Global"].items():
        key = category(op_key(k))
        flops[key] = flops.get(key, 0.0) + float(f)

    del case, vae
    gc.collect()
    if gpu:
        torch.cuda.empty_cache()
    ms = times[len(times) // 2]
    return dict(ms=ms, spread=100 * (times[-1] - times[0]) / ms if ms else 0.0,
                gpu_ms=sum(cat.values()) / 1000, cat={k: v / 1000 for k, v in cat.items()},
                flops=flops, stage_ms=stage / 1000, kernels=kernels,
                tflops=sum(flops.values()) / (ms / 1000) / 1e12 if ms else 0.0,
                peak=peak, peak_res=peak_res, sample=sample, out=out_layout,
                on_device=on_device)


def rel_diff(a, b) -> tuple[float, float]:
    """(worst relative element, overall relative L2) between the two layouts.

    The max is the number to worry about (one bad pixel is a bad pixel); the L2 is the
    number to believe, because a bf16 layout change reorders accumulation and one
    outlier in a random-weight net says much less than the norm of the difference.
    """
    if a.shape != b.shape:
        return float("inf"), float("inf")
    scale = a.abs().max().item() + 1e-9
    return ((a - b).abs().max().item() / scale,
            (a - b).norm().item() / (a.norm().item() + 1e-9))


def conv_of(r) -> float:
    return r["cat"].get("conv", 0.0)


def top_kernels(r, n, pattern=None, width=34) -> str:
    rows = [(k, v) for k, v in r["kernels"].items() if pattern is None or pattern.search(k)]
    rows.sort(key=lambda kv: -kv[1][0])
    return "  ".join(f"{k.split('(')[0][:width]}:{v[0] / 1000:.1f}ms x{v[1]}"
                     for k, v in rows[:n]) or "-"


def report(label, res, a, b, show_kernels):
    worst, l2 = rel_diff(a["sample"], b["sample"])
    identical = worst == 0.0 and a["out"] == b["out"]
    print(f"\n{label}  {res}   output {a['out']} -> {b['out']}   output difference: "
          f"{worst:.2e} worst element, {l2:.2e} relative L2")
    if identical:
        print("  !! the two arms produced bit-identical output from bit-identical "
              "kernels: the layout\n  !! never reached the convs, so this is NOT a null "
              "result — it is no experiment at all.\n  !! (aten::miopen_convolution "
              "returns NCHW whatever it is handed, so on ROCm it cannot.)")
    print(f"  {'layout':14s} {'ms':>8s} {'±%':>5s} {'gpu ms':>8s} {'TFLOPS':>7s} "
          f"{'conv ms':>8s} {'conv TF':>8s} {'staged':>8s} {'%conv':>6s} "
          f"{'GiB':>6s} {'resv':>6s}")
    for name, r in (("nchw", a), ("channels_last", b)):
        conv = conv_of(r)
        net = max(conv - r["stage_ms"], 0.0)
        print(f"  {name:14s} {r['ms']:8.1f} {r['spread']:5.1f} {r['gpu_ms']:26.1f} "
              f"{r['tflops']:7.1f} {conv:8.1f} "
              f"{(r['flops'].get('conv', 0.0) / net / 1e9 if net else 0.0):8.1f} "
              f"{r['stage_ms']:8.1f} "
              f"{(100 * r['stage_ms'] / conv if conv else 0.0):5.1f}% "
              f"{r['peak']:6.2f} {r['peak_res']:6.2f}")
    gain = 100 * (1 - b["ms"] / a["ms"]) if a["ms"] else 0.0
    print(f"  channels_last is {gain:+.1f}% on wall time"
          f"  (peak {a['peak']:.2f} -> {b['peak']:.2f} GiB allocated)"
          + ("" if a["on_device"] else "\n  no device activity: this is a CPU run, so "
                                            "'gpu ms' is CPU wall per op and "
                                            "'staged' is blank"))

    cats = sorted(set(a["cat"]) | set(b["cat"]), key=lambda c: -max(a["cat"].get(c, 0.0),
                                                                    b["cat"].get(c, 0.0)))
    print(f"  {'where the time is':22s} {'nchw ms':>9s} {'%':>5s} {'TFLOPS':>7s} "
          f"{'nhwc ms':>9s} {'%':>5s} {'TFLOPS':>7s} {'Δ':>8s}")
    tot = {n: sum(r["cat"].values()) for n, r in (("a", a), ("b", b))}
    for c in cats:
        ma, mb = a["cat"].get(c, 0.0), b["cat"].get(c, 0.0)
        f = a["flops"].get(c, 0.0)
        g = b["flops"].get(c, 0.0)
        d = 100 * (mb / ma - 1) if ma else float("inf")
        print(f"  {c:22s} {ma:9.1f} {100 * ma / tot['a'] if tot['a'] else 0:5.1f} "
              f"{(f / (ma / 1000) / 1e12 if ma and f else 0.0):7.1f} "
              f"{mb:9.1f} {100 * mb / tot['b'] if tot['b'] else 0:5.1f} "
              f"{(g / (mb / 1000) / 1e12 if mb and g else 0.0):7.1f} "
              f"{(f'{d:+.0f}%' if ma else 'new'):>8s}")
        if c == "conv":       # kernel-named copies: mostly launched inside the conv, so
            print(f"  {'  staged copies':22s} {a['stage_ms']:9.1f} "    # they overlap it
                  f"{'':>5s} {'':>7s} {b['stage_ms']:9.1f}")
    for name, r in (("nchw", a), ("channels_last", b)):
        print(f"    {name:13s} staged: {top_kernels(r, 3, CONV_STAGE)}")
    if show_kernels:
        for name, r in (("nchw", a), ("channels_last", b)):
            print(f"    {name:13s} top:    {top_kernels(r, 5)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="NCHW vs channels_last for a VAE case.")
    ap.add_argument("--models", default="flux,qwen_image,qwen_image21,flux2")
    ap.add_argument("--ops", default="decode")
    ap.add_argument("--res", default="1024x1024")
    ap.add_argument("--dtype", default="bf16", choices=sorted(vb.DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--kernels", action="store_true", help="also list top kernels per layout")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no cuda/rocm device visible — run this on the GPU box")
    res = vb.Res.parse(args.res)
    ops = tuple(o.strip() for o in args.ops.split(",") if o.strip())
    fixtures = vb.select(args.models)
    dtype = vb.DTYPES[args.dtype]
    gpu = args.device == "cuda" and torch.cuda.is_available()
    print(f"{torch.cuda.get_device_name(0) if gpu else args.device} | "
          f"torch {torch.__version__} | {args.dtype} | {res} | "
          f"MIOPEN_FIND_MODE={os.environ.get('MIOPEN_FIND_MODE', 'default')}")

    for fx in fixtures:
        for op in fx.ops:
            if op not in ops:
                continue
            a = run_case(fx, op, res, args.device, dtype, False, args.groups, args.iters)
            b = run_case(fx, op, res, args.device, dtype, True, args.groups, args.iters)
            report(f"{fx.model}/{op}", res, a, b, args.kernels)
            del a, b
            gc.collect()

    print("\nconv TF is conv FLOPs over conv time *net of staging*: if it moves to the "
          "staging column\nrather than the conv column, the layout fixed the copies and "
          "nothing else. A layout change\nthat only wins on a codec whose output layout "
          "then has to be converted back for the\nsave/VAE-decode handoff has not won "
          "anything — check the printed output layout.")


if __name__ == "__main__":
    main()
