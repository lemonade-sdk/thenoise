#!/usr/bin/env python
"""Replay the attention calls a VAE decode really makes, once per input layout.

A VAE attention block is one head as wide as the channel count, over every pixel of one
stage's feature map, and the codecs hand ``single_head_attention`` different memory
formats to do it in: ``vae.flux`` writes ``.contiguous()`` on all three of q/k/v,
``vae.flux2`` hands over ``view(b, c, hw).transpose(1, 2)`` — the same arithmetic with
the channel axis outermost. Same N, same C, different strides, and the two cases are
not equally fast: ``flux2/decode`` spends 1.27x what ``flux/decode`` does in
``aten::bmm``. This records what every production codec actually passes to attention,
then replays each of those calls with the operands left as they were and with them
forced row-major, one at a time and then all together.

Each case is recorded twice, once NCHW and once ``channels_last``, because the layout of
a conv's output decides the layout of q/k/v: an NHWC ``1x1`` conv produces
``(pixel, channel)`` rows, which is exactly the form those two GEMMs like. So the layout
change ``vae_layout.py`` measures may fix this anomaly by itself — that is one of the
questions this answers.

Only the shapes and strides are recorded; the operands are re-rolled from them, so a
2048x2048 decode does not have to hold its own attention inputs in memory to be
replayed. ``TF`` is analytic — two matmuls are ``4 N^2 C`` FLOPs, softmax excluded — so
every column in a row is the same arithmetic and a faster column is a better kernel, not
less work. ``copies`` times the three ``.contiguous()`` calls alone: if ``all`` beats
``as-called`` by more than that, normalising in the codec is a win, and if not it is a
copy wearing a win.

Usage
-----
    .venv/bin/python bench-scripts/probes/vae_attn.py --models flux,flux2 --res 1024x1024
    .venv/bin/python bench-scripts/probes/vae_attn.py --res 2048x2048
    .venv/bin/python bench-scripts/probes/vae_attn.py --models qwen_image --groups 5
"""
from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import vae_bench as vb  # noqa: E402
from vae_layout import Layout, build  # noqa: E402  (same layout + case plumbing)

# The vae modules import the name itself, so a spy has to be installed in each of them.
ATTN_USERS = ("thenoise.vae.flux", "thenoise.vae.flux2", "thenoise.vae.qwen_image",
              "thenoise.vae.wan22", "thenoise.vae.mage_flow")
# Which operands to hand over already row-major: none, one of them, all of them.
VARIANTS = ("as-called", "q", "k", "v", "all")
WHICH = {"as-called": (), "q": (0,), "k": (1,), "v": (2,), "all": (0, 1, 2)}


def contig(shape) -> tuple[int, ...]:
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= s
    return tuple(reversed(out))


def same(a, b, shape) -> bool:
    """Do two stride patterns agree on every axis that has something to agree about?"""
    return all(x == y for x, y, s in zip(a, b, shape) if s != 1)


def code(shape, stride) -> str:
    """What one recorded operand looks like to a GEMM, from its last two axes.

    ``row``   packed ``(N, C)``: what ``vae.flux`` deliberately makes of q/k/v.
    ``row+s`` row-major rows inside a wider buffer — a qkv projection split three ways
              leaves each of q, k and v with a row stride of ``3C``.
    ``chan``  the ``(C, N)`` buffer transposed: what ``vae.flux2`` hands over, and what an
              NCHW ``1x1`` conv becomes once it is viewed as tokens.
    """
    if same(stride, contig(shape), shape):
        return "row"
    if stride[-1] == 1:
        return "row+s" if stride[-2] != shape[-1] else "row"
    if stride[-2] == 1:
        return "chan"
    return "other"


def operand(spec, row_major: bool, dtype, device):
    """A fresh operand with the recorded shape and stride pattern (or row-major).

    Returns ``(tensor, honoured)``: ``honoured`` is False when the pattern could not be
    rebuilt, in which case row-major is all this probe can offer and the row says so.
    """
    shape, stride = spec
    buf = torch.randn(shape, dtype=dtype, device=device)
    kind = code(shape, stride)
    if row_major or kind == "row":
        return buf, True
    if kind == "row+s":                       # a slice of a packed qkv-style buffer
        wide = torch.randn((*shape[:-1], stride[-2]), dtype=dtype, device=device)
        made = wide[..., :shape[-1]]
        return (made, True) if same(made.stride(), stride, shape) else (buf, False)
    if kind == "chan":
        n, c = shape[-2], shape[-1]
        base = torch.randn((*shape[:-2], c, n), dtype=dtype, device=device)
        made = base.transpose(-1, -2)
        return (made, True) if same(made.stride(), stride, shape) else (buf, False)
    return buf, False


def record(fx, op, res, device, dtype, channels_last) -> list:
    """Run the case once, keeping the shape+stride of every q/k/v it hands to attention."""
    import thenoise.utils.attention as attn_mod

    orig = attn_mod.single_head_attention
    seen: list = []

    def spy(q, k, v, rows=None):
        seen.append(tuple((tuple(t.shape), tuple(t.stride())) for t in (q, k, v)))
        return orig(q, k, v, rows)

    patched = []
    for name in ATTN_USERS:
        mod = importlib.import_module(name)
        if hasattr(mod, "single_head_attention"):
            patched.append((mod, mod.single_head_attention))
            mod.single_head_attention = spy
    try:
        with Layout(channels_last):
            case, vae = build(fx, op, res, device, dtype)
            with torch.no_grad():
                case.run()                    # the untimed pass; the calls are recorded
            del case, vae
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for mod, fn in patched:
            mod.single_head_attention = fn
    return seen


def dedupe(seen: list) -> list:
    """Distinct (shape, stride) signatures, with how many times the case asked."""
    out: dict[tuple, list] = {}
    for spec in seen:
        out.setdefault(spec, [0])[0] += 1
    return [dict(spec=k, calls=v[0]) for k, v in out.items()]


def timeit(fn, device, warmup=3, groups=3, iters=1) -> float:
    sync = torch.cuda.synchronize if device == "cuda" and torch.cuda.is_available() \
        else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync()
        runs = []
        for _ in range(groups):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            sync()
            runs.append((time.perf_counter() - t0) * 1000 / iters)
    runs.sort()
    return runs[len(runs) // 2]


def measure(row, device, dtype, groups, iters, attn):
    """Every variant timed on one recorded signature, plus the copies on their own."""
    shape = row["spec"][0][0]
    n, c = shape[-2], shape[-1]
    flop = 4.0 * float(n) * float(n) * float(c) * float(shape[0])   # 2 matmuls, 4 N^2 C
    got: dict[str, tuple[float, float, bool]] = {}
    plain = None
    for name in VARIANTS:
        want = WHICH[name]
        made, ok = [], True
        for i, spec in enumerate(row["spec"]):
            t, honoured = operand(spec, i in want, dtype, torch.device(device))
            made.append(t)
            ok = ok and honoured
        if name == "as-called":
            plain = list(made)
        ms = timeit(lambda: attn(*made), device, groups=groups, iters=iters)
        got[name] = (ms, flop / (ms / 1000) / 1e12 if ms else 0.0, ok)
        del made
    # What the normalisation costs by itself, so a copy budget cannot pose as a win.
    copy_ms = timeit(lambda: [t.contiguous() for t in plain], device,
                     groups=groups, iters=iters) if plain else 0.0
    del plain
    return got, copy_ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Attention shapes x input layouts, replayed.")
    ap.add_argument("--models", default="flux,flux2,qwen_image,qwen_image21,mage_flow")
    ap.add_argument("--ops", default="decode")
    ap.add_argument("--res", default="1024x1024")
    ap.add_argument("--dtype", default="bf16", choices=sorted(vb.DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--nchw-only", action="store_true",
                    help="skip the channels_last recording (half the run time)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no cuda/rocm device visible — run this on the GPU box")
    import thenoise.utils.attention as attn_mod
    from thenoise.utils.attention import SCORE_LIMIT_BYTES, score_tile_rows

    res = vb.Res.parse(args.res)
    ops = tuple(o.strip() for o in args.ops.split(",") if o.strip())
    dtype = vb.DTYPES[args.dtype]
    gpu = args.device == "cuda" and torch.cuda.is_available()
    print(f"{torch.cuda.get_device_name(0) if gpu else args.device} | "
          f"torch {torch.__version__} | {args.dtype} | {res} | score matrices over "
          f"{SCORE_LIMIT_BYTES / 2**30:.0f} GiB are tiled")

    for fx in vb.select(args.models):
        for op in fx.ops:
            if op not in ops:
                continue
            for channels_last in ((False, True) if not args.nchw_only else (False,)):
                arm = "channels_last" if channels_last else "nchw"
                rows = dedupe(record(fx, op, res, args.device, dtype, channels_last))
                if not rows:
                    print(f"\n{fx.model}/{op} [{arm}]  {res}: no attention call")
                    continue
                print(f"\n{fx.model}/{op} [{arm}]  {res}   "
                      f"{sum(r['calls'] for r in rows)} calls, "
                      f"{len(rows)} distinct shape(s)")
                print(f"  {'N':>7s} {'C':>5s} {'x':>3s} {'q/k/v':>10s} "
                      + " ".join(f"{v:>16s}" for v in VARIANTS) + f" {'copies':>9s}")
                for row in sorted(rows, key=lambda r: -r["spec"][0][0][-2]):
                    spec = row["spec"]
                    n, c = spec[0][0][-2], spec[0][0][-1]
                    lay = "/".join(code(s[0], s[1]) for s in spec)
                    got, copy_ms = measure(row, args.device, dtype, args.groups,
                                           args.iters, attn_mod.single_head_attention)
                    cells = [f"{got[v][0]:8.1f} {got[v][1]:7.1f}" for v in VARIANTS]
                    print(f"  {n:7d} {c:5d} {row['calls']:3d} {lay:>10s} "
                          + " ".join(f"{x:>16s}" for x in cells)
                          + f" {copy_ms:8.1f}m")
                    rows_n = score_tile_rows(n)
                    win = 100 * (1 - got["all"][0] / got["as-called"][0]) \
                        if got["as-called"][0] else 0.0
                    net = win - (100 * copy_ms / got["as-called"][0] if got["as-called"][0] else 0)
                    print(f"  {'':28s}score matrix: "
                          f"{'whole' if rows_n >= n else f'tiled, rows={rows_n}'} | "
                          f"row-major everywhere is {win:+.1f}% "
                          f"({net:+.1f}% net of the {copy_ms:.1f} ms of copies)"
                          + ("" if all(g[2] for g in got.values()) else
                             " | * a recorded stride could not be rebuilt"))
                    del got
                    gc.collect()
                    if gpu:
                        torch.cuda.empty_cache()
    print("\nColumns are ms and analytic TF (4*N^2*C): the same arithmetic in every one, so\n"
          "which column is fastest says which operand's strides decided the kernel.")


if __name__ == "__main__":
    main()
