#!/usr/bin/env python
"""Can VAE attention run with no copies at all, in the layout the convs produce?

A 1x1 conv on an NCHW feature map hands over a (C, HW) matrix. A GEMM needs a leading
dimension, not a particular orientation, so in principle all three matmuls can consume
that form as-is — and the last one can even *write* its result as (C, HW) so the proj
conv needs no copy either. Open questions: does torch's matmul honour those views or
quietly copy them, and what do the GEMMs cost per orientation?

Forms compared, same arithmetic, operands produced as the codecs produce them (a
contiguous (b, c, hw) conv output, viewed as tokens):
  flux     copies q, k, v to token-major, copies the result back to NCHW
  flux2    hands over the transposed conv outputs, copies the result back
  nocopy   the same views in, a (C, HW) result out, score scale folded into q
  tiled    nocopy with thenoise's score tiling
Each is profiled, so a hidden copy shows up as a kernel instead of as a hunch.
"""
import math
import time
import torch
from collections import Counter
from torch.profiler import profile, ProfilerActivity

dev = "cuda"
torch.manual_seed(0)


def short(name, n=52):
    """Kernel names are enormous; the first words say enough."""
    return name if len(name) <= n else name[:n] + "…"


def timeit(fn, iters=5, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def staged(fn, iters=2):
    """Copy/transpose/cast kernels and aten copy calls made by fn."""
    aten = Counter()
    with torch.no_grad():
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for _ in range(iters):
                fn()
                torch.cuda.synchronize()
    ks = Counter()
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and getattr(e, "device_time", 0):
            ks[e.name] += 1
        elif e.key in ("aten::copy_", "aten::contiguous", "aten::to"):
            aten[e.key] += 1
    st = {short(n): c for n, c in ks.items()
          if any(t in n.lower() for t in ("copy", "transpose", "cast", "convert"))}
    return st, dict(aten)


for c, hw in ((512, 4096), (512, 16384)):
    b = 1
    print(f"\n=== C={c}, HW={hw}  score matrix {2 * hw * hw / 2**20:.0f} MiB bf16 ===")
    scale = math.sqrt(c)
    flop = 4 * hw * hw * c * b
    x = torch.randn(b, c, hw, device=dev, dtype=torch.bfloat16)
    wq = torch.randn(c, c, device=dev, dtype=torch.bfloat16) / c ** 0.5
    wk = torch.randn(c, c, device=dev, dtype=torch.bfloat16) / c ** 0.5
    wv = torch.randn(c, c, device=dev, dtype=torch.bfloat16) / c ** 0.5
    bq = torch.randn(c, device=dev, dtype=torch.bfloat16) * 0.1
    bk = torch.randn(c, device=dev, dtype=torch.bfloat16) * 0.1
    bv = torch.randn(c, device=dev, dtype=torch.bfloat16) * 0.1

    def conv(w, bb, gain=1.0):
        """A 1x1 conv over the feature map: contiguous (b, c, hw), as MIOpen hands it over.
        `gain` folds the score scale into weight and bias — free, done once at load."""
        return torch.matmul(w * gain, x) + bb.view(1, c, 1) * gain

    def tokens(t):
        """(b, c, hw) conv output -> (b, 1, hw, c) token view. A view, not a copy."""
        return t.view(b, 1, c, hw).transpose(2, 3)

    qc, kc, vc = conv(wq, bq), conv(wk, bk), conv(wv, bv)
    qcs = conv(wq, bq, 1.0 / scale)   # the same conv with 1/sqrt(C) folded into weight+bias
    _zero = torch.zeros((), dtype=torch.bfloat16, device=dev)

    def flux():
        q, k, v = (tokens(t).contiguous() for t in (qc, kc, vc))
        s = torch.matmul(q, k.transpose(-2, -1))
        s.div_(scale)
        o = torch.matmul(s.softmax(-1), v)
        return o.transpose(2, 3).reshape(b, c, hw).contiguous()

    def flux2():
        q, k, v = (tokens(t) for t in (qc, kc, vc))
        s = torch.matmul(q, k.transpose(-2, -1))
        s.div_(scale)
        o = torch.matmul(s.softmax(-1), v)
        return o.transpose(2, 3).reshape(b, c, hw).contiguous()

    def nocopy():
        """Views in, (C, HW) out: the weighted sum is computed transposed, so its
        contiguous axis is the channel axis the proj conv wants."""
        q, k, v = (tokens(t) for t in (qc, kc, vc))
        qs = qcs.view(b, 1, c, hw).transpose(2, 3)
        out = torch.empty((b, c, hw), dtype=torch.bfloat16, device=dev)
        s = torch.matmul(qs, k.transpose(-2, -1))
        p = s.softmax(-1)
        torch.matmul(v.transpose(-2, -1), p.transpose(-2, -1), out=out.view(b, 1, c, hw))
        return out

    def tiled(rows=1024):
        """nocopy with thenoise's score tiling."""
        k, v = tokens(kc), tokens(vc)
        qs = tokens(qcs)
        out = torch.empty((b, c, hw), dtype=torch.bfloat16, device=dev)
        outb = out.view(b, 1, c, hw)
        scratch = torch.empty((b, 1, rows, hw), dtype=torch.bfloat16, device=dev)
        kt, vt = k.transpose(-2, -1), v.transpose(-2, -1)
        for start in range(0, hw, rows):
            stop = min(start + rows, hw)
            tile = scratch[..., :stop - start, :]
            torch.matmul(qs[..., start:stop, :], kt, out=tile)
            p = tile.softmax(-1)
            torch.matmul(vt, p.transpose(-2, -1), out=outb[..., start:stop])
        return out

    def views_token_nodiv():
        """Isolate: views in, token-major out, no scale at all (wrong answer, right cost)."""
        q, k, v = (tokens(t) for t in (qc, kc, vc))
        s = torch.matmul(q, k.transpose(-2, -1))
        o = torch.matmul(s.softmax(-1), v)
        return o.transpose(2, 3).reshape(b, c, hw).contiguous()

    def views_cm_withdiv():
        """Isolate: views in, (C, HW) out, but the score still divided afterwards."""
        q, k, v = (tokens(t) for t in (qc, kc, vc))
        out = torch.empty((b, c, hw), dtype=torch.bfloat16, device=dev)
        s = torch.matmul(q, k.transpose(-2, -1))
        s.div_(scale)
        torch.matmul(v.transpose(-2, -1), s.softmax(-1).transpose(-2, -1),
                     out=out.view(b, 1, c, hw))
        return out

    def views_cm_alpha():
        """The scale inside the first GEMM: baddbmm's alpha is the BLAS epilogue
        scalar, so it is applied to the fp32 accumulator — no extra pass, no scaled
        operand, no second rounding of the score matrix."""
        q, k, v = (tokens(t) for t in (qc, kc, vc))
        out = torch.empty((b, c, hw), dtype=torch.bfloat16, device=dev)
        s = torch.empty((b, 1, hw, hw), dtype=torch.bfloat16, device=dev)
        if b != 1:
            return flux()
        q3, k3 = q.squeeze(0), k.squeeze(0).transpose(-2, -1)
        s3 = s.squeeze(0)
        torch.baddbmm(s3, q3, k3, beta=0.0, alpha=1.0 / scale, out=s3)
        torch.matmul(v.transpose(-2, -1), s.softmax(-1).transpose(-2, -1),
                     out=out.view(b, 1, c, hw))
        return out

    ref = flux().float()
    ref32 = None
    if hw <= 4096:
        qf, kf, vf = (tokens(t).float() for t in (qc, kc, vc))
        sf = torch.matmul(qf, kf.transpose(-2, -1)) / scale
        ref32 = torch.matmul(sf.softmax(-1), vf).transpose(2, 3).reshape(b, c, hw)
    for name, fn in (("flux   3 copies in, 1 out", flux),
                     ("flux2  views in,  1 out", flux2),
                     ("views in, token out, no div", views_token_nodiv),
                     ("views in, (C,HW) out, div", views_cm_withdiv),
                     ("nocopy views in, 0 out", nocopy),
                     ("alpha in GEMM, 0 out", views_cm_alpha),
                     ("tiled  views in, 0 out (rows=1024)", lambda: tiled(1024))):
        got = fn()
        rel = ((got.float() - ref).norm() / ref.norm()).item()
        rel32 = ("-" if ref32 is None else
                 f"{((got.float() - ref32).norm() / ref32.norm()).item():.1e}")
        ms = timeit(fn)
        st, at = staged(fn)
        note = "no copies" if not st and not at else f"COPIES {st or at}"
        print(f"  {name:<32s} {ms:7.2f} ms  {flop/ms*1e-9:5.1f} TF  "
              f"vs flux {rel:.1e}  vs fp32 {rel32:>5s}  {note}")
    del x, qc, kc, vc
