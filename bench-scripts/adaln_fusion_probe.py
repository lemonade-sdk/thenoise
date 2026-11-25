"""The block's input-side norm and modulation: today's fp32 chain against ``ck.rms_adaln``.

Distinct from the QK-norm question. A DiT block normalises twice, on two different tensors:

  input side   ``rmsnorm(x) * (1 + scale) + shift`` before every projection (AdaLN), which is
               what ``ck.rms_adaln`` fuses into one kernel and what
               ``ck.int8_linear_rms_modulated`` puts in an INT8 GEMM's producer.
  output side  QK-norm over ``head_dim`` on q and k after the projection, which no GEMM here
               fuses -- see ``sol_producer_probe.py``.

H3 computes the input side as ``self.norm1(x.float()) * (1 + scale) + shift``, in fp32, once per
projection. This measures that against the fused form at production widths, and reports whether
the INT8 producer's fused path could even engage on this architecture.

    .venv/bin/python bench-scripts/adaln_fusion_probe.py --tokens 16384 --hidden 5376
"""

from __future__ import annotations

import argparse
import statistics

import torch
import comfy_kitchen as ck
from comfy_kitchen.backends.hip import _has_nonduplicated_wmma

from thenoise.utils.rms_norm import RMSNorm


def _measure(fn, iters: int):
    out = fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    times, peak = [], 0
    for _ in range(iters):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        out = fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
        peak = max(peak, torch.cuda.max_memory_allocated() - baseline)
    return out.float(), statistics.median(times), peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--hidden", type=int, default=5376)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, args.tokens, args.hidden, device=dev, dtype=dt, generator=gen)
    scale = torch.randn(1, 1, args.hidden, device=dev, dtype=dt, generator=gen) * 0.1
    shift = torch.randn(1, 1, args.hidden, device=dev, dtype=dt, generator=gen) * 0.1
    norm = RMSNorm(args.hidden, eps=args.eps).to(dev, dt)

    def today():
        return (norm(x.float()) * (1 + scale) + shift).to(dt)

    def fused():
        return ck.rms_adaln(x, scale, shift, args.eps)

    print(f"adaln probe: {args.tokens} tokens x {args.hidden} hidden, eps {args.eps:g}")
    print(f"  int8 fused GEMM producer engaged on this device: "
          f"{_has_nonduplicated_wmma(torch.device('cuda', torch.cuda.current_device()))} "
          f"(arch {torch.cuda.get_device_capability()}, {torch.cuda.get_device_name()})")
    base = None
    for name, fn in (("rms_norm + modulation (today)", today),
                     ("ck.rms_adaln", fused)):
        out, ms, peak = _measure(fn, args.iters)
        if base is None:
            base = out
            print(f"  {name:>30}  {ms:7.1f} ms  peak {peak / 2**20:7,.0f} MiB   (reference)")
        else:
            err = float((out - base).norm() / base.norm())
            print(f"  {name:>30}  {ms:7.1f} ms  peak {peak / 2**20:7,.0f} MiB  "
                  f"rel err {err:.3e}")
    print("\n  three of these run per block in H3 (norm1 before qkv, norm3 before the FFN, and "
          "the head),\n  so multiply by the block count for the block's share of a step.")


if __name__ == "__main__":
    main()
