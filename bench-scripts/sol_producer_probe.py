"""What the chunked producer would buy, before any trunk is changed for it.

``sol_attn_chunked`` takes chunks of the fused qkv projection and never materialises Q/K/V:
its promise is memory, and its price is that K and V are carried as int8. Both sides are
measurable against operands rather than against a refactoring, so this script reproduces the
producer's contract (chunks of ``[M, 3*H*128]``, 64-aligned starts, ``B=1``, norm and rope
done inside) beside the path the engine ships, and reports the three numbers the decision
needs: peak memory, time, and how far the output moves from dense.

Arms, same operands throughout:

``dense``      fused projection, QKNorm and rope outside, SDPA. The shipped dense path.
``routed``     ... and the plain kernel on the projection's three slices. What ships today.
``producer``   the projection itself in chunks, straight into the kernel, K/V as int8.
``producer@0`` the same at tau 0, routing nothing: its error is the int8 carriers alone.

Run with identity rope coefficients first (the default): then any difference between
``producer`` and ``dense`` is quantization and convention, nothing else. With ``--rope random``
a difference also means the rotation table was read wrong, which is the convention most likely
to be got wrong in a real adoption.

    .venv/bin/python bench-scripts/sol_producer_probe.py --tokens 16384 --heads 56
    .venv/bin/python bench-scripts/sol_producer_probe.py --tokens 16384 --rope random
"""

from __future__ import annotations

import argparse
import statistics

import torch
from comfy_kitchen import sol_attn, sol_attn_chunked

from thenoise.dit.kvcache import attend
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rope import apply_rope_split_half

HEAD_DIM = 128


def _measure(fn, iters: int):
    """Median milliseconds and the largest allocation the calls caused, in bytes."""
    out = fn()                                   # warm, and the baseline is then honest
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    times, peak = [], 0
    for _ in range(iters):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        peak = max(peak, torch.cuda.max_memory_allocated() - baseline)
    return out.float(), statistics.median(times), peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=1.3)
    ap.add_argument("--sink-blocks", type=int, default=4,
                    help="key blocks held exact, as a clip's first frame would be")
    ap.add_argument("--rope", choices=("identity", "random"), default="identity")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    t, heads = args.tokens, args.heads
    inner, width, rot = heads * HEAD_DIM, 3 * heads * HEAD_DIM, HEAD_DIM
    gen = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, t, inner, device=dev, dtype=dt, generator=gen) * 0.5
    wT = (torch.randn(width, inner, device=dev, dtype=dt, generator=gen)
          * inner ** -0.5).t().contiguous()

    # One rotation, in both dialects: cos/sin for the engine's helper, and the kernel's table
    # of [[c, -s], [s, c]] per pair.
    half = rot // 2
    if args.rope == "identity":
        angle = torch.zeros(1, 1, t, half, device=dev)
    else:
        angle = torch.randn(1, 1, t, half, device=dev, generator=gen)
    # Both arms norm and rotate token-major, which is bit-identical to the head-major order.
    cos = torch.cos(angle).permute(0, 2, 1, 3).contiguous()
    sin = torch.sin(angle).permute(0, 2, 1, 3).contiguous()
    table = torch.empty(1, t, half, 2, 2, device=dev, dtype=torch.float32)
    table[..., 0, 0], table[..., 0, 1] = cos.reshape(t, half), -sin.reshape(t, half)
    table[..., 1, 0], table[..., 1, 1] = sin.reshape(t, half), cos.reshape(t, half)

    norm = QKNorm(HEAD_DIM, eps=1e-5).to(dev)
    with torch.no_grad():
        for p in norm.parameters():
            p.fill_(1.0)
    weights = (torch.ones(HEAD_DIM, device=dev, dtype=dt),
               torch.ones(HEAD_DIM, device=dev, dtype=dt))
    sink, sink_q = [0, args.sink_blocks], [0, 0]

    def normed():
        qkv = x @ wT
        q, k, v = (s.view(1, t, heads, HEAD_DIM) for s in qkv.split(inner, dim=-1))
        q, k = norm(q, k)
        return apply_rope_split_half(q, cos, sin), apply_rope_split_half(k, cos, sin), v

    def project():
        def arm():
            q, k, v = normed()
            return q.flatten(2)
        return arm

    def shipped(kernel: bool):
        def arm():
            q, k, v = normed()
            if kernel:
                return sol_attn(q, k, v, tau=args.tau, sink_blocks=sink, sink_q=sink_q,
                                token_aug=256).flatten(2)
            return attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return arm

    pooled: dict = {}

    def producer(at_tau: float):
        def chunks():
            for start in range(0, t, args.chunk):
                yield x[:, start:start + args.chunk] @ wT

        def arm():
            out, kmean, vscale = sol_attn_chunked(
                chunks, t, heads, table, weights, kmean=pooled.get("k"),
                vscale=pooled.get("v"), tau=at_tau, sink_blocks=sink, sink_q=sink_q,
                token_aug=256, rope_eps=1e-5, rope_split_half=True)
            pooled["k"], pooled["v"] = kmean, vscale
            return out.flatten(2)
        return arm

    print(f"sol producer probe: {t} tokens, {heads}x{HEAD_DIM}, chunk {args.chunk}, "
          f"tau {args.tau:g}, rope {args.rope}")
    print(f"  a full-length fused projection is {t * width * 2 / 2**20:,.0f} MiB; "
          f"one chunk of {args.chunk} is {args.chunk * width * 2 / 2**20:,.0f} MiB, so the "
          f"producer's ceiling on that buffer is "
          f"{(t - args.chunk) * width * 2 / 2**20:,.0f} MiB\n")

    dense = routed = None
    for name, arm in (("project", project()), ("dense", shipped(False)),
                      ("routed", shipped(True)), ("producer", producer(args.tau))):
        try:
            out, ms, peak = _measure(arm, args.iters)
        except Exception as exc:                        # a contract change is a result too
            print(f"  {name:>11}  refused: {type(exc).__name__}: {exc}")
            continue
        against = ""
        if name != "project":
            if dense is None:
                dense = out
            against = f"  vs dense {float((out - dense).norm() / dense.norm()):.3e}"
        if name == "routed":
            routed = out
        elif name == "producer" and routed is not None:
            # The same routing rule on the same operands, so this difference is the int8
            # carriers and the fused norm-and-rope, and nothing else.
            against += f"  vs routed {float((out - routed).norm() / routed.norm()):.3e}"
        print(f"  {name:>11}  {ms:8.1f} ms  peak {peak / 2**20:8,.0f} MiB{against}")
    print("\n  'project' is what the trunk holds before any attention: the producer's saving "
          "over it\n  is the projection buffer it never forms. 'vs routed' at equal tau is the "
          "int8 price alone.")


if __name__ == "__main__":
    main()
