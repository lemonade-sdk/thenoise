"""What the chunked producer and a fused norm+rope would each buy, before changing a trunk.

The peaks here are isolated and eager, so they include temporaries that Inductor fuses away in a
real block: read this as an implementation comparison (which kernel is better at the same job and
how closely they agree), and use bench-scripts/block_path_probe.py for any adoption decision.
Section 15 of docs/research/sparse-attention.md has both sets of numbers and the correction that
followed from the second one.

Two candidate changes to the attention path, measured on the same operands rather than argued
about:

``sol_attn_chunked``  takes chunks of the fused qkv projection and does QK-norm and RoPE inside
                      the kernel, carrying K and V as int8. Never materialises Q/K/V.
``rms_rope_split_half`` (Triton)  one kernel for QK-norm and RoPE over ``[B, T, H, D]`` operands,
                      replacing this engine's ``QKNorm`` + ``apply_rope_split_half``, which
                      broadcast fp32 phases against bf16 queries and so compute in fp32.

Both promise memory. Each row of the matrix is one norm+rope choice feeding one attention
choice, and the peak allocation, the time and the output's distance from the shipped
``engine + dense`` are printed for every row. The dense consumer is in the matrix because the
dense path is not a rare case: it is every step of every still, every step outside the routing
window, and the first step of every routed run.

    .venv/bin/python bench-scripts/sol_producer_probe.py --tokens 16384 --heads 56
    .venv/bin/python bench-scripts/sol_producer_probe.py --tokens 16384 --rope random
    .venv/bin/python bench-scripts/sol_producer_probe.py --tokens 40114 --heads 42 --iters 1
"""

from __future__ import annotations

import argparse
import statistics

import torch
from comfy_kitchen import rms_rope_split_half as ck_rms_rope, sol_attn, sol_attn_chunked

from thenoise.dit.kvcache import attend
from thenoise.utils.qk_norm import QKNorm
from thenoise.utils.rope import apply_rope_split_half

HEAD_DIM = 128
NORM_ROPE = ("engine", "engine_bf16bands", "ck_fused")


@torch.no_grad()              # a grad graph would inflate the peak being measured
def _measure(fn, iters: int):
    """Median milliseconds and the largest allocation the calls caused, in bytes."""
    out = fn()                                   # warm, so the baseline is honest
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
    ap.add_argument("--rope", choices=("identity", "random"), default="identity",
                    help="identity keeps the rotation out of the comparison; random puts the "
                         "table's convention in it, which is how a wrong table shows up")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--skip", action="append", default=[],
                    help="rows to skip, e.g. --skip ck_fused+dense")
    args = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    t, heads = args.tokens, args.heads
    inner, width, rot = heads * HEAD_DIM, 3 * heads * HEAD_DIM, HEAD_DIM
    half = rot // 2
    gen = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, t, inner, device=dev, dtype=dt, generator=gen) * 0.5
    wT = (torch.randn(width, inner, device=dev, dtype=dt, generator=gen)
          * inner ** -0.5).t().contiguous()

    if args.rope == "identity":
        angle = torch.zeros(1, t, 1, half, device=dev)
    else:
        angle = torch.randn(1, t, 1, half, device=dev, generator=gen)
    cos, sin = torch.cos(angle), torch.sin(angle)
    # The kernel's table is the same dialect the chunked producer takes: per pair, [[c,-s],[s,c]].
    table = torch.empty(1, t, 1, half, 2, 2, device=dev, dtype=torch.float32)
    table[..., 0, 0], table[..., 0, 1] = cos, -sin
    table[..., 1, 0], table[..., 1, 1] = sin, cos

    # A bf16 model carries its QK-norm weight in bf16, which is what lets rms_norm fuse.
    norm = QKNorm(HEAD_DIM, eps=1e-5).to(dev, dt)
    scale = torch.ones(HEAD_DIM, device=dev, dtype=dt)
    sink, sink_q = [0, args.sink_blocks], [0, 0]

    def projection():
        return (s.view(1, t, heads, HEAD_DIM) for s in (x @ wT).split(inner, dim=-1))

    def norm_rope(mode: str):
        """q, k, v token-major, normed and rotated, by one of the candidate implementations."""
        q, k, v = projection()
        if mode == "ck_fused":
            return (*ck_rms_rope(q, k, table, scale, scale, 1e-5, rot), v)
        q, k = norm(q, k)
        if mode == "engine_bf16bands":
            q = apply_rope_split_half(q, cos.to(dt), sin.to(dt))
            k = apply_rope_split_half(k, cos.to(dt), sin.to(dt))
        else:
            q = apply_rope_split_half(q, cos, sin)
            k = apply_rope_split_half(k, cos, sin)
        return q, k, v

    pooled: dict = {}

    def row(mode: str, attention: str):
        def arm():
            if attention == "producer":
                def chunks():
                    for start in range(0, t, args.chunk):
                        yield x[:, start:start + args.chunk] @ wT
                # Stats carried as an adopted trunk would carry them, so the timed call is a
                # single pass rather than the bootstrap's two.
                out, kmean, vscale = sol_attn_chunked(
                    chunks, t, heads, table, (scale, scale), kmean=pooled.get("k"),
                    vscale=pooled.get("v"), tau=args.tau, sink_blocks=sink, sink_q=sink_q,
                    token_aug=256, rope_eps=1e-5, rope_split_half=True)
                pooled["k"], pooled["v"] = kmean, vscale
                return out.flatten(2)
            q, k, v = norm_rope(mode)
            if attention == "none":
                return q.flatten(2)
            if attention == "routed":
                return sol_attn(q, k, v, tau=args.tau, sink_blocks=sink, sink_q=sink_q,
                                token_aug=256).flatten(2)
            return attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return arm

    print(f"sol attention-path probe: {t} tokens, {heads}x{HEAD_DIM}, chunk {args.chunk}, "
          f"tau {args.tau:g}, rope {args.rope}, norm weight {norm.query_norm.weight.dtype}")
    print(f"  a full-length fused projection is {t * width * 2 / 2**20:,.0f} MiB; a chunk of "
          f"{args.chunk} is {args.chunk * width * 2 / 2**20:,.0f} MiB\n")

    rows = [("engine", "none"), ("engine_bf16bands", "none"), ("ck_fused", "none"),
            ("engine", "dense"), ("engine_bf16bands", "dense"), ("ck_fused", "dense"),
            ("engine", "routed"), ("engine_bf16bands", "routed"), ("ck_fused", "routed"),
            ("producer", "producer")]
    got: dict = {}
    for mode, attention in rows:
        name = f"{mode}+{attention}"
        if name in args.skip:
            continue
        try:
            got[name] = _measure(row(mode, attention), args.iters)
        except Exception as exc:                        # a contract change is a result too
            print(f"  {name:>26}  refused: {type(exc).__name__}: {str(exc)[:80]}")

    def against(name, other):
        """Relative distance between two rows that are supposed to compute the same thing."""
        if name not in got or other not in got:
            return ""
        a, b = got[name][0], got[other][0]
        return f"  vs {other:>18} {float((a - b).norm() / b.norm()):.3e}"

    for mode, attention in rows:
        name = f"{mode}+{attention}"
        if name not in got:
            continue
        out, ms, peak = got[name]
        mate = f"{mode}+{attention}"
        peer = "engine+none" if attention == "none" else "engine+dense"
        line = f"  {name:>26}  {ms:8.1f} ms  peak {peak / 2**20:8,.0f} MiB"
        line += "" if name == peer else against(name, peer)
        if name == "producer+producer":
            line += against("producer+producer", "engine+routed")
        print(line)

    print("\n  Each norm+rope row is the same operands through three implementations of QK-norm and "
          "\n  rope; each attention row adds one attention. Error is against the engine's own "
          "row,\n  so a routed row's number carries the routing error on Gaussian content as "
          "well as\n  whatever the row costs -- compare routed to routed, never to dense. "
          "The producer\n  carries norm and rope inside it, hence no norm+rope counterpart.")


if __name__ == "__main__":
    main()
