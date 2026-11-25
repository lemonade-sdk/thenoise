"""Does torch.compile still pay now that the attention path is an opaque kernel?

The trunk's blocks are compiled, and the attention island deliberately breaks the graph. Two
other opaque kernels are in the picture too: the quantized GEMMs, and — if adopted — the fused
QK-norm/RoPE. This measures the block with and without compilation, dense and routed, so the
answer is a number per shape rather than a position.

Run it twice; the mode comes from the environment, because ``TORCHDYNAMO_DISABLE`` is read where
``torch.compile`` is applied:

    .venv/bin/python bench-scripts/compile_value_probe.py --tokens 8192
    TORCHDYNAMO_DISABLE=1 .venv/bin/python bench-scripts/compile_value_probe.py --tokens 8192

Arms alternate within a process, because two processes on an iGPU disagree on the dense arm by
30% and that is larger than the effect being looked for.
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from thenoise.dit.h3.sparse import TAU
from thenoise.dit.h3.trunk import DiTBlock, H3Params, _spans
from thenoise.utils.sol_attention import SolAttention

HID, HEADS, HDIM, FFN, ROT_HALF = 5376, 56, 128, 14336, 48


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--quantized", action="store_true",
                    help="run the block's linears through a quantized weight instead of bf16")
    args = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    params = H3Params(hidden_size=HID, num_layers=1, token_refiner_layers=0, num_heads=HEADS,
                      head_dim=HDIM, ffn_hidden_size=FFN, latents_dim=24, text_dim=5120,
                      rope_inv_freq=16, adaln_grid=100, adaln_dim=32, pdd_steps=1,
                      pdd_sigmas=(1.0, 0.0), audio_dim=32)
    torch.manual_seed(0)
    block = DiTBlock(params, compute_dtype=dt).to(dev, dt).eval().requires_grad_(False)
    compiled = not torch._dynamo.config.disable
    gen = torch.Generator(device=dev).manual_seed(2)
    t_emb = torch.randn(1, 32, device=dev, dtype=torch.float32)
    plan = SolAttention(TAU, sink_blocks=(0, 4), sink_q=(4, 8))

    def timed(fn, iters):
        with torch.no_grad():
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            out = []
            for _ in range(iters):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                fn()
                b.record()
                torch.cuda.synchronize()
                out.append(a.elapsed_time(b))
        return statistics.median(out)

    print(f"compile value probe: {HID} hidden, {HEADS}x{HDIM}, ffn {FFN}, "
          f"{'compiled' if compiled else 'EAGER (TORCHDYNAMO_DISABLE)'}")
    for tokens in args.tokens:
        x = torch.randn(1, tokens, HID, device=dev, dtype=dt, generator=gen) * 0.05
        cos = torch.randn(1, 1, tokens, ROT_HALF, device=dev, dtype=torch.float32,
                          generator=gen)
        sin = torch.randn(1, 1, tokens, ROT_HALF, device=dev, dtype=torch.float32,
                          generator=gen)
        spans = _spans([(0, tokens, 0)], block.adaln_proj(t_emb))
        dense = lambda: block(x, spans, (cos, sin), None)                    # noqa: E731
        routed = lambda: block(x, spans, (cos, sin), None, plan)             # noqa: E731
        first_dense = timed(dense, 1)
        first_routed = timed(routed, 1)
        acc = {"dense": [], "routed": []}
        for _ in range(args.rounds):
            acc["dense"].append(timed(dense, args.iters))
            acc["routed"].append(timed(routed, args.iters))
        print(f"  T={tokens}: dense {' '.join(f'{m:7.1f}' for m in acc['dense'])} | "
              f"routed {' '.join(f'{m:7.1f}' for m in acc['routed'])}  (ms)")
        print(f"           first call after the switch: dense {first_dense:.1f} ms, "
              f"routed {first_routed:.1f} ms "
              f"({'compile' if first_dense > 4 * min(acc['dense']) else 'steady'})")


if __name__ == "__main__":
    main()
