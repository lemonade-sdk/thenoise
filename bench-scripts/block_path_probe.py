"""Whole compiled block: does a fused AdaLN beat Inductor's own fusion?

Isolated elementwise measurements mislead here. The engine compiles its blocks, so today's
rms_norm plus modulation is not a chain of materialised tensors -- Inductor fuses it into its
neighbours, and an opaque comfy_kitchen call replaces that fusion with a graph cut.
ck.rms_adaln measured 10x faster and 6x smaller in isolation; this is where it has to prove it.
The variation is installed on the module-level name the compiled forward calls, so the block
under test is the real one and each variation compiles as its own artifact.

    .venv/bin/python bench-scripts/block_path_probe.py --tokens 8192

Two things this cannot vary, recorded rather than worked around. ck.rms_adaln takes no
norm_weight, so it stands in only for a pre-norm whose learned weight is identity: the rows below
fill ones, which is a fresh block and not a checkpoint. And the fused QK-norm plus RoPE has no
single-tensor form with rot_dim, while this trunk rotates a prefix (96 of 128) and rotates q and
k in two separate calls -- so that comparison has no drop-in seam, and the pair kernel needs both
tensors at the call site.
"""

import argparse
import statistics

import torch
import comfy_kitchen as ck

import thenoise.dit.h3.trunk as trunk
from thenoise.dit.h3.sparse import TAU
from thenoise.dit.h3.trunk import DiTBlock, H3Params, _spans
from thenoise.utils.sol_attention import SolAttention

HID, HEADS, HDIM, FFN = 5376, 56, 128, 14336
ENGINE_MODULATE = trunk._modulate


def ck_modulate(x, spans, norm, attn):
    parts = []
    for span in spans:
        mod = span.attn if attn else span.mlp
        parts.append(ck.rms_adaln(x[:, span.start:span.end], mod.scale, mod.shift, norm.eps))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    params = H3Params(hidden_size=HID, num_layers=1, token_refiner_layers=0, num_heads=HEADS,
                      head_dim=HDIM, ffn_hidden_size=FFN, latents_dim=24, text_dim=5120,
                      rope_inv_freq=16, adaln_grid=100, adaln_dim=32, pdd_steps=1,
                      pdd_sigmas=(1.0, 0.0), audio_dim=32)
    torch.manual_seed(0)
    block = DiTBlock(params, compute_dtype=dt).to(dev, dt).eval().requires_grad_(False)
    with torch.no_grad():                       # ck.rms_adaln has no learned norm weight
        for norm in (block.norm1, block.norm2):
            norm.weight.fill_(1.0)
    g = torch.Generator(device=dev).manual_seed(2)
    T = args.tokens
    x = torch.randn(1, T, HID, device=dev, dtype=dt, generator=g) * 0.05
    cos = torch.randn(1, 1, T, 48, device=dev, dtype=torch.float32, generator=g)
    sin = torch.randn(1, 1, T, 48, device=dev, dtype=torch.float32, generator=g)
    spans = _spans([(0, T, 0)],
                   block.adaln_proj(torch.randn(1, 32, device=dev, dtype=torch.float32)))
    plan = SolAttention(TAU, sink_blocks=(0, 4), sink_q=(4, 8))

    def timed(fn):
        with torch.no_grad():
            fn()                                   # compile, then measure
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            out = []
            for _ in range(args.iters):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                fn()
                b.record()
                torch.cuda.synchronize()
                out.append(a.elapsed_time(b))
            return statistics.median(out), torch.cuda.max_memory_allocated()

    print(f"block path probe: compiled DiTBlock {HID} hidden, {HEADS}x{HDIM}, T={T}, "
          f"{'compiled' if not torch._dynamo.config.disable else 'EAGER'}")
    for label, impl in (("engine", ENGINE_MODULATE), ("ck", ck_modulate)):
        trunk._modulate = impl
        for att in ("dense", "routed"):
            extra = (plan,) if att == "routed" else ()
            try:
                rounds = [timed(lambda: block(x, spans, (cos, sin), None, *extra))
                          for _ in range(args.rounds)]
            except Exception as exc:
                print(f"  adaln={label:<6} att={att:<6} refused: "
                      f"{type(exc).__name__}: {str(exc)[:60]}")
                continue
            ms = min(r[0] for r in rounds)
            peak = min(r[1] for r in rounds)
            print(f"  adaln={label:<6} att={att:<6}  {ms:8.1f} ms  "
                  f"peak {peak / 2**20:8,.0f} MiB   rounds: "
                  f"{' '.join(f'{r[0]:.0f}' for r in rounds)}")
    trunk._modulate = ENGINE_MODULATE


if __name__ == "__main__":
    main()
