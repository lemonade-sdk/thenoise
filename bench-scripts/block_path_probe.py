"""Whole compiled block: does a fused kernel beat Inductor's own fusion?

Isolated elementwise measurements mislead here. The engine compiles its blocks, so today's
rms_norm plus modulation is not a chain of materialised tensors -- Inductor fuses it into its
neighbours, and an opaque comfy_kitchen call replaces that fusion with a graph cut. ck.rms_adaln
measured 10x faster and 6x smaller in isolation and came out 1.7% *slower* in the block, so this
is the only probe in the directory whose numbers may be used to adopt anything.

Variations are installed on the class- and module-level names the compiled forward calls, so the
block under test is the real one and each variation compiles as its own artifact:

``adaln=ck``  ``trunk._modulate`` -> ck.rms_adaln. Rejected: no learned norm weight at all.
``qk=ck``     ``Attention.forward`` -> a mirror that does QK-norm and RoPE in one
              ck.rms_rope_split_half call, which needs both tensors at the call site and so has
              no drop-in seam in the trunk. The mirror is guarded below against the real forward
              so it cannot drift silently.

    .venv/bin/python bench-scripts/block_path_probe.py --tokens 8192
    .venv/bin/python bench-scripts/block_path_probe.py --tokens 40114 --iters 2
"""

import argparse
import statistics

import torch
import comfy_kitchen as ck

import thenoise.dit.h3.trunk as trunk
from thenoise.dit.h3.sparse import TAU
from thenoise.dit.h3.trunk import Attention, DiTBlock, H3Params, _spans, attend
from thenoise.utils.sol_attention import SolAttention

HID, HEADS, HDIM, FFN = 5376, 56, 128, 14336
ENGINE_MODULATE = trunk._modulate
ENGINE_ATTN_FORWARD = Attention.forward


def ck_modulate(x, spans, norm, attn):
    parts = []
    for span in spans:
        mod = span.attn if attn else span.mlp
        parts.append(ck.rms_adaln(x[:, span.start:span.end], mod.scale, mod.shift, norm.eps))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def rope_table(cos, sin):
    """The trunk's (cos, sin) as the kernel's dialect: [B, T, 1, rot/2, 2, 2], rows [[c,s],[-s,c]].

    The sign placement is the opposite of what the rotation formula suggests, and nothing
    complains when it is wrong: the kernel accepts the transposed-sign table and returns garbage
    at 1.1 relative error. fp32 only -- an fp16 table costs 1.2e-3.

    Built per call rather than cached: it is a couple of megabytes, and a cache keyed on tensor
    identity would hand the compiled region a constant the graph cannot see through.
    """
    c, s = cos.transpose(-2, -3), sin.transpose(-2, -3)      # [B, T, 1, rot/2]
    return torch.stack((torch.stack((c, s), dim=-1),
                        torch.stack((-s, c), dim=-1)), dim=-1)


def fused_attention_forward(self, x, rope=None, bufs=None, sol=None):
    """Mirror of ``Attention.forward`` whose only difference is one fused norm-and-rope call."""
    batch, rows, _ = x.shape
    inner = self.heads * self.head_dim
    q, k, v = self.qkv_proj(x).split(inner, dim=-1)
    q, k, v = (t.view(batch, rows, self.heads, self.head_dim) for t in (q, k, v))
    norm = self.qk_norm
    q, k = ck.rms_rope_split_half(q, k, rope_table(*rope),
                                  norm.query_norm.weight, norm.key_norm.weight,
                                  norm.query_norm.eps, 2 * rope[0].shape[-1])
    if sol is None:
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        return self.out_proj(attend(q, k, v, bufs))
    return self.out_proj(trunk.sol_attention(q, k, v, sol).reshape(batch, rows, inner))


def guards(attn, x, cos, sin, plan):
    """Refuse to measure a mirror that does not compute what the trunk computes."""
    with torch.no_grad():
        ref = {}
        Attention.forward = ENGINE_ATTN_FORWARD
        for label, sol in (("dense", None), ("routed", plan)):
            ref[label] = attn(x, (cos, sin), None, sol).float()
        Attention.forward = fused_attention_forward
        for label, sol in (("dense", None), ("routed", plan)):
            got = attn(x, (cos, sin), None, sol).float()
            err = float((got - ref[label]).norm() / ref[label].norm())
            print(f"  mirror vs trunk, {label:<6} rel err {err:.3e}")
            if err > 5e-3:
                raise SystemExit(f"mirror drifted from the trunk on the {label} path "
                                 f"({err:.3e}); the fused kernel and the chain disagree, or the "
                                 f"mirror is out of date")
    Attention.forward = ENGINE_ATTN_FORWARD


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--adaln", action="store_true",
                    help="also vary the input-side norm, which is already rejected")
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
    cos = torch.randn(1, 1, T, HDIM // 2, device=dev, dtype=torch.float32, generator=g)
    sin = torch.randn(1, 1, T, HDIM // 2, device=dev, dtype=torch.float32, generator=g)
    spans = _spans([(0, T, 0)],
                   block.adaln_proj(torch.randn(1, 32, device=dev, dtype=torch.float32)))
    plan = SolAttention(TAU, sink_blocks=(0, 4), sink_q=(4, 8))

    print(f"block path probe: compiled DiTBlock {HID} hidden, {HEADS}x{HDIM}, T={T}, "
          f"{'compiled' if not torch._dynamo.config.disable else 'EAGER'}")
    guards(block.attn, x, cos, sin, plan)

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

    adaln_arms = (("engine", ENGINE_MODULATE),) + (("ck", ck_modulate),) if args.adaln else ()
    for qk, attn_impl in (("engine", ENGINE_ATTN_FORWARD),
                          ("ck", fused_attention_forward)):
        Attention.forward = attn_impl
        for adaln, mod_impl in (("engine", ENGINE_MODULATE),) + adaln_arms:
            trunk._modulate = mod_impl
            for att in ("dense", "routed"):
                extra = (plan,) if att == "routed" else ()
                try:
                    rounds = [timed(lambda: block(x, spans, (cos, sin), None, *extra))
                              for _ in range(args.rounds)]
                except Exception as exc:
                    print(f"  qk={qk:<6} adaln={adaln:<6} att={att:<6} refused: "
                          f"{type(exc).__name__}: {str(exc)[:60]}")
                    continue
                ms = min(r[0] for r in rounds)
                peak = min(r[1] for r in rounds)
                print(f"  qk={qk:<6} adaln={adaln:<6} att={att:<6}  {ms:8.1f} ms  "
                      f"peak {peak / 2**20:8,.0f} MiB   rounds: "
                      f"{' '.join(f'{r[0]:.0f}' for r in rounds)}")
    Attention.forward = ENGINE_ATTN_FORWARD
    trunk._modulate = ENGINE_MODULATE


if __name__ == "__main__":
    main()
