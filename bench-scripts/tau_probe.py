"""What a routing threshold actually keeps — a measurement, so it lives here and not in
the engine.

``tau`` is a significance level on a proxy row rather than a budget: the same value keeps a
different share of the sequence depending on how the scores happen to be distributed, which
is why shipping a readout inside ``sol_attention`` was a per-call cost paid by every run for
a number somebody reads once. It is the kernel's own test, counted instead of applied: a
proxy row per query block, its centroid against every centred key-block centroid, and
``tau`` sigmas of that row under an independence null.

    python bench-scripts/tau_probe.py --tokens 16384 --heads 56 --head-dim 128
    python bench-scripts/tau_probe.py --capture operands.pt

Random operands show the mechanism working; the number worth having is a real run's q and k,
captured by saving the two tensors at the ``sol_attention`` call site.

Run with the engine's own venv (``.venv/bin/python``) on the GPU the routing runs on.
"""

from __future__ import annotations

import argparse

import torch

#: The kernel's block, shared with the engine: what a kept fraction is counted in.
from thenoise.utils.sol_attention import BLOCK

TAUS = (0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0)


def kept_fraction(q: torch.Tensor, k: torch.Tensor, tau: float,
                  sink_blocks: tuple[int, int] = (0, 0),
                  sink_q: tuple[int, int] = (0, 0)) -> tuple[int, float, float]:
    """``(key blocks, share the threshold chose, share kept exact)`` for one ``tau``.

    The always-exact parts are counted the way the kernel adds them — the neighbouring key
    block, the sink range, every key for a sink query row — because tuning wants the total
    budget, not the share the threshold chose.
    """
    batch, tokens, heads, head_dim = q.shape
    n = -(-tokens // BLOCK)
    pad = n * BLOCK - tokens
    lengths = torch.full((n,), float(BLOCK), device=q.device)
    if pad:
        lengths[-1] = float(BLOCK - pad)
        q, k = (torch.cat([t, t.new_zeros(batch, pad, heads, head_dim)], dim=1)
                for t in (q, k))

    def pool(t: torch.Tensor, mean: bool) -> torch.Tensor:
        pooled = t.float().reshape(batch, n, BLOCK, heads, head_dim).sum(2)
        return pooled / lengths.view(1, -1, 1, 1) if mean else pooled

    kc = pool(k, mean=True)                                  # [B, N, H, D] summary keys
    kc_var = (kc - kc.mean(1, keepdim=True)).pow(2).mean(1)  # [B, H, D]
    centroid = pool(q, mean=True)
    q32, kc32 = centroid.permute(0, 2, 1, 3), kc.permute(0, 2, 1, 3)
    proxy = (q32 @ (kc32 - kc32.mean(2, keepdim=True)).transpose(-1, -2)).float()
    sigma = (centroid.float().pow(2).permute(0, 2, 1, 3)
             * kc_var.unsqueeze(2)).sum(-1).clamp_min(1e-6).sqrt()
    routed = proxy > tau * sigma.unsqueeze(-1)

    idx = torch.arange(n, device=q.device)
    exact = routed | ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1)
    exact[:, :, :, sink_blocks[0]:sink_blocks[1]] = True
    exact[:, :, sink_q[0]:sink_q[1], :] = True
    return n, float(routed.float().mean()), float(exact.float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", help="a .pt holding q and k from a real call")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--sink-rows", type=int, default=0,
                    help="rows every query attends exactly, as a sink range")
    ap.add_argument("--taus", type=float, nargs="*", default=list(TAUS))
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.capture:
        saved = torch.load(args.capture, map_location=device, weights_only=True)
        q, k = saved["q"].to(device), saved["k"].to(device)
        print(f"captured operands: {tuple(q.shape)}")
    else:
        shape = (1, args.tokens, args.heads, args.head_dim)
        q = torch.randn(shape, device=device, dtype=torch.bfloat16)
        k = torch.randn(shape, device=device, dtype=torch.bfloat16)
        print(f"random operands: {shape}  (the mechanism only, not a clip)")

    sinks = (0, -(-args.sink_rows // BLOCK)) if args.sink_rows else (0, 0)
    for tau in args.taus:
        blocks, routed, kept = kept_fraction(q, k, tau, sink_blocks=sinks)
        print(f"tau {tau:>4}: {kept:>6.1%} of {blocks} key blocks exact "
              f"({routed:.1%} chosen by the threshold)")


if __name__ == "__main__":
    main()
