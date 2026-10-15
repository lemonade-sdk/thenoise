#!/usr/bin/env python
"""Where the time inside one VAE case actually goes: time x work x bytes, per op.

``vae_bench.py`` says a case costs N milliseconds and M FLOPs; this says *which aten
ops* spent them, and whether each of those op types was compute-bound, bandwidth-bound
or neither. It is a probe, not a bench: run a case, read it, throw it away.

Two passes over the same case, because the two instruments get in each other's way:

1. ``torch.profiler`` (CPU + CUDA activities) for **time per aten op** — kineto
   attributes every kernel to the aten op that launched it, so a conv's MIOpen/Tensile
   kernel lands on ``aten::convolution`` instead of on an unreadable ``Cijk_...`` row.
2. A ``TorchDispatchMode`` for **bytes touched** (input + output element bytes of every
   op as executed), plus ``FlopCounterMode`` for **FLOPs per op**.

Joined per op, that yields the achieved TFLOPS and GB/s of every op *type* in the case —
which is what a codec-level TFLOPS number hides. A decoder at 12 TFLOPS can be a
compute-bound conv next to a 40 GB/s GroupNorm, or one genuinely bad conv kernel, and
those two want opposite responses.

The case is built by ``vae_bench``'s own fixtures, so the shapes are exactly the ones
the bench photographs. Random weights, no checkpoints, nothing loaded.

Usage
-----
    .venv/bin/python bench-scripts/probes/vae_kernels.py                    # decoders @1024
    .venv/bin/python bench-scripts/probes/vae_kernels.py --res 2048x2048 --top 14
    .venv/bin/python bench-scripts/probes/vae_kernels.py --models qwen_image --ops encode,decode
    .venv/bin/python bench-scripts/probes/vae_kernels.py --kernels          # raw kernel list too
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")   # silence kineto's USDT chatter

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vae_bench as vb  # noqa: E402

# ------------------------------------------------------------------ op categories
#
# Matched on the *whole* op name, never on a substring: ``t`` (the transpose alias) is a
# metadata-only op and every conv/bmm name contains a ``t``, so a substring pass files
# convolutions under "free" and the roofline quietly loses the convs it is about. An op
# that is not listed lands in ``other``, which prints as its own row: an unexplained
# category is much cheaper than a plausible one.

# Metadata-only ops: they change how a tensor looks and touch no elements, so counting
# their "bytes" as traffic invents bandwidth out of nothing.
FREE = frozenset("""
    view _reshape_alias _unsafe_view alias as_strided expand expand_as detach detach_
    narrow select slice split split_with_sizes chunk unbind permute transpose t squeeze
    unsqueeze size stride storage_offset numel dim is_contiguous is_floating_point
    device dtype requires_grad requires_grad_ set_ set_data resize_ resize_as_ shape
    lift_fresh lift_fresh_copy empty empty_like empty_permuted empty_strided new_empty
    new_empty_strided
""".split())

OPS = {
    "conv": """
        convolution _convolution _conv_forward conv_depthwise_3x3 conv1d conv2d conv3d
        conv_transpose1d conv_transpose2d conv_transpose3d convolution_overrideable
        cudnn_convolution cudnn_convolution_add_relu cudnn_convolution_relu
        cudnn_convolution_transpose miopen_convolution miopen_convolution_add
        miopen_convolution_add_relu miopen_convolution_depthwise
        miopen_convolution_transpose mkldnn_convolution
    """.split(),
    "matmul": """
        mm bmm addmm addbmm baddbmm mv matmul linalg_matmul linear _int_mm
        _addmm_activation scaled_dot_product_attention _scaled_dot_product_attention_math
        _scaled_dot_product_flash_attention _scaled_dot_product_flash_attention_for_cpu
        _scaled_dot_product_efficient_attention _scaled_dot_product_cudnn_attention
        _flash_attention_forward _efficient_attention_forward _cudnn_attention_forward
    """.split(),
    "norm": """
        group_norm native_group_norm batch_norm native_batch_norm
        _native_batch_norm_legit _native_batch_norm_legit_no_training native_layer_norm
        layer_norm norm vector_norm linalg_vector_norm normalize _fused_rms_norm rms_norm
        local_response_norm
    """.split(),
    "softmax": "softmax _softmax log_softmax _log_softmax".split(),
    "activation": """
        silu relu relu6 leaky_relu prelu gelu hardswish hardsigmoid hardtanh mish elu
        softplus log_sigmoid sigmoid tanh threshold
    """.split(),
    "resample": """
        interpolate upsample_nearest1d upsample_nearest2d upsample_nearest3d
        upsample_bilinear2d upsample_bicubic2d upsample_linear1d upsample_trilinear3d
        _upsample_nearest_exact1d _upsample_nearest_exact2d _upsample_nearest_exact3d
        _upsample_bilinear2d_aa _upsample_bicubic2d_aa avg_pool1d avg_pool2d avg_pool3d
        adaptive_avg_pool1d adaptive_avg_pool2d adaptive_avg_pool3d max_pool1d max_pool2d
        max_pool3d adaptive_max_pool2d pixel_shuffle pixel_unshuffle space_to_depth
        depth_to_space
    """.split(),
    # Element traffic that computes nothing: copies, casts, concatenations, padding,
    # gathers, and the initialisers that fill a fresh buffer.
    "copy/layout": """
        copy_ _to_copy to contiguous clone cat stack hstack vstack dstack column_stack
        row_stack pad constant_pad_nd reflection_pad1d reflection_pad2d
        replication_pad1d replication_pad2d replication_pad3d zeropad2d unfold fold
        repeat repeat_interleave tile flip roll gather index_select index_put_
        index_copy scatter scatter_ masked_fill masked_fill_ masked_select fill_ zero_
        ones zeros ones_like zeros_like new_zeros new_ones new_full full full_like arange
        linspace eye rand randn rand_like randn_like randint uniform_ normal_
        im2col col2im
    """.split(),
    "reduce": """
        sum mean var var_mean std std_mean amax amin argmax argmin min max median prod
        cumsum cumprod any all count_nonzero
    """.split(),
    "elementwise": """
        add add_ addcdiv addcmul sub sub_ rsub mul mul_ div div_ true_divide
        floor_divide where clamp clamp_min clamp_max minimum maximum neg abs sign
        reciprocal sqrt rsqrt pow exp exp2 log log2 log10 log1p erf sin cos tan asin
        acos atan atan2 sinh cosh trunc floor ceil frac logical_and logical_or
        logical_not bitwise_and bitwise_or eq ne lt le gt ge isnan isinf lerp
    """.split(),
}
CATEGORY = {name: cat for cat, names in OPS.items() for name in names}


def op_key(func) -> str:
    """``aten.convolution.default`` / ``aten.convolution`` -> ``aten::convolution``.

    The profiler's row keys, the flop counter's op keys and a dispatch mode's ``func``
    are three spellings of one op name; the join below needs them to agree.
    """
    parts = str(func).replace("::", ".").split(".")
    return f"aten::{parts[1]}" if len(parts) > 1 and parts[0] == "aten" else str(func)


def category(op: str) -> str:
    """``aten::convolution`` and ``aten::convolution.default`` both -> ``conv``.

    ``layout (free)`` means metadata only: it is listed so a row is never silent, but it
    is not traffic and must not be summed with the categories that are.
    """
    name = op.replace("aten::", "").split(".")[0].split("(")[0]
    if name in FREE:
        return "layout (free)"
    return CATEGORY.get(name, "other")


class Bytes(TorchDispatchMode):
    """Bytes touched per aten op: every input read plus every output written."""

    def __init__(self):
        super().__init__()
        self.totals: dict[str, float] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        outs = func(*args, **kwargs)
        flat = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
        out_list = outs if isinstance(outs, (tuple, list)) else [outs]
        flat += [o for o in out_list if isinstance(o, torch.Tensor)]
        # Three spellings of one op name (see ``op_key``): normalise before counting.
        key = op_key(func)
        self.totals[key] = self.totals.get(key, 0) + sum(
            t.numel() * t.element_size() for t in flat)
        return outs


class TileSpy:
    """Record the score-matrix sizes a run asks for, without changing its answers.

    ``score_tile_rows`` is where ``utils.attention`` decides between scoring the whole
    matrix and tiling the queries, so the Ns it is handed say which side of the
    ``SCORE_LIMIT_BYTES`` line this resolution lands on — the thing that makes the top
    of the ladder disproportionately expensive for an 8x codec.
    """

    def __init__(self, state: list):
        self.state = state

    def __enter__(self):
        import thenoise.utils.attention as attn_mod

        self.mod, self.orig = attn_mod, attn_mod.score_tile_rows
        state = self.state

        def spy(n):
            state.append(int(n))
            return self.orig(n)

        attn_mod.score_tile_rows = spy
        return self

    def __exit__(self, *exc):
        self.mod.score_tile_rows = self.orig
        return False


def profile_case(fx, op, res, device, dtype, iters, shapes):
    """One case, twice: kineto for time per op, dispatch modes for bytes and FLOPs."""
    from torch.profiler import ProfilerActivity, profile

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
    vae = captured["vae"]
    latent, detail = case.latent, case.detail
    gpu = device == "cuda" and torch.cuda.is_available()
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if gpu else [])
    tiles: list[int] = []

    with torch.no_grad():
        for _ in range(2):                       # per-shape conv algorithm search first
            case.run()
        if gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        with TileSpy(tiles), profile(activities=acts, record_shapes=shapes,
                                     with_stack=False) as prof:
            for _ in range(iters):
                case.run()
            if gpu:
                torch.cuda.synchronize()

    rows = prof.key_averages(group_by_input_shape=False)
    per_op: dict[str, float] = {}
    kernels: dict[str, tuple[float, int]] = {}
    for row in rows:
        self_us = float(getattr(row, "self_device_time_total", 0.0))
        if self_us > 0:
            key = row.key
            if key.startswith("aten::"):
                per_op[key] = per_op.get(key, 0.0) + self_us
            else:
                kernels[key] = (self_us, int(row.count))
    on_device = bool(per_op)
    if not on_device:                            # CPU run: wall time per op instead
        for row in rows:
            if row.key.startswith("aten::") and row.self_cpu_time_total > 0:
                per_op[row.key] = per_op.get(row.key, 0.0) + float(row.self_cpu_time_total)

    bytes_mode = Bytes()
    with bytes_mode, FlopCounterMode(display=False) as counter:
        case.run()
    if gpu:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        peak_res = torch.cuda.max_memory_reserved() / 2**30
    else:
        peak = peak_res = 0.0

    del vae, case
    gc.collect()
    if gpu:
        torch.cuda.empty_cache()
    return dict(latent=latent, detail=detail,
                time={k: v / 1000 / iters for k, v in per_op.items()},
                bytes={k: v / iters for k, v in bytes_mode.totals.items()},
                flops={op_key(k): float(v)
                       for k, v in counter.get_flop_counts()["Global"].items()},
                kernels=kernels, peak=peak, peak_res=peak_res, on_device=on_device,
                tiles=sorted(set(tiles)))


def report(name, p, top, show_kernels):
    total = sum(p["time"].values())
    unit = "ms GPU" if p["on_device"] else "ms CPU wall (no device activity)"
    print(f"\n{name}  latent {p['latent']}  {p['detail']}")
    head = f"  {total:8.1f} {unit}"
    if p["on_device"]:
        head += f" | peak {p['peak']:.2f} GiB allocated, {p['peak_res']:.2f} reserved"
    print(head)
    if p["tiles"]:
        from thenoise.utils.attention import SCORE_LIMIT_BYTES, score_tile_rows

        for n in p["tiles"]:
            rows = score_tile_rows(n)
            mode = "whole score matrix" if rows >= n else f"TILED, rows={rows}"
            print(f"  attention N={n}: {mode}  "
                  f"({2 * n * n / 2**30:.2f} GiB of bf16 scores, "
                  f"limit {SCORE_LIMIT_BYTES / 2**30:.0f} GiB)")

    # Time, FLOPs and bytes are counted by three instruments that spell op names
    # differently (the profiler attributes a conv's time to the backend op that
    # launched its kernel, the flop counter to ``convolution``). Joining them per op is
    # therefore unreliable — join them per *category*, which every spelling maps into.
    cat: dict[str, list] = {}

    def bucket(op):
        return cat.setdefault(category(op), [0.0, 0.0, 0.0])

    for op, ms in p["time"].items():
        bucket(op)[0] += ms
    for op, f in p["flops"].items():
        bucket(op)[1] += f
    for op, b in p["bytes"].items():
        bucket(op)[2] += b

    def line(label, ms, f, b, pct_of=None):
        tf = f / (ms / 1000) / 1e12 if ms and f else 0.0
        gb = b / (ms / 1000) / 2**30 if ms and b else 0.0
        pct = 100 * ms / pct_of if pct_of and ms else 0.0
        print(f"  {label[:32]:32s} {ms:8.1f} {pct:5.1f} {tf:7.1f} {gb:7.0f}")

    print(f"  {'category':32s} {'ms':>8s} {'%':>5s} {'TFLOPS':>7s} {'GiB/s':>7s}")
    for c, (ms, f, b) in sorted(cat.items(), key=lambda kv: -kv[1][0]):
        line(c, ms, f, b, total)
    print(f"  {'total':32s} {total:8.1f} {100.0:5.1f} "
          f"{sum(v[1] for v in cat.values()) / (total / 1000) / 1e12:7.1f} "
          f"{sum(v[2] for v in cat.values()) / (total / 1000) / 2**30:7.0f}")

    print(f"  {'slowest ops (time only)':32s} {'ms':>8s} {'%':>5s}")
    for op, ms in sorted(p["time"].items(), key=lambda kv: -kv[1])[:top]:
        if ms / total >= 0.004:
            print(f"  {op.replace('aten::', '')[:32]:32s} {ms:8.1f} {100 * ms / total:5.1f}")
    if show_kernels and p["kernels"]:
        ktotal = sum(v[0] for v in p["kernels"].values()) / 1000
        print(f"  {'top kernels':32s} {'ms':>8s} {'%':>5s} {'count':>7s}")
        for key, (us, count) in sorted(p["kernels"].items(), key=lambda kv: -kv[1][0])[:top]:
            print(f"  {key[:32]:32s} {us / 1000:8.1f} {100 * us / 1000 / ktotal:5.1f} "
                  f"{count:7d}")


def check_categories(device: str) -> int:
    """Ask the codecs what they actually dispatch, and fail on anything unclassified.

    The category table is the only reason these probes are readable, and it rots
    silently: a rename moves a fat op into ``other``, and an over-loose rule can move it
    into ``layout (free)`` — which is what used to happen here, because ``t`` is a
    metadata-only op and ``convolution`` contains a ``t``. So the fixtures are built at
    64x64 on the CPU (no GPU, no weights) and every op they dispatch is printed with the
    category it lands in. Unexplained is fine; unlisted is not.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    class Collect(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops: set[str] = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.add(op_key(func))
            return func(*args, **(kwargs or {}))

    # The traps, in both directions.
    assert category("aten::convolution") == "conv", category("aten::convolution")
    assert category("aten::t") == "layout (free)"
    assert category("aten::native_group_norm") == "norm"
    assert category("aten::bmm") == "matmul"

    ctx = vb.Ctx(device=device, dtype=torch.bfloat16)
    res = vb.Res(64, 64)
    seen: dict[str, set[str]] = {}
    cases = 0
    for fx in vb.FIXTURES:
        for op in fx.ops:
            try:
                case = fx.build(ctx, res, op)         # setup is not what gets measured
                with Collect() as col:
                    case.run()
            except Exception as exc:                            # noqa: BLE001
                print(f"  {fx.model}/{op}: FAILED {type(exc).__name__}: {str(exc)[:70]}")
                continue
            cases += 1
            for o in col.ops:
                seen.setdefault(o, set()).add(f"{fx.model}/{op}")

    print(f"{len(seen)} distinct ops over {cases} cases "
          f"({device} {res}, seeded random weights):")
    for o in sorted(seen):
        who = ",".join(sorted(seen[o]))
        print(f"  {o.replace('aten::', ''):32s} {category(o):14s} {who[:56]}")
    bad = {o: v for o, v in seen.items() if category(o) == "other"}
    if bad:
        print(f"\nUNCLASSIFIED ({len(bad)}): add them to OPS in {Path(__file__).name} — "
              f"a category that is guessed is a roofline that is wrong")
        for o in sorted(bad):
            print(f"  {o.replace('aten::', ''):32s} {','.join(sorted(bad[o]))[:56]}")
        return 1
    print("\nevery dispatched op has a category")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-op cost of a VAE case.")
    ap.add_argument("--models", default="flux,qwen_image21,qwen_image",
                    help="codec filter (default %(default)s)")
    ap.add_argument("--ops", default="decode", help="encode,decode,features")
    ap.add_argument("--res", default="1024x1024", help="one resolution (default %(default)s)")
    ap.add_argument("--dtype", default="bf16", choices=sorted(vb.DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=1, help="runs inside the profiler")
    ap.add_argument("--top", type=int, default=10, help="ops to list per case")
    ap.add_argument("--kernels", action="store_true", help="also list raw GPU kernels")
    ap.add_argument("--shapes", action="store_true",
                    help="record input shapes (slower; the per-op split is the same)")
    ap.add_argument("--check", action="store_true",
                    help="print which category every op the codecs dispatch lands in "
                         "(CPU, no GPU needed) and exit non-zero if any is unclassified")
    args = ap.parse_args()

    if args.check:
        sys.exit(check_categories("cpu"))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no cuda/rocm device visible — run this on the GPU box")
    res = vb.Res.parse(args.res)
    ops = tuple(o.strip() for o in args.ops.split(",") if o.strip())
    fixtures = vb.select(args.models)
    dtype = vb.DTYPES[args.dtype]
    gpu = args.device == "cuda" and torch.cuda.is_available()

    print(f"{torch.cuda.get_device_name(0) if gpu else args.device} | "
          f"torch {torch.__version__} | {args.dtype} | {res} | ops {','.join(ops)}")
    for fx in fixtures:
        for op in fx.ops:
            if op not in ops:
                continue
            report(f"{fx.model}/{op}",
                   profile_case(fx, op, res, args.device, dtype, args.iters, args.shapes),
                   args.top, args.kernels)
    print("\nTFLOPS and GiB/s are per op type over the whole case. A conv category near half\n"
          "the machine's peak with norm/activation at tens of GiB/s is a bandwidth-diluted\n"
          "codec, which is a different problem from one slow conv kernel.")


if __name__ == "__main__":
    main()
