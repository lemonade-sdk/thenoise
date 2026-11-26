#!/usr/bin/env python
"""Attribute one compiled block's time to the work inside it.

Where ``block_bench.py`` is a camera — every block, one protocol, no switches — this is a
lab: one block, several arms, timed against each other. It is for the question a snapshot
cannot answer: a block that costs more per FLOP than its neighbours is photographed
perfectly by the camera and explained by nothing in it.

An arm is a patch held over the block the camera itself builds (``block_bench``'s own
fixture, weights and inputs, reached by spying on ``new_block``), so what is measured is
the cost *inside* the compiled region — which is the only place a change would live. Each
arm re-traces the block, and that is intended: every arm is its own Inductor graph.

Two numbers per arm, and the second is the one to read:

* **gain** — what the arm gave back, as a share of the block's time. Each arm is measured
  in a process of its own against a baseline timed there before its patch was applied, and
  a second baseline timed after it is printed as ``drift%``: where the two baselines
  disagree, the arm changed the block and its gain is not a comparison. Interleaving one
  baseline through a whole ladder used to be how a drifting box was cancelled, and it is
  not enough for this — see ``spawned_ladder``.
* **ops** — the aten ops one call of the arm issues, against its baseline's. A gain is only
  a saving if the arm is still the compiled block; an arm that broke the graph saves time
  by giving up compilation, and that is visible here and nowhere else in a child, which
  cannot profile device activity.
* **rate** — the FLOPs the arm removed divided by the time it saved. A term that comes
  back at 40 TF was paid for at 40 TF, and no fusion will make it cheaper. A term whose
  FLOPs cost 3 TF bought them at a thirteenth of the rate the rest of the block pays, and
  that is where to work.

Rate only means that while the removed work is additive and the rest is untouched, so it
is printed beside the numerics: an arm that changes the answer is still a timing fact
about the kernels it removes, but arms are then timing different functions.

Sections
--------
``ablate``   the arm ladder above.
``sdpa``     the block's attention operands in eager, with and without their mask, under
             each backend forced: which answers, which refuses, what each costs. Eager,
             because the backend is chosen per call there — a compiled graph bakes the
             choice in at trace time, so forcing one inside an arm proves nothing.
``gemm``     the block's own largest GEMMs — taken from its profile, so any block offers
             them without anyone listing them — each in the forms the same call could take
             and then with TunableOp choosing the kernel: which of shape, operand layout,
             epilogue and kernel selection a slow GEMM's cost belongs to. Under each row, its
             operand pitch, since a row pitch that is a power of two is its own disease.
             ``--lt-algos`` widens the search; the tuned kernels land in the file named by
             ``--tune-out``, which is a per-machine configuration a build could carry.
``profile``  one arm's kernel table, bucketed into attention / GEMM / copy / pointwise.
``buckets``  the same table for several arms, subtracted from the baseline's: which kind
             of work an arm's gain came from, and which kernels appeared or vanished. Give
             it ``--arms``, since every arm costs a trace.
``shapes``   one arm's GEMM and attention calls with the shapes they were given, their
             device time and their implied rate: which projection and which attention
             call, not just which kernel.

Every run opens with a ``graph`` line saying how many of the block's attentions the
compiled graph kept fused, how many kernels it launched and where its elementwise time
lives. Read it first: an Inductor graph is cached under the graph, not under the backends
available when it was written, so a block once compiled without
``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`` keeps serving its math-path attention after
the variable is set — and an uncompiled block prints the same table with ATen kernels in
place of triton ones. ``--fresh-cache`` compiles into an empty cache instead.

Usage
-----
    .venv/bin/python bench-scripts/block_probe.py --list-arms
    .venv/bin/python bench-scripts/block_probe.py --tokens 16384
    .venv/bin/python bench-scripts/block_probe.py --tokens 16384 --section sdpa
    .venv/bin/python bench-scripts/block_probe.py --tokens 16384 --section gemm \
        --lt-algos 16
    .venv/bin/python bench-scripts/block_probe.py --tokens 9216 --arms prompt_mask_operand,sdpa_eager
    .venv/bin/python bench-scripts/block_probe.py --tokens 16384 --section buckets \
        --arms open_graph,sdpa_eager,rope_bf16
    .venv/bin/python bench-scripts/block_probe.py --tokens 16384 --section profile
    .venv/bin/python bench-scripts/block_probe.py --tokens 16384 --weights int8
    .venv/bin/python bench-scripts/block_probe.py --block qwen_image21/block --tokens 4096 \
        --section gemm
    .venv/bin/python bench-scripts/block_probe.py --block flux2_klein9b/double \
        --arms pitch_pad
    .venv/bin/python bench-scripts/block_probe.py --block h3/av_block --tokens 40114 \
        --arms dense_attention,adaln_ck,qk_rope_ck,sol_producer

Any case ``block_bench`` can build can be probed: pass its ``model/block``, and the sections
run with the GEMM list read off that block's own profile and four layout arms standing in for
an arm set — ``pitch_pad``, widening the operand rows whose pitch the sweep found the penalty
on; ``weight_transposed`` and ``weight_plain_store``, a pair: the first stores a wide-output
projection's weight the other way round, same numbers, which is what the engine now does as
the weights land, so it reads flat and the second puts the checkpoint's store back to price
it; and ``transposed_problem``, answering a narrow-output projection as ``Cᵀ`` and leaving its
consumers to cope. All three select on module widths alone, so every block is asked the same
questions the LTX findings came from.

What a block needs to be asked sharper questions is an arm set of its own, and a FLOPs model
decomposing its terms for the rate column; neither is needed to time it or to read its
kernels. LTX has both, because its terms are the ones that were worth ablating one at a
time: the arms are the part that does not transfer, which is the point of writing them down.

``h3/av_block`` is the audiovisual trunk's block over a pack the adapter would really build,
and its arms are the fused-kernel candidates: what ``comfy_kitchen`` offers against what
Inductor already does with the same terms when they are inside a compiled region. Isolated
elementwise numbers are worth nothing for that question — ``ck.rms_adaln`` measured 10x
faster and 6x smaller on its own and came out 1.7% *slower* here — so nothing about this
trunk should be adopted from a measurement taken outside it. Every arm is compared against
the baseline in its own process, and the numerics line says how far the arm's answer is from
the shipped one, which is also the only thing that catches a rope table built with the wrong
sign dialect.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn

import block_bench as bb
from thenoise.dit.ltx_video import transformer as ltx
from thenoise.dit.ltx_video.rope import AVRope
from thenoise.utils import attention as attention_mod
from thenoise.dit.quantized import PITCH_MIN_BYTES, PITCH_MIN_ROWS, PITCH_PAD_ELEMENTS
from thenoise.utils.layout import TRANSPOSE_MIN_RATIO
from thenoise.utils.attention import eager_attention
from thenoise.utils.dynamo import apply_recompile_limit
from thenoise.dit.h3.sparse import (BLOCK as SOL_BLOCK, sol_plan,
                                     sol_sinks as sink_policy)

#: Below this a delta is smaller than the box's own drift over a ladder.
FLOOR_PCT = 1.0
#: What the ablation ladder charges for nothing, so an arm can be read as "free".
NAME_W = 24


# --------------------------------------------------------------------- patching
@contextlib.contextmanager
def patched(obj, attr, value):
    """One attribute swapped for the duration, including one that did not exist."""
    had = hasattr(obj, attr)
    old = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, attr, old)
        elif attr in vars(obj):
            delattr(obj, attr)


@contextlib.contextmanager
def patches(items):
    with contextlib.ExitStack() as stack:
        for obj, attr, value in items:
            stack.enter_context(patched(obj, attr, value))
        yield


def zero_out(x: torch.Tensor, **_kw) -> torch.Tensor:
    """A stand-in for a whole sub-path that leaves the stream's shape alone.

    Every attention and FFN of this block is width-preserving, so the same stand-in
    covers all of them and the block's remaining operands keep their real layouts.
    """
    return torch.zeros_like(x)


def _now() -> float:
    return time.perf_counter()


def fresh_cache() -> str:
    """Point Inductor at an empty cache, so nothing measured here was compiled elsewhere.

    Returns the directory, which is left on disk for inspection.
    """
    import tempfile

    path = tempfile.mkdtemp(prefix="probe-inductor-")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = path
    return path


def _cache_dir() -> str:
    from torch._inductor.codecache import cache_dir

    return str(cache_dir())


_FUSED = ("aten::_flash_attention_forward", "aten::_efficient_attention_forward",
          "aten::_cudnn_attention_forward")


def raise_recompile_limits() -> None:
    """``apply_recompile_limit`` plus the two accumulated counters, which it leaves alone.

    A ladder recompiles the same frames far more than a render ever does, and Dynamo's
    ``accumulated_*`` counters — 256 by default, global across code objects — end the
    tracing session for good, after which the block silently runs eager. Raising them is
    cheap insurance against that confound; it is not the explanation of the degradation
    ``graph_check`` catches, which happens with these raised and is caused by tracing the
    arms themselves.
    """
    apply_recompile_limit()
    for name in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        torch._dynamo.config._config[name].default = 100_000


def graph_check(run: Callable[[], object], attentions: Optional[int],
                label: str) -> float:
    """One profiled call, read for whether the block is the compiled thing being timed.

    Two failures are worth catching here, and both were silent in runs of this script:

    * an attention that is not fused, which is what a graph compiled while flash was
      refused holds — the artifact outlives the environment variable, because the cache
      keys on the graph and not on the backends available when it was written;
    * no compiled region at all, which is what Dynamo falls back to when a patched callee
      poisons the frame. Silent, and it reads as a slow block: locally, tracing
      ``sdpa_eager`` — whose ``eager_attention`` is ``@torch._dynamo.disable()``d — left
      the block uncompiled for every measurement after it, 158 launches becoming 263 and
      triton's 14 ms becoming 0.5 ms with 24 ms of ``at::native`` in their place; on
      gfx1151 a 15-arm ladder measured its own baseline at 627 ms against the 498 ms the
      same block gives in a short run, 1 GiB heavier.

    Unfused attention is counted from the fused backends' own op records; compiled-ness
    from the shape of the kernel table, since a compiled block spends its elementwise time
    in triton and an uncompiled one in ``at::native``.

    Returns how long the block took to trace, which is the baseline's compile cost: this is
    the call that traces it, so the ladder would otherwise report a cache hit.
    """
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        t0 = _now()
        run()
        trace_s = _now() - t0
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
            torch.cuda.synchronize()
    counts = {e.key: e.count for e in prof.key_averages()}
    fused = sum(counts.get(key, 0) for key in _FUSED)
    decomposed = sum(counts.get(key, 0) for key in ("aten::bmm", "aten::_softmax"))
    kernels = _dedupe([(e.key, e.self_device_time_total / 1000, e.count)
                       for e in prof.key_averages()
                       if e.self_device_time_total > 0 and _kernel(e.key)])
    triton = sum(ms for key, ms, _ in kernels if key.startswith("triton"))
    aten_ew = sum(ms for key, ms, _ in kernels if "at::native" in key)
    launches = sum(c for _, _, c in kernels)
    fused_note = f"{fused}/{attentions} fused" if attentions else f"{fused} fused"
    print(f"  graph {label}: {fused_note}, {decomposed} bmm/softmax,"
          f" {launches:.0f} launches, triton {triton:.1f} ms, at:: elementwise"
          f" {aten_ew:.1f} ms   cache {_cache_dir()}"
          + ("" if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
             else ", AOTriton experimental OFF"))
    if not kernels:
        print("    WARNING the profile captured no device kernels, so this line measures "
              "nothing: CUPTI serves one client per device and something else holds it.")
    elif fused < attentions:
        print("    WARNING an unfused attention. With the environment set that is a stale "
              "artifact: rerun with --fresh-cache.")
    if aten_ew > 4 * triton:
        print("    WARNING most of the elementwise time is in ATen kernels rather than "
              "triton: this block is running uncompiled, so it is not the shipped block.")
    return trace_s


# ------------------------------------------------------------------ the fixture
@dataclass
class Fixture:
    """The camera's case for one block, plus the terms its FLOPs are made of."""

    blk: nn.Module
    run: Callable[[], object]
    seq: int
    detail: str
    terms: dict[str, float]
    attentions: int


def ltx_fixture(ctx: bb.Ctx, tokens: int, *, i2v: bool) -> Fixture:
    """The camera's ``ltx/av`` case, with the block instance in hand.

    The block is captured rather than rebuilt: ``_ltx_case`` makes it through
    ``new_block``, so spying there hands over the exact instance the camera's ``run``
    closure holds — same seeded weights, same seeded inputs, one construction, no way for
    this script to drift from the camera.
    """
    captured: list[nn.Module] = []
    original = bb.new_block

    def spy(c, name, cls, *args, **kwargs):
        blk = original(c, name, cls, *args, **kwargs)
        captured.append(blk)
        return blk

    bb.new_block = spy
    try:
        role = "av_i2v" if i2v else "av"
        case = bb._ltx_case(ctx, tokens, role, token_timestep=i2v)
    finally:
        bb.new_block = original
    blk, = captured

    video = (bb.LTX_VDIM // ltx.VIDEO_HEAD_DIM, ltx.VIDEO_HEAD_DIM)
    audio = (bb.LTX_ADIM // ltx.AUDIO_HEAD_DIM, ltx.AUDIO_HEAD_DIM)
    a_tokens = case.seq - tokens
    terms = {}
    attentions = 0
    for name, mod, shape, queries, keys in (
            ("self_video", blk.attn1, video, tokens, tokens),
            ("self_audio", blk.audio_attn1, audio, a_tokens, a_tokens),
            ("prompt_video", blk.attn2, video, tokens, ctx.txt),
            ("prompt_audio", blk.audio_attn2, audio, a_tokens, ctx.txt),
            ("a2v", blk.audio_to_video_attn, audio, tokens, a_tokens),
            ("v2a", blk.video_to_audio_attn, audio, a_tokens, tokens),
            ("ffn_video", blk.ff, None, tokens, 0),
            ("ffn_audio", blk.audio_ff, None, a_tokens, 0)):
        if shape is None:
            terms[name] = bb.gemm_flops((bb.gemm_params(mod), queries))
        else:
            attentions += 1
            query, key_values = bb._ltx_projections(mod)
            terms[name] = (bb.gemm_flops((query, queries), (key_values, keys))
                           + bb.attn_flops(*shape, ((queries, keys),)))
    return Fixture(blk, case.run, case.seq, case.detail, terms, attentions)


#: The trunk's own geometry, read where the camera reads it: 56x128 over a 5376-wide
#: stream, the checkpoint's adaln and rope shapes, nothing invented here.
H3_FRAMES = tuple(range(5, 125, 17))        # the codec's valid clip lengths


def h3_pack(tokens: int):
    """The pack whose sequence is nearest ``tokens``, off the release's two canvases."""
    from thenoise.dit.h3.layout import Layout
    from thenoise.vae.h3 import TemporalPlan
    from thenoise.vae.h3_audio import LATENTS_PER_SECOND

    best = None
    for _, width, height, frames in SOL_PACKS:
        plan = TemporalPlan.of_frames(frames)
        layout = Layout.av(text_len=SOL_TEXT, latent_t=plan.latent_t,
                           latent_h=-(-height // 16), latent_w=-(-width // 16),
                           audio_t=max(1, round(frames / SOL_FPS * LATENTS_PER_SECOND)))
        gap = abs(layout.seq_len - tokens)
        if best is None or gap < best[0]:
            best = (gap, layout)
    return best[1]


def h3_fixture(ctx: bb.Ctx, tokens: int) -> Fixture:
    """The audiovisual trunk's block, over a pack a request would really build.

    The camera has ``h3_image/block``, which is a still: a still never routes, so there is
    no plan to hand it and the question this lab asks — what is a fused kernel worth inside
    a routed block — cannot be asked of it. The block is the camera's own construction,
    reached by the same spy ``ltx_fixture`` uses.
    """
    from thenoise.dit.h3.checkpoint import RELAY_STORES
    from thenoise.dit.h3.trunk import DiTBlock, _spans
    from thenoise.utils.rope import split_half_rope_axes

    layout = h3_pack(tokens)
    params = bb.h3_params()
    captured: list[nn.Module] = []
    original = bb.new_block

    def spy(c, name, cls, *args, **kwargs):
        blk = original(c, name, cls, *args, **kwargs)
        captured.append(blk)
        return blk

    steps = layout.timesteps(video=0.5, audio=0.5)
    bb.new_block = spy
    try:
        blk = bb.new_block(ctx, "h3/av_block", DiTBlock, params, ctx.dtype,
                           relay_stores=RELAY_STORES)
        x = bb.randn(ctx, 1, layout.seq_len, params.hidden_size)
        t_emb = bb.randn(ctx, len(steps.values), params.adaln_dim)
        rows = layout.modulation_rows(steps)
        # The frequencies come off the checkpoint in a real run; only their count is a shape.
        inv_freq = 1.0 / 1000 ** (torch.arange(params.rope_inv_freq, device=ctx.tdev,
                                               dtype=torch.float32) / params.rope_inv_freq)
        pos = layout.position_ids.to(ctx.tdev, torch.float32)
        cos, sin = (one.to(ctx.dtype) for one in split_half_rope_axes(inv_freq)(pos))
        spans = _spans(rows, blk.adaln_proj(t_emb))
    finally:
        bb.new_block = original

    plan = sol_plan(layout=layout, tokens=layout.seq_len, dtype=ctx.dtype,
                    head_dim=SOL_HEAD_DIM, device=ctx.tdev, sigma=0.5,
                    sample_sigmas=torch.linspace(1.0, 0.0, 5, device=ctx.tdev),
                    method="optimized")
    if plan is None:
        raise SystemExit("no sol plan: this device cannot route, so the routed arms of this "
                         "fixture have nothing to be measured against")
    rows_all = layout.seq_len

    def run():
        return blk(x, spans, (cos, sin), None, plan)

    inner = params.num_heads * params.head_dim
    terms = {
        "self_proj": bb.gemm_flops((bb.gemm_params(blk.attn.qkv_proj), rows_all),
                                   (bb.gemm_params(blk.attn.out_proj), rows_all)),
        "self_attn": bb.attn_flops(params.num_heads, params.head_dim,
                                   ((rows_all, rows_all),)),
        "ffn": bb.gemm_flops((bb.gemm_params(blk.mlp), rows_all)),
    }
    detail = f"{layout.latents} latent, txt={SOL_TEXT}"
    # No attention is counted as expected inside the graph: sol_attention is a
    # ``torch._dynamo.disable`` island, so its absence there is the shipped design and not a
    # stale artifact.
    return Fixture(blk, run, rows_all, detail, terms, 0)


def camera_fixture(ctx: bb.Ctx, name: str, tokens: int,
                   scenario: Optional[str]) -> Fixture:
    """Any case the camera can build, with its block instance in hand.

    Every camera fixture makes its block through ``new_block``, so the same spy that
    ``ltx_fixture`` uses hands over the instance its ``run`` closure holds: same seeded
    weights and inputs, one construction, no way for this script to drift from the
    measurement the camera took.

    What comes back has no FLOPs model and no attention count. The camera holds one
    number for a block it does not decompose, and inventing terms to fill the column
    would be worse than the blank.
    """
    model, _, block = name.partition("/")
    found = [fx for fx in bb.FIXTURES if fx.model == model
             and (not block or fx.block == block)]
    if not found:
        raise SystemExit(f"no camera case named {name!r}: block_bench.py --list has them")
    fx = found[0]
    which = scenario or fx.scenarios[0]
    captured: list[nn.Module] = []
    original = bb.new_block

    def spy(c, bname, cls, *args, **kwargs):
        blk = original(c, bname, cls, *args, **kwargs)
        captured.append(blk)
        return blk

    bb.new_block = spy
    try:
        case = fx.build(ctx, tokens, which)
    finally:
        bb.new_block = original
    if not captured:
        raise SystemExit(f"{name}: its fixture builds no block through new_block")
    return Fixture(blk=captured[0], run=case.run, seq=case.seq,
                   detail=f"{fx.block} {which}: {case.detail}", terms={}, attentions=0)


#: The arm's defaults are the engine's own rule, so a flat gain means the engine already
#: does this and a nonzero one means it does not; the flags are for asking about the bounds.
PAD_MIN_PITCH = PITCH_MIN_BYTES


PITCH_NOTE = ("every projection whose input rows are {min_pitch} bytes apart or further, "
              "fed from a buffer eight elements wider where the operand has at least "
              "{min_rows} rows: rows 32 KiB apart cost about half the rate the same block "
              "gets elsewhere, and short streams lose to the same change")
WTRANSPOSE_NOTE = ("every projection at least {min_ratio}x wider out than in, its weight "
                   "stored the other way round, same numbers: the engine relays these as the "
                   "weights land, so a flat row at maxabs 0 is the expected reading and "
                   "weight_plain_store is the arm that prices the relay")
WEIGHT_PLAIN_NOTE = ("every projection the engine relays, its weight put back the way a "
                     "checkpoint stores it: the same numbers read strided again, so what "
                     "this arm loses is what the load-time relay earns")
TRANSPOSED_NOTE = ("every projection reducing to {max_ratio}x or less of its width "
                   "answering for the transposed problem, C transposed, its consumers left "
                   "to read it: for those shapes the tile problem is better the other way "
                   "round and the pitch has nothing to do with it")

#: The bounds the layout arms select on. Each is a measurement, not a law: the pitches
#: taken from the sweep over every block the camera can build, the ratios from the rows
#: that moved in it. ``LayoutRules`` carries them so an arm set takes one argument and a
#: flag can move one bound without reaching through the script.
TRANSPOSED_MAX_RATIO = 0.5    # anima's winner reduced 8192 to 2048, LTX's audio rows too


@dataclass(frozen=True)
class LayoutRules:
    """The bounds the layout arms select on, as one required argument."""

    min_pitch: int = PAD_MIN_PITCH
    min_rows: int = PITCH_MIN_ROWS
    transpose_min_ratio: float = TRANSPOSE_MIN_RATIO
    transposed_max_ratio: float = TRANSPOSED_MAX_RATIO


def _projections(blk: nn.Module, ratio: Callable[[int, int], bool]) -> list[nn.Module]:
    """The full-precision 2-D projections of ``blk`` that ``ratio(in, out)`` selects.

    Floating-point weights only: a quantized weight is a different tensor with a different
    kernel behind it, and none of these rules were measured on one.
    """
    return [mod for mod in blk.modules()
            if getattr(mod, "weight", None) is not None
            and mod.weight.dim() == 2 and mod.weight.is_floating_point()
            and ratio(mod.weight.shape[1], mod.weight.shape[0])]


def pitch_pad_patches(blk: nn.Module, rules: LayoutRules) -> list:
    """Wrap every projection whose input rows are at least ``rules.min_pitch`` bytes apart.

    What the engine-wide sweep says the rule is: a GEMM's rate follows its reduction length,
    because that is its activation's row pitch, and rows 32768 bytes apart are charged at
    about half what the same block charges anywhere else — LTX's and krea2's 16384-wide FFN
    activations at 1.47x and 1.54x, h3_image's and wan22's 14336-wide ones at 1.24x and
    1.29x, flux2's and qwen's 12288-wide ones at 1.16x. Nothing at or below 19 KiB moved in
    any block, and anima's slow 16 KiB row answered a kernel search rather than a pitch,
    which is a different disease. Rows narrower than ``min_rows``, on the other hand, lose:
    the flux2 text stream was 0.89x on the same problem its image stream won 1.17x with.

    So the two bounds are the finding, and the engine applies them in
    ``thenoise.dit.quantized``, which is why this arm now reads flat. Run it with the bounds
    moved to price the boundary rather than the fix.

    The wrapper copies rather than writing straight into the wider buffer, which is what an
    arm can do from outside a forward; the buckets section says whether Inductor folds that
    copy into the kernel that produced the activation, and it did for the LTX FFN.
    """
    out = []
    for mod in _projections(blk, lambda i, o: True):
        if mod.weight.shape[-1] * mod.weight.element_size() < rules.min_pitch:
            continue

        def forward(x, _m=mod):
            rows = x.dim() >= 2 and x.numel() // x.shape[-1] >= rules.min_rows
            if rows and x.stride(-2) * x.element_size() >= rules.min_pitch:
                wide = torch.empty((*x.shape[:-1], x.shape[-1] + PITCH_PAD_ELEMENTS),
                                   dtype=x.dtype, device=x.device)
                wide[..., :x.shape[-1]] = x
                return type(_m).forward(_m, wide[..., :x.shape[-1]])
            return type(_m).forward(_m, x)

        out.append((mod, "forward", forward))
    return out


def weight_transposed_patches(blk: nn.Module, rules: LayoutRules) -> list:
    """Wide-output projections with their weight stored the other way round, same numbers.

    Where the row count is small and the output wide, the weight is the traffic and
    ``F.linear`` reads it strided by its stored row. A transposed store hands the GEMM that
    operand contiguous: 1.43x and 1.38x on flux2's two text-stream rows, 1.24x on moss's and
    1.16x on qwen_image's and fibo2's, and nothing anywhere the stream was long.

    The patch swaps the *tensor*, not the forward, which is what a load-time transpose does:
    the projection sees a weight of the shape it expects and a layout it never asked for, and
    its operand padding and LoRA branch stay in the graph untouched. Which is why this arm
    reads flat on the models that ask for the store, and reads its gain on the ones that
    declined: the camera builds a block the way its loader leaves it, so the arm is only a
    no-op where the engine already did it.
    """
    return [(mod, "weight",
             nn.Parameter(mod.weight.t().contiguous().t(), requires_grad=False))
            for mod in _projections(blk, lambda i, o: o >= rules.transpose_min_ratio * i)]


def weight_plain_store_patches(blk: nn.Module, rules: LayoutRules) -> list:
    """Wide-output projections with the checkpoint's own store put back under them.

    The undo of what the loader does as the weights land (``thenoise.utils.layout``), and
    therefore the arm that says what the relay is worth on a block that has had it. It
    copies rather than patching the GEMM, because the relay is a copy too: the difference
    between the two rows is the layout, not the arithmetic.
    """
    return [(mod, "weight",
             nn.Parameter(mod.weight.data.contiguous(), requires_grad=False))
            for mod in _projections(blk, lambda i, o: o >= rules.transpose_min_ratio * i)]


def transposed_problem_patches(blk: nn.Module, rules: LayoutRules) -> list:
    """Narrow-output projections answering for ``Cᵀ``, their consumers left to cope.

    The same reduction with its free dimensions exchanged — ``W @ xᵀ`` for ``x @ Wᵀ`` — was
    1.45x on anima's ``4096 x 8192 x 2048``, the row the pitch had no effect on, and 1.46x
    on LTX's small audio projections: where a projection reduces to a fraction of its width
    the tile problem is better the other way round.

    The answer comes back as a transposed view rather than a copy, because whether the block
    eats the strided reads that follow is the question this arm asks, and transposing here
    would answer it wrong. It also asks the question the sweep answered about the operand
    layout the other way round: a column-major operand cost 2.7x on LTX's down projection.
    """
    out = []
    for mod in _projections(blk, lambda i, o: o <= rules.transposed_max_ratio * i):
        def forward(x, _m=mod):
            y = torch.matmul(_m.weight, x.transpose(-1, -2))
            if _m.bias is not None:
                y = y + _m.bias[:, None]
            return y.transpose(-1, -2)

        out.append((mod, "forward", forward))
    return out


def layout_arms(blk: Optional[nn.Module], rules: LayoutRules) -> list[Arm]:
    """The layout arms, the same three for every block: reach for the operand, the weight,
    and the problem itself.

    With no block they come back unpatched, which is what ``--list-arms`` wants — the widths
    are only known once the block exists, and that costs a GPU. Each note counts what its
    rule reached, since "0% and nothing to do" and "0% and nothing to reach" are different
    results.
    """
    built = ((pitch_pad_patches,
              PITCH_NOTE.format(min_pitch=rules.min_pitch, min_rows=rules.min_rows),
              "pitch_pad"),
             (weight_transposed_patches,
              WTRANSPOSE_NOTE.format(min_ratio=rules.transpose_min_ratio),
              "weight_transposed"),
             (weight_plain_store_patches, WEIGHT_PLAIN_NOTE, "weight_plain_store"),
             (transposed_problem_patches,
              TRANSPOSED_NOTE.format(max_ratio=rules.transposed_max_ratio),
              "transposed_problem"))
    arms = []
    for build, note, name in built:
        patches = build(blk, rules) if blk is not None else []
        arms.append(Arm(name,
                        note + (f" — {len(patches)} of this block's projections"
                                if blk is not None else ""),
                        exact=False, patches=patches))
    return arms


def generic_arms(blk: Optional[nn.Module], rules: LayoutRules) -> dict[str, Arm]:
    """The arms every block has, without anyone writing them down for it.

    ``baseline`` so a block can be timed against the changes the script knows how to make
    without knowing anything about it, then the layout arms, which read the module widths.
    A block worth real arms gets its own set and this one stops being used.
    """
    return {"baseline": Arm("baseline", "the block as shipped"),
            **{arm.name: arm for arm in layout_arms(blk, rules)}}


def phases_dtype(dtype: torch.dtype):
    """``AVRope._phases`` rebuilt with the tables in ``dtype``.

    Held while the case is built, so the tables the block is handed are the narrow ones
    rather than widened on the way into each attention.
    """
    real = AVRope._phases

    def build(self, grid, batch):
        return tuple(t.to(dtype) for t in real(self, grid, batch))

    return build


# ----------------------------------------------------------------------- arms
@dataclass
class Arm:
    """One ablation: what to hold while timing, and which terms it takes out.

    ``inputs`` patches are held while the case is built as well, so an arm can change
    what the block is *given* rather than what it does — which needs its own inputs, and
    therefore its own case.
    """

    name: str
    note: str
    terms: tuple[str, ...] = ()
    patches: list = field(default_factory=list)
    inputs: list = field(default_factory=list)
    #: An arm that does not reproduce the block's answer is still a timing fact about the
    #: kernels it removes; say which arms are of that kind.
    exact: bool = True
    result: Optional["Result"] = None


def prompt_mask_patches(*attns: nn.Module) -> list:
    """Give each attention back the additive all-valid operand LTX used to build for it.

    The transformer now reads the prompt mask once per step, outside the compiled block, and
    hands the block a crop or no operand at all. This undoes that decision at the attention,
    which is where the key count is known, so the block pays for a mask again and the arm
    says what the decision is worth on a block that already has it.
    """
    out = []
    for att in attns:
        def forward(x, *, context=None, mask=None, attn_params=None, q_rope=None,
                    k_rope=None, _m=att):
            empty = (attn_params is not None and attn_params.attention_mask is None
                     and attn_params.key_prefix is None)
            if context is not None and empty:
                mask = torch.zeros(x.shape[0], context.shape[1],
                                   dtype=x.dtype, device=x.device)
                attn_params = None
            return type(_m).forward(_m, x, context=context, mask=mask,
                                    attn_params=attn_params, q_rope=q_rope, k_rope=k_rope)

        out.append((att, "forward", forward))
    return out


def ltx_arms() -> dict[str, Arm]:
    """Every arm that needs no block instance; the sub-module ones are filled in later."""
    forward = ltx._AVBlock.forward
    open_graph = torch.compile(fullgraph=False)(forward._torchdynamo_orig_callable)

    return {a.name: a for a in (
        Arm("baseline", "the block as shipped"),
        Arm("prompt_mask_operand",
            "an additive all-valid operand put back on both prompt attentions, the way LTX "
            "built one for every block before the transformer started reading the mask once "
            "per step: the same function and the same FLOPs answered by a different SDPA "
            "kernel, so there is no rate here, only the time an operand costs", exact=False),
        Arm("open_graph", "the block compiled without fullgraph, semantics untouched",
            patches=[(ltx._AVBlock, "forward", open_graph)]),
        Arm("open_graph_rope",
            "the two above at once: open graph and bf16 phases, which is where the rope "
            "chain stops being promoted to fp32 beside the query",
            inputs=[(AVRope, "_phases", phases_dtype(torch.bfloat16))],
            patches=[(ltx._AVBlock, "forward", open_graph)]),
        Arm("ff_down_contig",
            "the FFN's down projection fed contiguous: it runs at half the rate its twin "
            "does, and the gelu output is the operand that might be strided"),
        Arm("ff_down_padded",
            "the FFN's activation written into rows 8 elements wider, so the down "
            "projection reads its operand with a 32784-byte pitch rather than 32768: eager "
            "says that costs this GEMM half its rate, and a tuned kernel gets the same "
            "answer without the pitch",
            exact=False),
        Arm("pitch_pad", PITCH_NOTE.format(min_pitch=PAD_MIN_PITCH,
                                           min_rows=PITCH_MIN_ROWS), exact=False),
        Arm("weight_transposed",
            WTRANSPOSE_NOTE.format(min_ratio=TRANSPOSE_MIN_RATIO), exact=False),
        Arm("weight_plain_store", WEIGHT_PLAIN_NOTE, exact=False),
        Arm("transposed_problem",
            TRANSPOSED_NOTE.format(max_ratio=TRANSPOSED_MAX_RATIO), exact=False),
        Arm("sdpa_eager",
            "SDPA outside the compiled region (attention -> eager_attention), which "
            "breaks the graph around every attention call",
            patches=[(ltx._AVBlock, "forward", open_graph),
                     (ltx, "attention", eager_attention)]),
        Arm("sdpa_eager_masked",
            "SDPA eager with the additive operand restored on the prompt attentions, which "
            "is what an operand looks like once the region around it is unfused",
            exact=False,
            patches=[(ltx._AVBlock, "forward", open_graph),
                     (ltx, "attention", eager_attention)]),
        Arm("rope_bf16",
            "the rotary phase tables carried in bf16 instead of fp32: half the table "
            "traffic, and the doubled band assembled in bf16",
            inputs=[(AVRope, "_phases", phases_dtype(torch.bfloat16))], exact=False),
        Arm("no_layout_fixup",
            "q, k and v handed to SDPA in whatever layout the projections left them, which "
            "is what ``uniform_layout`` exists to prevent: it copies each one into the "
            "L-major order the fast kernels want, and the RMSNorm and the rope ride along "
            "inside that copy for free, which is why removing either of them changes the "
            "codegen around it rather than the work. The same math either way — the question "
            "is whether the copy is what the strided operands really cost, or whether they "
            "cost less than it does", exact=False),
        Arm("no_qk_norm",
            "the query and key RMSNorms out of every attention. LTX norms over the whole "
            "head vector rather than per head, so this is not a norm that can be moved "
            "after the split into heads and fused with the rope there: it is removed, which "
            "is what separates its cost from the rope and the layout fix-up riding along in "
            "the same kernels", exact=False),
        Arm("no_rope", "no rotary rotation at all",
            patches=[(ltx, "apply_rope_split_half", lambda x, *_phases: x)], exact=False),
        Arm("no_gate", "the per-head 2*sigmoid gate off every attention",
            terms=("self_video", "prompt_video", "a2v", "v2a"), exact=False),
        Arm("no_self_video",
            "video self-attention out — control, since it is the block's largest term",
            terms=("self_video",), exact=False),
        Arm("no_ffn_video", "the video FFN out — control for the wide projections",
            terms=("ffn_video",), exact=False),
        Arm("no_prompt_attn", "both prompt cross-attentions out",
            terms=("prompt_video", "prompt_audio"), exact=False),
        Arm("no_a2v", "audio-to-video out: 16384 video queries over 126 audio keys",
            terms=("a2v",), exact=False),
        Arm("no_v2a", "video-to-audio out: 126 audio queries over 16384 video keys, and "
                      "the two projections that build those keys from the video stream",
            terms=("v2a",), exact=False),
        Arm("no_cross_modal",
            "the reciprocal cross-attentions out, which is what isolate_modalities is",
            terms=("a2v", "v2a"), exact=False),
        Arm("no_audio_stream",
            "the audio stream's attentions and FFN out; its tokens, norms and the "
            "cross-modal side stay",
            terms=("self_audio", "prompt_audio", "ffn_audio"), exact=False),
    )}


# ------------------------------------------- the audiovisual trunk's fused-kernel arms
#: Isolated elementwise measurements mislead here. The engine compiles its blocks, so
#: today's rms_norm plus modulation is not a chain of materialised tensors: Inductor fuses
#: it into its neighbours, and an opaque comfy_kitchen call replaces that fusion with a
#: graph cut. ``ck.rms_adaln`` measured 10x faster and 6x smaller in isolation and came out
#: 1.7% *slower* inside the block, which is why these arms exist and why no adoption in the
#: project rests on an isolated number.
#: The producer's chunk. 2 048 and 4 096 measured the same to 0.7 ms at 40 114 tokens, so
#: the cost is not per chunk and there is nothing to tune here.
PRODUCER_CHUNK = 4096


def h3_modulate(x, spans, norm, attn):
    import comfy_kitchen as ck
    import thenoise.dit.h3.trunk as trunk

    parts = []
    for span in spans:
        mod = span.attn if attn else span.mlp
        parts.append(ck.rms_adaln(x[:, span.start:span.end], mod.scale, mod.shift, norm.eps))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def h3_rope_table(cos, sin):
    """The trunk's (cos, sin) in the kernel's dialect: ``[B, T, 1, rot/2, 2, 2]``, rows ``[[c,s],[-s,c]]``.

    The sign placement is the opposite of what the rotation formula suggests, and nothing
    complains when it is wrong: the kernel accepts the transposed-sign table and returns
    garbage at 1.1 relative error, which is what the numerics line is for. Built per call:
    a cache keyed on tensor identity would hand the compiled region a constant the graph
    cannot see through.
    """
    c, s = cos.transpose(-2, -3), sin.transpose(-2, -3)
    return torch.stack((torch.stack((c, s), dim=-1),
                        torch.stack((-s, c), dim=-1)), dim=-1)


def h3_arms() -> dict[str, Arm]:
    """Fused replacements for the three terms a routed trunk block still pays for itself.

    Each one patches a name the compiled forward calls, so the block under test is the real
    one and every arm is its own Inductor artifact. None of them answers what the shipped
    block answers: two because a fused kernel computes a slightly different thing, one
    because ``ck.rms_adaln`` has no learned norm weight at all. The numerics line says by how
    much, and a rope table with the wrong dialect shows up near 1.0 rather than 5e-4.
    """
    import thenoise.dit.h3.trunk as trunk
    from thenoise.dit.h3.trunk import Attention, attend

    dense = "ignore the plan: the same block run with exact attention"
    shipped = Attention.forward

    def dense_forward(self, x, rope=None, bufs=None, sol=None):
        return shipped(self, x, rope, bufs, None)

    def fused_rope_forward(self, x, rope=None, bufs=None, sol=None):
        """Mirror of ``Attention.forward`` with QK-norm and rope in one kernel."""
        import comfy_kitchen as ck
        from thenoise.utils.rope import apply_rope_split_half

        batch, rows, _ = x.shape
        inner = self.heads * self.head_dim
        q, k, v = self.qkv_proj(x).split(inner, dim=-1)
        q, k, v = (t.view(batch, rows, self.heads, self.head_dim)
                   for t in (q, k, v))
        norm = self.qk_norm
        q, k = ck.rms_rope_split_half(q, k, h3_rope_table(*rope),
                                      norm.query_norm.weight, norm.key_norm.weight,
                                      norm.query_norm.eps, 2 * rope[0].shape[-1])
        if sol is None:
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            return self.out_proj(attend(q, k, v, bufs))
        return self.out_proj(trunk.sol_attention(q, k, v, sol).reshape(batch, rows, inner))

    @torch._dynamo.disable()      # the producer takes a generator, which Inductor cannot trace
    def producer(self, x, rope, sol, chunk):
        import comfy_kitchen as ck

        norm = self.qk_norm

        def chunks():
            for start in range(0, x.shape[1], chunk):
                yield self.qkv_proj(x[:, start:start + chunk])

        out, *_carried = ck.sol_attn_chunked(
            chunks, x.shape[1], self.heads, h3_rope_table(*rope),
            (norm.query_norm.weight, norm.key_norm.weight), kmean=None, vscale=None,
            tau=sol.tau, sink_blocks=sol.sink_blocks, sink_q=sol.sink_q,
            token_aug=sol.token_aug, rope_eps=norm.query_norm.eps, rope_split_half=True)
        return out.flatten(2)

    def producer_forward(self, x, rope=None, bufs=None, sol=None):
        return self.out_proj(producer(self, x, rope, sol, PRODUCER_CHUNK))

    return {a.name: a for a in (
        Arm("baseline", "routed as shipped: sol_attn on token-major slices, "
            "Inductor fusing the norms, the modulation and the rope"),
        Arm("dense_attention", dense, terms=("self_attn",), exact=False,
            patches=[(Attention, "forward", dense_forward)]),
        Arm("adaln_ck", "ck.rms_adaln for both modulations: rejected, it has no learned "
            "norm weight and the block has one",
            exact=False, patches=[(trunk, "_modulate", h3_modulate)]),
        Arm("qk_rope_ck", "ck.rms_rope_split_half for q and k in one call: needs both "
            "tensors at the call site, so the trunk has no drop-in seam for it",
            exact=False, patches=[(Attention, "forward", fused_rope_forward)]),
        Arm("sol_producer", "ck.sol_attn_chunked: the one candidate that never materialises "
            "q, k and v for the sequence, and it pays for the projection running eager",
            exact=False, patches=[(Attention, "forward", producer_forward)]),
    )}


ARMS: dict[str, Callable[[], dict[str, Arm]]] = ({"ltx/av": ltx_arms,
                                                   "ltx/av_i2v": ltx_arms,
                                                   "h3/av_block": h3_arms})
FIXTURES = {"ltx/av": ltx_fixture, "ltx/av_i2v": partial(ltx_fixture, i2v=True),
            "h3/av_block": h3_fixture}


def arm_modules(arms: dict[str, Arm], blk: nn.Module, rules: LayoutRules) -> None:
    """Fill in the arms that name sub-modules, now that the block exists.

    A sub-module is replaced with a stand-in forward on the *instance*, which Inductor
    guards on, so each such arm gets its own graph without touching the class.
    """
    if "no_ffn_video" not in arms:
        return
    sub = dict(blk.named_modules())

    def mods(*names: str) -> list:
        return [(sub[name], "forward", zero_out) for name in names]

    arms["no_self_video"].patches = mods("attn1")
    arms["no_ffn_video"].patches = mods("ff")
    arms["no_prompt_attn"].patches = mods("attn2", "audio_attn2")
    arms["no_a2v"].patches = mods("audio_to_video_attn")
    arms["no_v2a"].patches = mods("video_to_audio_attn")
    arms["no_cross_modal"].patches = mods("audio_to_video_attn", "video_to_audio_attn")
    arms["no_audio_stream"].patches = mods("audio_attn1", "audio_attn2", "audio_ff")
    if "no_qk_norm" in arms:
        arms["no_qk_norm"].patches = [
            (sub[f"{name}.qk_norm"], "forward", lambda q, k, *_kw: (q, k))
            for name in ("attn1", "attn2", "audio_attn1", "audio_attn2",
                         "audio_to_video_attn", "video_to_audio_attn")
            if f"{name}.qk_norm" in sub]
    if "no_layout_fixup" in arms:
        arms["no_layout_fixup"].patches = [
            (attention_mod, "uniform_layout", lambda q, k, v: (q, k, v))]
    operand = prompt_mask_patches(sub["attn2"], sub["audio_attn2"])
    arms["prompt_mask_operand"].patches = operand
    arms["sdpa_eager_masked"].patches += operand
    ff = sub["ff"]

    def ff_forward(x, _ff=ff):
        return _ff.net[2](_ff.net[1](_ff.net[0](x)).contiguous())

    def ff_padded(x, _ff=ff, pad=8):
        y = _ff.net[0](x)
        wide = torch.empty((*y.shape[:-1], y.shape[-1] + pad), dtype=y.dtype,
                           device=y.device)
        wide[..., :y.shape[-1]] = y
        return _ff.net[2](_ff.net[1](wide[..., :y.shape[-1]]))

    arms["ff_down_contig"].patches = [(ff, "forward", ff_forward)]
    arms["ff_down_padded"].patches = [(ff, "forward", ff_padded)]
    for arm in layout_arms(blk, rules):
        if arm.name in arms:
            arms[arm.name] = arm
    arms["no_gate"].patches = [(mod, "to_gate_logits", None) for mod in sub.values()
                               if getattr(mod, "to_gate_logits", None) is not None]


# --------------------------------------------------------------------- timing
@dataclass
class Result:
    ms: float
    gain_pct: float
    cov_pct: float
    peak_gib: float
    compile_s: float
    groups: list[float] = field(default_factory=list)
    maxabs: float = float("nan")
    cos: float = float("nan")
    ops: int = 0

    def as_json(self) -> dict:
        return {k: getattr(self, k) for k in
                ("ms", "cov_pct", "peak_gib", "compile_s", "maxabs", "cos", "ops")}


def cpu_ops(run: Callable[[], object], items: list = ()) -> int:
    """Aten ops issued by one call, which is how a child can tell a compiled block from not.

    A child of this script cannot profile device activity — kineto collects none in a
    process whose parent has touched the GPU — but CPU-side records still arrive. The
    compiled block's call issues a steady count of them; one that broke out of the graph
    issues plainly more, which is the only way a child can say whether an arm removed work
    or removed compilation. Read as a ratio against the arm's own baseline, not as an
    absolute.
    """
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad(), patches(items):
        run()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            run()
            torch.cuda.synchronize()
        return sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))


def child_cmd(arm: str, args, out: Path) -> list[str]:
    """The command that measures one arm against a baseline of its own.

    ``TORCHINDUCTOR_CACHE_DIR`` is inherited rather than passed, so a parent running
    ``--fresh-cache`` gives every child the same empty cache instead of a new one each.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
           "--arms", f"baseline,{arm}", "--block", args.block,
           "--tokens", str(args.tokens), "--txt", str(args.txt),
           "--weights", args.weights, "--device", args.device,
           "--warmup", str(args.warmup), "--iters", str(args.iters),
           "--repeats", str(args.repeats), "--json", str(out),
           "--pad-min-pitch", str(args.pad_min_pitch),
           "--pad-min-rows", str(args.pad_min_rows),
           "--weight-transposed-min-ratio", str(args.weight_transposed_min_ratio),
           "--transposed-problem-max-ratio", str(args.transposed_problem_max_ratio)]
    if args.scenario:
        cmd += ["--scenario", args.scenario]
    if args.tunable:
        cmd += ["--tunable", str(args.tunable)]
        if args.tune:
            cmd.append("--tune")
    return cmd


def spawned_ladder(args, chosen: list[Arm], fixture: Fixture, total: float) -> None:
    """Measure each arm in a process of its own, against a baseline timed in there.

    Arms are not independent inside one process. Tracing one changes what the next trace of
    the block produces: routing SDPA through the dynamo-disabled ``eager_attention`` leaves
    the block uncompiled for everything measured afterwards, attribute restored or not. So
    the 15-arm ladder this script used to run had no healthy baseline in it — 627 ms against
    the 498 ms the same block gives in a short run, 1 GiB heavier, its elementwise work in
    ATen kernels instead of triton — and every gain in it was relative to that. One process
    per arm, with the arm's own baseline timed before its patch is applied and again after
    (``pair``), is the smallest thing that can still be believed.

    A child measures time only. Kineto collects no device activity in a process whose parent
    has touched the GPU, so the kernel-table sections (``profile``, ``buckets``) belong to a
    single process, where the block can also be walked arm by arm and read directly.
    """
    picked = [a for a in chosen if a.name != "baseline"]
    rows: list[tuple[Arm, dict, float, float, float]] = []
    for arm in picked:
        out = Path(tempfile.gettempdir()) / f"probe-{os.getpid()}-{arm.name}.json"
        print(f"  {arm.name}: child process", flush=True)
        child = subprocess.run(child_cmd(arm.name, args, out), capture_output=True,
                               text=True, check=False)
        if not out.exists():
            print(child.stdout[-3000:], child.stderr[-3000:], sep="\n")
            raise SystemExit(f"{arm.name}: the child wrote no results")
        got = json.loads(out.read_text())
        rows.append((arm, got[arm.name], got["baseline"]["ms"], got["baseline"]["ops"],
                     got["drift_pct"]))
        out.unlink()

    print(f"\n{'arm':{NAME_W}s} {'ms':>9s} {'gain%':>7s} {'cov%':>6s} {'GiB':>6s}"
          f" {'its base':>10s} {'drift%':>7s} {'ops':>7s} {'gone TFLOP':>11s}"
          f" {'rate TF/s':>10s} {'maxabs':>9s} {'cos':>8s}")
    bases = []
    for arm, got, base_ms, base_ops, drift in rows:
        gone = sum(fixture.terms[t] for t in arm.terms)
        saved = base_ms - got["ms"]
        bases.append(base_ms)
        rate = gone / saved * 1e-9 if saved > 0.05 and gone else float("nan")
        gain = 100 * saved / base_ms
        flag = "" if abs(gain) >= FLOOR_PCT else "  <- under the floor"
        print(f"{arm.name:{NAME_W}s} {got['ms']:9.2f} {gain:7.2f} {got['cov_pct']:6.2f}"
              f" {got['peak_gib']:6.2f} {base_ms:10.2f} {drift:7.2f}"
              f" {got['ops']:7d} {gone / 1e12:11.3f} {rate:10.1f} {got['maxabs']:9.2e}"
              f" {got['cos']:8.5f}{flag}")
        if abs(drift) > max(3.0, 3 * got["cov_pct"]):
            print(f"      {'':{NAME_W}s} tracing this arm left the baseline "
                  f"{drift:+.1f}% off, so the block it measured may not be the one the "
                  f"gain is about: read it with --section buckets")
        if got["ops"] > 1.5 * base_ops:
            print(f"      {'':{NAME_W}s} {got['ops']} aten ops per call against the "
                  f"baseline's {base_ops}: part of this arm ran outside the compiled "
                  f"region, so its gain measures giving up compilation as well as the "
                  f"work it removed")
    print()
    for arm in picked:
        print(f"  {arm.name:{NAME_W}s} {arm.note}")
    mean_base = sum(bases) / len(bases)
    print(f"\n  each row against the baseline timed in its own process before its patch was "
          f"applied, which averaged {mean_base:.2f} ms"
          + (f": the block's own {total / mean_base * 1e-9:.1f} TF/s over the terms above, a "
             "floor since they omit the projections and the rope."
             if total else ", a block this script holds no FLOPs model for.")
          + " drift% is a second baseline timed after the arm; where it is far from flat, "
            "tracing the arm changed the block and its gain is called into question, not "
            "corrected.")


def sweep(arms: list[Arm], runs: dict[str, Callable], ctx: bb.Ctx,
          traced_s: float = 0.0) -> dict[str, Result]:
    """Compile and warm every arm, then time one group of each per sweep.

    The arms re-trace each other out of their cached graphs — an instance attribute an arm
    adds is a guard every other arm watched — so one untimed pass over the whole ladder
    comes first. Without it a recompile lands inside a timed group, which reads as a
    median no better than its own worst group.

    ``traced_s`` is what the baseline took to trace when it was traced, which is not
    necessarily here: the graph line runs the block once to look at it.
    """
    sync = torch.cuda.synchronize if ctx.device == "cuda" else (lambda: None)
    groups: dict[str, list[float]] = {a.name: [] for a in arms}
    compile_s, peak = {}, {}
    with torch.no_grad():
        for arm in arms:
            with patches(arm.patches):
                run = runs[arm.name]
                if ctx.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t0 = _now()
                run()
                sync()
                compile_s[arm.name] = (traced_s if traced_s and arm.name == "baseline"
                                       else _now() - t0)
                for _ in range(ctx.warmup):
                    run()
                sync()
                if ctx.device == "cuda":
                    peak[arm.name] = torch.cuda.max_memory_allocated() / 2**30
        for settled in range(ctx.repeats + 1):
            for arm in arms:
                with patches(arm.patches):
                    run = runs[arm.name]
                    t0 = _now()
                    for _ in range(ctx.iters):
                        run()
                    sync()
                    if settled:
                        groups[arm.name].append((_now() - t0) * 1000 / ctx.iters)

    base = groups.get("baseline") or [1.0]
    results = {}
    for arm in arms:
        med = statistics.median(groups[arm.name])
        result = Result(med,
                        statistics.median([100 * (1 - a / b) for a, b in
                                           zip(groups[arm.name], base)]),
                        100 * statistics.pstdev(groups[arm.name]) / med,
                        peak.get(arm.name, 0.0), compile_s[arm.name],
                        [round(g, 1) for g in groups[arm.name]])
        arm.result = result
        results[arm.name] = result
    return results


def pair(arms: list["Arm"], runs: dict[str, Callable], ctx: bb.Ctx,
         traced_s: float = 0.0) -> dict:
    """One arm, measured between two baselines, in a process that traces nothing else.

    The baseline comes first because tracing an arm changes what follows it: with
    ``attention`` swapped for the dynamo-disabled ``eager_attention``, the block stops being
    compiled afterwards even once the attribute is back, and a baseline measured after that
    is a quarter slower than the same block measured before it. The second baseline is the
    control — where the two disagree, the arm's gain is meaningless and says so.

    Each measurement carries its aten op count, since the other thing an arm can silently
    lose is its compilation.
    """
    base, arm = arms[0], arms[1]
    first = sweep([base], runs, ctx, traced_s)[base.name]
    first.ops = cpu_ops(runs[base.name])
    measured = sweep([arm], runs, ctx)[arm.name]
    measured.ops = cpu_ops(runs[arm.name], arm.patches)
    again = sweep([base], runs, ctx)[base.name]
    measured.gain_pct = 100 * (first.ms - measured.ms) / first.ms
    return {"baseline": first, arm.name: measured, "base_again": again,
            "drift_pct": 100 * (again.ms / first.ms - 1)}


def _tensors(out) -> list[torch.Tensor]:
    """Every tensor an output holds, in order, however it is nested."""
    if torch.is_tensor(out):
        return [out.float()]
    if isinstance(out, (tuple, list)):
        return [t for item in out for t in _tensors(item)]
    return []


def numerics(arms: list[Arm], runs: dict[str, Callable]) -> None:
    """Each arm's outputs against the baseline's, worst over everything the block returns.

    Every output tensor is compared rather than the first one, because an arm can be exact
    on the video side and wrong on the audio one — which is the side a reciprocal
    cross-attention writes — and a block returning one tensor still wants checking. A block
    whose arms change the shape of what comes back has no number to report, and gets blank
    rather than a comparison of different things.
    """
    with torch.no_grad(), patches(arms[0].patches):
        base = _tensors(runs["baseline"]())
    for arm in arms[1:]:
        with torch.no_grad(), patches(arm.patches):
            got = _tensors(runs[arm.name]())
        if len(got) != len(base) or any(g.shape != b.shape
                                        for g, b in zip(got, base)):
            arm.result.maxabs = arm.result.cos = float("nan")
            continue
        arm.result.maxabs = max(float((g - b).abs().max())
                                for g, b in zip(got, base))
        arm.result.cos = min(float(nn.functional.cosine_similarity(
            g.reshape(-1), b.reshape(-1), dim=0)) for g, b in zip(got, base))
        del got


# ----------------------------------------------------------------------- sdpa
BACKENDS = ("default", "flash", "efficient", "cudnn", "math")


@dataclass(frozen=True)
class Operand:
    """One attention call of the block, as its operands reach ``attention()``."""

    name: str
    heads: int
    head_dim: int
    queries: int
    keys: int
    masked: bool

    @property
    def tflop(self) -> float:
        return bb.attn_flops(self.heads, self.head_dim, ((self.queries, self.keys),))


def ltx_operands(ctx: bb.Ctx, tokens: int) -> list[Operand]:
    """Every attention of one step at ``tokens`` video tokens, masked and unmasked.

    The mask is the additive one the transformer builds, and q/k/v are L-major with the
    phases already applied, so these are the tensors the kernel sees.
    """
    vheads = bb.LTX_VDIM // ltx.VIDEO_HEAD_DIM
    aheads = bb.LTX_ADIM // ltx.AUDIO_HEAD_DIM
    seconds = ((bb.LTX_VIDEO_SCALE * (bb.LTX_FRAMES - 1) + 1) / bb.LTX_FPS)
    a_tokens = max(1, round(seconds * bb.LTX_AUDIO_TPS))
    rows = [Operand("video self", vheads, ltx.VIDEO_HEAD_DIM, tokens, tokens, False),
            Operand("video prompt", vheads, ltx.VIDEO_HEAD_DIM, tokens, ctx.txt, True),
            Operand("audio prompt", aheads, ltx.AUDIO_HEAD_DIM, a_tokens, ctx.txt, True),
            Operand("a2v  video q / audio k,v", aheads, ltx.AUDIO_HEAD_DIM,
                    tokens, a_tokens, False),
            Operand("v2a  audio q / video k,v", aheads, ltx.AUDIO_HEAD_DIM,
                    a_tokens, tokens, False)]
    out = []
    for row in rows:
        out.append(row)
        if row.masked:
            out.append(Operand(row.name + " [unmasked]", row.heads, row.head_dim,
                               row.queries, row.keys, False))
    return out


def sdpa_section(ctx: bb.Ctx, tokens: int, iters: int, verbose: bool = False) -> None:
    """Every operand against every forced backend, in eager.

    Eager, because that is where the backend is chosen per call: a compiled graph bakes
    the choice in at trace time. ``refused`` is the answer that matters -- it says which
    operand a fused kernel will not take, and therefore what ``default`` really ran.
    ``no room`` is a different fact: the backend would have answered, at a footprint this
    machine does not have, which the math path on a 16384-token self-attention is.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    pick = {"default": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH],
            "flash": [SDPBackend.FLASH_ATTENTION],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION],
            "cudnn": [SDPBackend.CUDNN_ATTENTION],
            "math": [SDPBackend.MATH]}
    print(f"\nsdpa: eager scaled_dot_product_attention, {ctx.dtype}, {iters} iters, ms "
          f"per call\n")
    print(f"{'operand':32s} {'TFLOP':>6s}" + "".join(f"{b:>12s}" for b in BACKENDS)
          + "   TFLOP/s  which one 'default' reached for")
    for op in ltx_operands(ctx, tokens):
        q = torch.randn(1, op.heads, op.queries, op.head_dim, device=ctx.tdev,
                        dtype=ctx.dtype)
        k = torch.randn(1, op.heads, op.keys, op.head_dim, device=ctx.tdev,
                        dtype=ctx.dtype)
        v = torch.randn_like(k)
        mask = (torch.zeros(1, 1, 1, op.keys, device=ctx.tdev, dtype=ctx.dtype)
                if op.masked else None)
        cells, times, notes = [], {}, []
        call = lambda: torch.nn.functional.scaled_dot_product_attention(  # noqa: E731
            q, k, v, attn_mask=mask)
        for name in BACKENDS:
            try:
                with sdpa_kernel(pick[name]):
                    call()
                    torch.cuda.synchronize()
                    t0 = _now()
                    for _ in range(iters):
                        call()
                    torch.cuda.synchronize()
                    ms = (_now() - t0) * 1000 / iters
                cells.append(f"{ms:12.2f}")
                times[name] = ms
            except Exception as exc:                                    # noqa: BLE001
                first = str(exc).splitlines()[0]
                room = "out of memory" in first.lower()
                notes.append(f"{name} {'ran out of room' if room else 'refused'}: "
                             f"{first[:70]}")
                cells.append(f"{'no room' if room else 'refused':>12s}")
                if room:
                    torch.cuda.empty_cache()
        which = next((n for n in BACKENDS[1:]
                      if n in times and abs(times[n] / times["default"] - 1) < 0.03),
                     "?")
        print(f"{op.name:32s} {op.tflop / 1e12:6.3f}" + "".join(cells)
              + f"   {op.tflop / times['default'] * 1e-9:8.1f}  {which}")
        if verbose:
            for note in notes:
                print(f"  {op.name:32s} {note}")
        del q, k, v, mask


# --------------------------------------------------------------- sparse attention
#: The trunk's own attention geometry — 56 heads of 128 over the 5376-wide stream, which
#: is also the only head width ``sol_attn`` is compiled for. The stock clip decoder is
#: 64 and cannot use it; the trunk can.
SOL_HEADS, SOL_HEAD_DIM = 56, 128

#: Packs to price, as ``(name, width, height, frames)``: the release's canvas plus the
#: same one at 544p, each with a short clip so the crossover below the kernel's own ~12 k
#: token threshold is in the table. The latent grid, the audio length, the spans and the
#: sequence length all come out of the codec's temporal plan and the layout's frame grid,
#: so a row is the shape a request builds rather than a rounding of it.
SOL_PACKS = (
    ("544p/5f", 960, 544, 5),
    ("768p/22f", 1344, 768, 22),
    ("544p/124f", 960, 544, 124),
    ("768p/124f", 1344, 768, 124),
)
SOL_TEXT = 220          # a prompt's worth of rows; the prefix is a sink at any length
SOL_FPS = 24            # the release's frame rate, which is what sets the audio length


def sol_pack(width: int, height: int, frames: int, fps: int = SOL_FPS):
    """One pack as the adapter builds it: latent grid, audio length, layout."""
    from thenoise.dit.h3.layout import Layout
    from thenoise.vae.h3 import TemporalPlan
    from thenoise.vae.h3_audio import LATENTS_PER_SECOND

    plan = TemporalPlan.of_frames(frames)
    return Layout.av(text_len=SOL_TEXT, latent_t=plan.latent_t,
                     latent_h=-(-height // 16), latent_w=-(-width // 16),
                     audio_t=max(1, round(frames / fps * LATENTS_PER_SECOND)))


def sol_sinks(layout, mode: str = "exact_rows") -> tuple[tuple[int, int], tuple[int, int]]:
    """The engine's sink policy for a pack, in the mode this run is pricing.

    :func:`thenoise.dit.h3.sparse.sol_sinks` is the policy a run uses — every row before the
    video span exact-KV for every query, and on ``exact_rows`` the audio query rows dense as
    well, because routing them damages the sound. ``off`` and ``exact`` exist here and not in
    the engine to price that policy rather than assume it.
    """
    if mode == "off":
        return (0, 0), (0, 0)
    return sink_policy(layout, audio_rows_dense=mode == "exact_rows")


def sol_section(ctx: bb.Ctx, iters: int, repeats: int, taus: list[float],
                only: str, sinks: str, verbose: bool = False) -> None:
    """Dense attention against ``comfy_kitchen.sol_attn``, on the shapes the trunk runs.

    One q/k/v per pack, read the way each path wants it: SDPA takes ``[B, H, T, D]``,
    which is what a projected stream becomes on its way into the engine's attention, and
    the kernel takes ``[B, T, H, D]``, which is what that stream is before the transpose.
    Both are views of the same bytes, so neither is handed a copy the other does not pay.

    ``speedup`` is against the dense call at the same shape. ``rel`` is how far the sparse
    answer sits from the dense one — but on unit-variance operands, where no query row is
    selective and every key block carries comparable mass, which is the pessimistic end for
    a router: the trunk's rows are normed and peaked, and the question of what a finished
    clip tolerates is not answerable from a microbenchmark. Read the speedup column here and
    the error from a run. The short packs are in the table because the kernel's own
    documentation puts the crossover near 12 k tokens, and a device without the compiled
    kernel reports the refusal per shape rather than leaving a blank.
    """
    import comfy_kitchen as ck

    from thenoise.utils.attention import eager_attention

    heads, head_dim = SOL_HEADS, SOL_HEAD_DIM
    here = ck.sol_attn_is_available(ctx.tdev)
    print(f"\nsol: sol_attn against dense SDPA, {ctx.dtype}, {heads}x{head_dim}, "
          f"sinks={sinks}, {iters} iters x {repeats} groups\n"
          + (f"  kernel available on {ctx.tdev}" if here
             else "  NO COMPILED KERNEL FOR THIS DEVICE: the dense column only"))
    print(f"\n{'pack':12s} {'tokens':>7s} {'exact':>9s} {'dense q':>9s} {'ms':>8s} {'TF/s':>6s}"
          + "".join(f"{f'ms (x) rel @tau {t:g}':>22s}" for t in taus))
    print(f"{'':12s} {'':7s}   (64-row blocks; a pack's frame count is on the codec's "
          f"own clip ladder)")

    for name, width, height, frames in SOL_PACKS:
        if only and name not in (p.strip() for p in only.split(",")):
            continue
        layout = sol_pack(width, height, frames)
        tokens = layout.seq_len
        sink, sink_q = sol_sinks(layout, sinks)
        streams = [torch.randn(1, tokens, heads * head_dim, device=ctx.tdev, dtype=ctx.dtype)
                   for _ in range(3)]
        bthd = [s.view(1, tokens, heads, head_dim) for s in streams]

        def dense():
            return eager_attention([t.transpose(1, 2) for t in bthd])

        row = f"{name:12s} {tokens:7d} {sink[1]:9d} {sink_q[1] - sink_q[0]:9d}"
        try:
            base = _median_ms(dense, iters, repeats)
            flops = 4 * heads * head_dim * tokens * tokens
            row += f" {base:8.1f} {flops / base * 1e-9:6.1f}"
            reference = dense().float()
        except torch.cuda.OutOfMemoryError:
            print(row + "   no room for the dense call at this shape")
            del streams, bthd
            torch.cuda.empty_cache()
            continue
        for tau in taus:

            def call():
                return ck.sol_attn(bthd[0], bthd[1], bthd[2], tau=tau,
                                   sink_blocks=list(sink), sink_q=list(sink_q))

            try:
                ms = _median_ms(call, iters, repeats)
                got = call().float().reshape(reference.shape)
                rel = ((got - reference).norm() / reference.norm()).item()
                row += f"  {ms:8.1f} ({base / ms:4.2f}x) {rel:6.1e}"
            except Exception as exc:                                # noqa: BLE001
                first = str(exc).splitlines()[0]
                room = "out of memory" in first.lower()
                row += f"{'no room' if room else 'refused':>22s}"
                print(f"  {name}: tau {tau:g} {'ran out of room' if room else 'refused'}: "
                      f"{first[:80]}")
            finally:
                torch.cuda.empty_cache()
        print(row)
        del streams, bthd, reference
        torch.cuda.empty_cache()
    print("  rel is the distance from the dense answer on random operands, the pessimistic\n"
          "  end for routing: no row of those is selective. What a clip tolerates is a run.")


# ------------------------------------------------------------------ eager GEMMs
def _median_ms(form: Callable[[], object], iters: int, repeats: int) -> float:
    """ms per call: the middle group of ``repeats`` groups of ``iters`` calls.

    The middle rather than the average, because on this hardware one throttled group out
    of five is ordinary and it moves an average by more than the effect being measured.
    """
    form()
    torch.cuda.synchronize()
    groups = []
    for _ in range(repeats):
        t0 = _now()
        for _ in range(iters):
            form()
        torch.cuda.synchronize()
        groups.append((_now() - t0) * 1000 / iters)
    groups.sort()
    return groups[len(groups) // 2]


def _tuned_ms(form: Callable[[], object], iters: int, repeats: int, tuning: Path,
              lt_algos: int) -> tuple[float, str]:
    """The same call with TunableOp choosing its kernel: let it tune, then time the record.

    The file is as much the deliverable as the number: one line per problem, each with a
    validator that expires the whole file when torch, HIP, rocBLAS or the architecture
    changes, so a build can carry it and a stale one cannot silently misdirect a GEMM.
    TunableOp writes each record out as it finishes tuning, and there is no call to ask it
    to: a run that dies mid-section still leaves what it learned.

    Which library answered is read out of what the tuning added, since that is the only
    thing identifying the problem — the record keys are TunableOp's own. A problem already
    in the file yields nothing new, and its library is then reported as unknown rather
    than guessed at from a key that this script does not own.
    """
    from torch.cuda import tunable

    was_enabled, was_tuning = tunable.is_enabled(), tunable.tuning_is_enabled()
    was_file = tunable.get_filename()
    known = {tuple(row) for row in tunable.get_results()}
    tunable.set_filename(str(tuning))
    tunable.enable(True)
    tunable.tuning_enable(True)
    if lt_algos:
        tunable.set_cublaslt_requested_algo_count(lt_algos)
    try:
        for _ in range(3):
            form()
        torch.cuda.synchronize()
        fresh = [row for row in tunable.get_results() if tuple(row) not in known]
        lib = ("hipBLASLt" if any("Hipblaslt" in str(row) for row in fresh)
               else "rocBLAS" if fresh else "already tuned")
        tunable.tuning_enable(False)
        return _median_ms(form, iters, repeats), lib
    finally:
        tunable.enable(was_enabled)
        tunable.tuning_enable(was_tuning)
        tunable.set_filename(was_file)


def _gemm_shapes(run: Callable[[], object], iters: int) -> tuple[list[tuple[int, int, int, float]], int]:
    """The block's own GEMM calls as ``(M, K, N, ms per call)``, the expensive ones first.

    Read out of the profiler's shape-grouped records, so any block the camera can build
    hands over its projections without anyone listing them. A half-rate GEMM is only found
    by asking every block the same question.

    The second value counts the records whose op is a known GEMM but whose operands the
    record cannot name — a batched ``bmm`` has no single ``(M, K, N)`` — and it is printed:
    a table of a block's GEMMs that quietly omitted some of them would be read as complete.
    """
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        run()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            for _ in range(iters):
                run()
            torch.cuda.synchronize()
    seen: dict[tuple[int, int, int], list[float]] = {}
    skipped = 0
    for e in prof.key_averages(group_by_input_shape=True):
        if e.count == 0 or e.self_device_time_total <= 0:
            continue
        dims = [list(s) for s in (getattr(e, "fused_cpu_input_shapes", None)
                                  or e.input_shapes or [])]
        got = _projection(e.key, dims)
        if got is None:
            if e.key in _GEMM_OPS:
                skipped += 1             # batched, or a pair the record cannot name
            continue
        slot = seen.setdefault(got, [0.0, 0.0])
        slot[0] += e.self_device_time_total / 1000
        slot[1] += e.count
    return (sorted(((m, k, n, ms / calls) for (m, k, n), (ms, calls) in seen.items()
                    if calls), key=lambda row: -row[3])), skipped


# The LTX projections, named by the operand widths that are unique inside the block.
LTX_GEMMS = {(16384, 4096): "ffn video down", (4096, 16384): "ffn video up",
             (4096, 4096): "video projections", (4096, 2048): "cross k/v",
             (2048, 4096): "v2a out"}


def gemm_section(ctx: bb.Ctx, run: Callable[[], object], iters: int, repeats: int,
                 tuning: Optional[Path], lt_algos: int, names: dict,
                 top: int = 8) -> None:
    """Every GEMM the block runs, in each form the same call could take, in eager.

    A GEMM that costs twice what its FLOPs should can be slow for four unrelated reasons,
    and a kernel table cannot tell them apart. Each has its own column, so the answer names
    the change to make:

    * ``shipped`` — ``addmm(bias, x, weight.t())``, the call a projection makes, with the
      weight arriving as a transposed view of its stored ``[N, K]``.
    * ``no bias`` — without the epilogue.
    * ``B copied`` — the weight pre-transposed, which a load-time layout change would hand
      every call forever.
    * ``A strided`` — the activation in rows eight elements wider than it needs. This is
      the one to read against the pitch under the row: where the operand's rows are a
      power-of-two number of bytes apart, this column is often faster than the operand
      that fits exactly.
    * ``A flipped`` — column-major, which is what a producer that chose the other layout
      leaves behind. Usually much worse, which is the answer to a different question.
    * ``W Aᵀ`` — the same reduction with the free dimensions swapped: the shape on its own,
      with nothing about the operands changed but the tile problem.
    * ``tuned`` — TunableOp's pick for the shipped call, out of what rocBLAS and
      ``--lt-algos`` hipBLASLt algorithms will offer for that exact problem. Where it wins,
      the file it writes is a per-machine configuration a build could carry.
    * ``pad tuned`` — TunableOp's pick for the *padded* call, which is a different problem
      and gets its own search: what the fix the engine now applies is worth once a kernel
      has been chosen for it. ``--no-tune`` skips both tuned columns, which is what a sweep
      over many blocks wants: the eager levers are free and a search is not.

    ``in blk`` is the same GEMM's cost as the block itself recorded it, so a row whose
    ``shipped`` disagrees with it is measuring something the graph does not do.
    """
    shapes, skipped = _gemm_shapes(run, max(2, iters))
    if ctx.weights != "bf16":
        print(f"\ngemm: declined — these are full-precision forms and the block was built "
              f"with {ctx.weights} weights, whose kernels are not these calls. What the "
              "block's own GEMMs cost, whatever op they are, is in --section shapes.")
        return
    shapes = shapes[:top]
    if not shapes:
        print("\ngemm: the block recorded no mm/addmm calls to bench")
        return
    itemsize = torch.empty(0, dtype=ctx.dtype).element_size()
    print(f"\ngemm: the block's own {len(shapes)} largest GEMMs in eager, {ctx.dtype}, "
          f"{iters} iters x {repeats} groups, ms per call"
          + (f", tuning to {tuning}" if tuning else "")
          + (f"; {skipped} batched records left out, they have no one shape" if skipped
             else "") + "\n")
    print(f"{'projection':20s}{'M x K x N':>21s}{'in blk':>8s}{'shipped':>9s}"
          f"{'no bias':>9s}{'B copied':>10s}{'A strided':>10s}{'A flipped':>10s}"
          f"{'W A\u1d40':>9s}{'tuned':>9s}{'pad tuned':>10s}  {'TF/s':>5s}")
    for m, k, n, in_blk in shapes:
        a = torch.randn(m, k, device=ctx.tdev, dtype=ctx.dtype)
        w = torch.randn(n, k, device=ctx.tdev, dtype=ctx.dtype)
        bias = torch.randn(n, device=ctx.tdev, dtype=ctx.dtype)
        wide = torch.randn(m, k + 8, device=ctx.tdev, dtype=ctx.dtype)
        a_flip = torch.randn(k, m, device=ctx.tdev, dtype=ctx.dtype)
        wt, wt_c = w.t(), w.t().contiguous()
        cells = [_median_ms(form, iters, repeats) for form in (
            lambda: torch.addmm(bias, a, wt), lambda: torch.mm(a, wt),
            lambda: torch.addmm(bias, a, wt_c),
            lambda: torch.addmm(bias, wide[:, :k], wt),
            lambda: torch.addmm(bias, a_flip.t(), wt), lambda: torch.mm(w, a.t()))]
        shipped = cells[0]
        if tuning is None:
            tuned, lib, pad_tuned = float("nan"), "", float("nan")
        else:
            tuned, lib = _tuned_ms(lambda: torch.addmm(bias, a, wt), iters, repeats,
                                   tuning, lt_algos)
            # The padded operand is a different problem for the library, so it gets its own
            # search: where it beats the padded form's own timing, a shipped kernel file is
            # worth more than the layout fix alone.
            pad_tuned, _ = _tuned_ms(lambda: torch.addmm(bias, wide[:, :k], wt),
                                     iters, repeats, tuning, lt_algos)
        print(f"{names.get((k, n), ''):20s}{f'{m} x {k} x {n}':>21s}{in_blk:8.2f}"
              + "".join(f"{c:10.2f}" for c in cells)
              + f"{tuned:9.2f}{pad_tuned:10.2f}"
              + f"  {2 * m * k * n / shipped * 1e-9:5.1f}")
        pitch = k * itemsize
        won = [f"{label} {shipped / ms:.2f}x" for label, ms in
               (("wider pitch", cells[3]), ("copied weight", cells[2]),
                ("hipBLASLt", tuned), ("padded and tuned", pad_tuned),
                ("swapped shape", cells[5]))
               if ms == ms and ms < shipped * 0.87]
        detail = (f"{pitch // 1024} KiB pitch"
                  + (" (power of two)" if not pitch & (pitch - 1) else ""))
        print(f"      {detail}; " + ("nothing here moves it" if not won
                                     else (f"{lib}: " if lib else "")
                                     + ", ".join(won))
              + ("  <- eager disagrees with the block"
                 if in_blk > 0 and abs(shipped / in_blk - 1) > 0.12 else ""))
        del a, w, bias, wide, a_flip, wt, wt_c


# ------------------------------------------------------------------- per shape
_GEMM_OPS = ("aten::mm", "aten::addmm", "aten::linear", "aten::_int_mm",
             "aten::bmm")
_ATTN_OPS = ("aten::_flash_attention_forward", "aten::_efficient_attention_forward",
             "aten::_scaled_dot_product_flash_attention",
             "aten::_scaled_dot_product_efficient_attention",
             "aten::_scaled_dot_product_cudnn_attention",
             "aten::_scaled_dot_product_attention_math")


def shape_section(run: Callable[[], object], name: str, iters: int, weight_bytes: int) -> None:
    """What each GEMM shape and each attention call cost, with the shapes they ran.

    Read from the profiler's own op records, since those carry the shapes: the kernel
    table says a ``Cijk_`` tile cost 30 ms, this says which projection it was. Attention
    rows also name the backend that answered, which is the difference between a slow
    kernel and a kernel that was never offered.

    ``MB/s`` is the weight traffic a GEMM would need at ``weight_bytes`` per element, so a
    row far under the machine's bandwidth has the bandwidth available and a row far over
    it is compute-bound. A small-M GEMM that lands nowhere near either is launch-bound.

    Read the attention rows as an index of which backend answered, not as a cost: under
    the profiler a fused attention's device time comes back inflated — the masked prompt
    attention here reads twice the cost of the same call timed in eager
    (``--section sdpa``), which is the instrument for what an attention is worth.
    """
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        run()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            for _ in range(iters):
                run()
            torch.cuda.synchronize()

    print(f"\nshapes: {name}, device time per call over {iters} calls\n")
    for label in ("GEMM", "attention"):
        rows = []
        for e in prof.key_averages(group_by_input_shape=True):
            if e.count == 0:
                continue
            dims = [list(s) for s in (getattr(e, "fused_cpu_input_shapes", None)
                                      or e.input_shapes or [])]
            # A projection is recognised by its operands, not by a name: a quantized
            # backend calls its own op, and the block's GEMMs are not fewer for that.
            if label == "GEMM":
                if _projection(e.key, dims) is None:
                    continue
            elif e.key not in _ATTN_OPS:
                continue
            rows.append((e.self_device_time_total / iters, e.count / iters, e.key, dims))
        rows.sort(key=lambda r: -r[0])
        print(f"  {label}")
        if not rows:
            print("    (none — either this block projects nothing, or its projection "
                  "arrives in a shape no record here can be read as one)")
            continue
        for us, calls, key, dims in rows:
            flops = _shape_flops(key, dims)
            rate = f"{flops * calls / us * 1e-6:6.1f} TF/s" if flops > 0 and us > 0 else " " * 12
            mats = [d for d in dims if len(d) == 2]
            # The trailing 2-D entry an addmm row carries is the result; operands first.
            # bytes per microsecond is megabytes per second, which is the whole conversion.
            mbps = (f"{mats[1][0] * mats[1][1] * weight_bytes * calls / us:7.0f} MB/s"
                    if len(mats) >= 2 and us > 0 else " " * 13)
            print(f"    {us / calls / 1000:8.2f} ms {calls:5.1f}x  {rate} {mbps}  "
                  f"{key.replace('aten::', ''):34s} {' x '.join(str(d) for d in dims if d)}")
        print()
    seen = _GEMM_OPS + _ATTN_OPS
    rest = sorted(((e.self_device_time_total / iters, e.key)
                   for e in prof.key_averages()
                   if e.self_device_time_total > 0 and e.key.startswith(("aten::",))
                   and e.key not in seen), reverse=True)
    if rest:
        print("  other aten ops carrying device time")
        for us, key in rest[:12]:
            print(f"    {us / 1000:8.2f} ms  {key}")


def _projection(op: str, dims: list[list[int]]) -> Optional[tuple[int, int, int]]:
    """``(M, K, N)`` of a recorded GEMM, or None when the record is not one.

    Both operand orientations are read, because both are in use. ``addmm`` is handed the
    weight as a transposed view, so its pair is ``[M, K] x [K, N]``; a projection that hands
    over its stored ``[N, K]`` — ``aten::linear``, and ``comfy_kitchen::int8_linear``, which
    is on no list this script keeps — presents ``[M, K] x [N, K]``. Reading only the first
    called every weight-transposed projection in the engine a non-GEMM, which is how an INT8
    block came back with no GEMMs at all.

    Detection is by shape rather than by name for that reason: a quantized backend names
    its own ops, and a probe that has to be told each one is a probe that has to be told.
    Only dispatched ops qualify — a record whose name carries no namespace is a kernel, and
    an Inductor fusion named after the ``int8_linear`` it swallowed has two matching 2-D
    operands and is not a GEMM.
    """
    if "::" not in op or op in _ATTN_OPS:
        return None
    mats = [d for d in dims if len(d) == 2]
    if len(mats) < 2:
        return None
    a, b = mats[0], mats[1]
    if a[1] == b[0]:
        return a[0], a[1], b[1]
    if a[1] == b[1]:
        return a[0], a[1], b[0]
    return None


def _shape_flops(op: str, dims: list[list[int]]) -> float:
    """FLOPs of one recorded call, from the shapes the op was given.

    ``addmm`` lists its bias ahead of the two operands and the profiler appends the
    scalars, so the operands are the first two 2-D entries. Attention arrives in the
    packed physical layout, where the heads and the sequence can be either middle
    dimension — the rate needs their product, which is the same either way.
    """
    got = _projection(op, dims)
    if got is not None:
        m, k, n = got
        return 2 * m * k * n
    four = [d for d in dims if len(d) == 4]
    if len(four) >= 2:
        q, k = four[0], four[1]
        # The heads axis is dim 1 for the composite op and dim 2 for the packed one the
        # fused kernels are recorded with. Whichever it is, it is the smaller of the two
        # middle axes — and where a short sequence makes that ambiguous the two readings
        # of the formula agree.
        heads, lq = sorted((q[1], q[2]))
        return 4 * heads * q[-1] * lq * max(k[1], k[2])
    return 0.0


# --------------------------------------------------------------------- profile
_BUCKETS = (("attention", ("attn_fwd", "fmha", "flash", "efficient_attention",
                           "cudnn_attention", "softmax")),
            # A triton kernel is Inductor's fused elementwise code whatever it is named
            # after, and these names carry the aten ops the fusion swallowed — including
            # ``scaled_dot_product_attention``, which is not a kernel that ran. Naming it
            # after its inputs is what made 34 ms of rope read as attention.
            ("gemm", ("triton_mm", "triton_bmm", "cijk", "gemm", "gemv", "hipblaslt",
                      "tensile", "wmma", "matmul", "sgemm", "int8", "convrot",
                      "quant")),
            ("pointwise", ("triton_", "elementwise", "vectorized")),
            ("copy", ("copy", "cat", "transpose", "contiguous", "slice", "index")))


def bucket(name: str) -> str:
    low = name.lower()
    for label, marks in _BUCKETS:
        if any(m in low for m in marks):
            return label
    return "other"


_KERNEL_MARKS = ("aten::", "cuda", "hip", "Memcpy", "Memset", "autograd::",
                 "CompiledFunction", "TorchDynamo")


def _kernel(name: str) -> bool:
    """Whether a profiler row is a device kernel rather than the CPU op that launched it.

    A CPU op's ``self_device_time_total`` already carries the kernels it launched, so
    summing both halves the wall clock twice over.
    """
    return not name.startswith(_KERNEL_MARKS)


def _dedupe(rows: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Merge the ``.kd`` twin ROCm reports beside a kernel's execution record."""
    out: dict[str, list[float]] = {}
    for key, us, count in rows:
        name = key[:-3] if key.endswith(".kd") else key
        slot = out.setdefault(name, [0.0, 0.0])
        slot[0] = max(slot[0], us)
        slot[1] = max(slot[1], count)
    return sorted(((k, v[0], v[1]) for k, v in out.items()), key=lambda r: -r[1])


def _renumbered(rows: list[tuple[str, float, float]]) -> dict[str, list[float]]:
    """Kernel rows keyed without the number Inductor gives a fused kernel.

    Two compilations of the same graph emit the same fused kernels under different numbers
    — ``triton_poi_fused_gelu_view_35`` beside ``..._31`` — which in a kernel-by-kernel
    subtraction reads as one vanishing and another appearing. Stripping the suffix pairs
    them, at the cost of merging two genuinely distinct kernels that share an op list.
    """
    out: dict[str, list[float]] = {}
    for key, ms, count in rows:
        slot = out.setdefault(re.sub(r"_\d+$", "", key), [0.0, 0.0])
        slot[0] += ms
        slot[1] += count
    return out


def device_rows(run: Callable[[], object], iters: int) -> list[tuple[str, float, float]]:
    """Per-kernel device milliseconds and launch counts, for one call of the block."""
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        run()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                run()
            torch.cuda.synchronize()
    return _dedupe([(e.key, e.self_device_time_total / iters / 1000, e.count / iters)
                    for e in prof.key_averages()
                    if e.self_device_time_total > 0 and _kernel(e.key)])


def _by_bucket(rows: list[tuple[str, float, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key, ms, _count in rows:
        totals[bucket(key)] = totals.get(bucket(key), 0.0) + ms
    return totals


def buckets_section(arms: list["Arm"], runs: dict[str, Callable], iters: int) -> None:
    """Where each arm's time went, kernel against kernel, beside the baseline.

    The ladder says an arm gave 90 ms back and nothing about which 90. This profiles each
    arm's graph and subtracts: the bucket rows show which kind of work shrank, and the
    movers list names the kernels that appeared and vanished — which is how a gain that
    looks like "compiled SDPA is slow on ROCm" turns into a named cost, or stays a mystery
    in writing rather than in a guess.

    The aten op row is the check on all of it: an arm that stopped being compiled has a
    different graph from the baseline and its deltas describe that, not the change. And
    because this section profiles every arm in one process, where tracing one changes the
    block for the rest, the baseline is profiled a second time at the end as a control.
    """
    tables, ops = {}, {}
    for arm in arms:
        with patches(arm.patches):
            tables[arm.name] = device_rows(runs[arm.name], iters)
            ops[arm.name] = cpu_ops(runs[arm.name])
    with patches(arms[0].patches):
        back = device_rows(runs[arms[0].name], iters)
        back_ops = cpu_ops(runs[arms[0].name])
    names = [a.name for a in arms]
    base = tables[names[0]]
    totals = {n: _by_bucket(t) for n, t in tables.items()}

    print(f"\nbuckets: device ms per block call, {iters} profiled calls each\n")
    print(f"  {'bucket':11s}" + "".join(f"{n:>20s}" for n in names))
    for label in ("gemm", "attention", "copy", "pointwise", "other"):
        line = f"  {label:11s}{totals[names[0]].get(label, 0.0):>20.2f}"
        for n in names[1:]:
            ms = totals[n].get(label, 0.0)
            line += f"{f'{ms:.2f} ({ms - totals[names[0]].get(label, 0.0):+.1f})':>20s}"
        print(line)
    line = f"  {'total':11s}{sum(totals[names[0]].values()):>20.2f}"
    for n in names[1:]:
        ms = sum(totals[n].values())
        line += f"{f'{ms:.2f} ({ms - sum(totals[names[0]].values()):+.1f})':>20s}"
    print(line + "\n")
    print(f"  {'aten ops':11s}{ops[names[0]]:>20d}"
          + "".join(f"{f'{ops[n]} ({ops[n] - ops[names[0]]:+d})':>20s}"
                    for n in names[1:]))
    for n in names[1:]:
        if ops[n] > 1.5 * ops[names[0]]:
            print(f"    {n} does not run the same compiled region as the baseline, so its "
                  f"deltas are about compilation as much as about the work it removed")
    base_ms = sum(totals[names[0]].values())
    back_ms = sum(_by_bucket(back).values())
    print(f"\n  control: the baseline profiled again after every arm: {back_ms:.2f} ms "
          f"against {base_ms:.2f}, {back_ops} aten ops against {ops[names[0]]}")
    if abs(back_ms - base_ms) > max(2.0, 0.03 * base_ms) or back_ops > 1.5 * ops[names[0]]:
        print("    WARNING tracing the arms changed the block, so this table subtracts a "
              "baseline that no longer exists. Give the section one arm at a time.")
    print()

    print("  kernels that moved, |delta| >= 1 ms, against the baseline; names have their "
          "Inductor number stripped, which two compilations give the same kernel")
    for n in names[1:]:
        mine = _renumbered(tables[n])
        seen = _renumbered(base)
        moved = []
        for key in set(mine) | set(seen):
            before, after = seen.get(key, [0.0, 0.0]), mine.get(key, [0.0, 0.0])
            if abs(after[0] - before[0]) >= 1.0:
                moved.append((after[0] - before[0], key, before, after))
        moved.sort()
        print(f"\n    {n}")
        if not moved:
            print("      nothing moved by 1 ms")
        for delta, key, (b_ms, b_n), (a_ms, a_n) in moved:
            short = key if len(key) <= 52 else key[:49] + "..."
            print(f"      {delta:+8.2f} ms  {b_ms:7.2f} -> {a_ms:7.2f}  "
                  f"{b_n:.0f} -> {a_n:.0f} launches  {short}")
    print()


def profile_section(run: Callable[[], object], name: str, iters: int) -> None:
    rows = device_rows(run, iters)
    total = sum(r[1] for r in rows) or 1.0
    print(f"\nprofile: {name}, device time per block call, {total:.1f} ms total, "
          f"{iters} calls\n")
    for key, ms in sorted(_by_bucket(rows).items(), key=lambda kv: -kv[1]):
        print(f"  {key:10s} {ms:9.2f} ms  {100 * ms / total:5.1f}%")
    print(f"\n  {'kernel':66s} {'ms':>8s} {'launches':>9s} {'us/launch':>10s} {'share':>6s}")
    for key, ms, count in rows[:30]:
        short = key if len(key) <= 66 else key[:63] + "..."
        print(f"  {short:66s} {ms:8.2f} {count:9.1f} {1000 * ms / count:10.1f}"
              f" {100 * ms / total:6.1f}%")


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--block", default="ltx/av",
                    help="a camera case, model/block. Any case block_bench can build works;"
                         " one with no arm set of its own gets the generic arms")
    ap.add_argument("--scenario", default=None,
                    help="which of the case's scenarios to build, default its first")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--txt", type=int, default=512)
    ap.add_argument("--weights", default="bf16", choices=bb.WEIGHT_MODES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--section", default="ablate",
                    choices=("ablate", "sdpa", "gemm", "profile", "shapes", "buckets",
                             "sol", "all"))
    ap.add_argument("--sol-tau", default="1.0,1.3,2.0",
                    help="routing thresholds to price; higher routes fewer key blocks "
                         "exactly (the reference's default is 1.3)")
    ap.add_argument("--sol-packs", default="",
                    help="comma-separated subset of " + ", ".join(p[0] for p in SOL_PACKS)
                         + "; default all of them")
    ap.add_argument("--sol-sinks", default="exact_rows", choices=("off", "exact", "exact_rows"),
                    help="the conditioning rows kept out of the routing: none, the packed "
                         "prefix as exact keys, or that plus the audio query rows dense")
    ap.add_argument("--arms", default="", help="comma-separated, in order; default all")
    ap.add_argument("--profile-arm", default="baseline")
    ap.add_argument("--profile-iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--fresh-cache", action="store_true",
                    help="compile into an empty Inductor cache instead of the shared one")
    ap.add_argument("--in-process", action="store_true",
                    help="measure every arm in this one process, which is what degrades "
                         "them: only for a two-arm comparison of things that do not patch "
                         "the same frame")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--json", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--list-arms", action="store_true")
    ap.add_argument("--tunable", type=Path, default=None,
                    help="read this TunableOp file and run the whole thing with its "
                         "kernels: the per-machine half of a GEMM finding. Read-only "
                         "unless --tune says otherwise")
    ap.add_argument("--tune", action="store_true",
                    help="let TunableOp tune problems it has no record for, as it goes")
    ap.add_argument("--tune-out", type=Path, default=None,
                    help="where --section gemm records the kernels it tunes "
                         "(default a file named after the device in the temp dir)")
    ap.add_argument("--no-tune", action="store_true",
                    help="skip --section gemm's two tuned columns: the eager levers cost "
                         "nothing, a TunableOp search costs a minute per GEMM")
    ap.add_argument("--lt-algos", type=int, default=0,
                    help="ask hipBLASLt for this many candidate algorithms per GEMM")
    ap.add_argument("--pad-min-pitch", type=int, default=PAD_MIN_PITCH,
                    help="the input row pitch, in bytes, from which the pitch_pad arm "
                         "starts padding: the engine's bound, 24576, catches every GEMM the "
                         "sweep found paying for its operand pitch")
    ap.add_argument("--pad-min-rows", type=int, default=PITCH_MIN_ROWS,
                    help="rows an operand must have for the pitch_pad arm to widen it; the "
                         "engine uses the same bound, since short streams were measured "
                         "losing to widening. 1 removes it, which is how to price that")
    ap.add_argument("--weight-transposed-min-ratio", type=float,
                    default=TRANSPOSE_MIN_RATIO,
                    help="how much wider in output than input a projection must be for the "
                         "weight_transposed arm to restore its weight transposed")
    ap.add_argument("--transposed-problem-max-ratio", type=float,
                    default=TRANSPOSED_MAX_RATIO,
                    help="how narrow a projection's output must be, against its input, for "
                         "the transposed_problem arm to answer it as C transposed")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print each arm's raw group times")
    args = ap.parse_args()

    raise_recompile_limits()
    if args.fresh_cache:
        print(f"fresh Inductor cache: {fresh_cache()}")
    if args.tunable:
        from torch.cuda import tunable
        tunable.set_filename(str(args.tunable))
        tunable.enable(True)
        tunable.read_file(str(args.tunable))
        tunable.tuning_enable(args.tune)
        rows = tunable.get_results()
        print(f"tuned kernels from {args.tunable}: {len(rows)} problems, "
              f"{sum('Hipblaslt' in row[2] for row in rows)} answered by hipBLASLt, "
              f"{'tuning anything new' if args.tune else 'reading only'}")
    if args.device == "cuda":
        from thenoise.utils.device_patches import apply_device_patches
        apply_device_patches(args.device)

    rules = LayoutRules(min_pitch=args.pad_min_pitch, min_rows=args.pad_min_rows,
                        transpose_min_ratio=args.weight_transposed_min_ratio,
                        transposed_max_ratio=args.transposed_problem_max_ratio)
    if args.list_arms:
        for name, arm in (ARMS[args.block]() if args.block in ARMS
                          else generic_arms(None, rules)).items():
            print(f"{name:{NAME_W}s} {'exact' if arm.exact else 'ablate':6s} {arm.note}")
        return

    ctx = bb.Ctx(device=args.device, txt=args.txt, warmup=args.warmup, iters=args.iters,
                 repeats=args.repeats, weights=args.weights)
    if args.section == "sol":
        with torch.no_grad():
            sol_section(ctx, args.iters, args.repeats,
                        [float(t) for t in args.sol_tau.split(",") if t.strip()],
                        args.sol_packs, args.sol_sinks, args.verbose)
        return
    known = args.block in FIXTURES
    fixture = (FIXTURES[args.block](ctx, args.tokens) if known
               else camera_fixture(ctx, args.block, args.tokens, args.scenario))
    total = sum(fixture.terms.values())
    print(f"{args.block} {args.weights} tokens={args.tokens} txt={args.txt} "
          f"seq={fixture.seq} ({fixture.detail})")
    if fixture.terms:
        print("  terms, TFLOP: " + "  ".join(f"{k}={v / 1e12:.2f}"
                                             for k, v in fixture.terms.items())
              + f"   total={total / 1e12:.2f}")
    wanted = ([a.strip() for a in args.arms.split(",") if a.strip()]
              or list(ARMS[args.block]() if known else generic_arms(fixture.blk, rules)))
    arms = (ARMS[args.block]() if known
            else generic_arms(fixture.blk, rules))
    arm_modules(arms, fixture.blk, rules)
    for name in wanted:
        if name not in arms:
            ap.error(f"unknown arm {name!r}; see --list-arms")
    chosen = [arms[n] for n in wanted]
    if chosen and chosen[0].name != "baseline":
        chosen.insert(0, arms["baseline"])

    runs = {"baseline": fixture.run}
    for arm in chosen:
        if arm.name == "baseline":
            continue
        if arm.inputs:
            with patches(arm.inputs):
                runs[arm.name] = FIXTURES[args.block](ctx, args.tokens).run
        else:
            runs[arm.name] = fixture.run

    # A child measures time and nothing else: kineto collects no device activity in a
    # process whose parent has already touched the GPU, so a kernel table asked for there
    # comes back empty. The one graph line this run gets is the parent's, of the shipped
    # block, before any child exists.
    traced_s = 0.0 if args.child else graph_check(fixture.run, fixture.attentions, "before")
    spawn = (args.section in ("ablate", "all") and len(chosen) > 1
             and not (args.in_process or args.child))

    if args.section in ("ablate", "all") and chosen:
        if args.child:
            results = pair(chosen, runs, ctx, traced_s)
            numerics(chosen, runs)
            args.json.write_text(json.dumps({
                "baseline": results["baseline"].as_json(),
                chosen[1].name: results[chosen[1].name].as_json(),
                "base_again_ms": results["base_again"].ms,
                "drift_pct": results["drift_pct"]}))
            return
        if spawn:
            spawned_ladder(args, chosen, fixture, total)
            return
        if len(chosen) > 1:
            print("  one process measures all of these arms: tracing an arm changes what "
                  "follows it, so read the graph after line before believing the gains")
        sweep(chosen, runs, ctx, traced_s)
        numerics(chosen, runs)
        if args.json:
            args.json.write_text(json.dumps({a.name: a.result.as_json()
                                             for a in chosen if a.result}))
        base = arms["baseline"].result
        print(f"\n{'arm':{NAME_W}s} {'ms':>9s} {'gain%':>7s} {'cov%':>6s} {'GiB':>6s}"
              f" {'compile':>8s} {'gone TFLOP':>11s} {'rate TF/s':>10s} "
              f"{'maxabs':>9s} {'cos':>8s}")
        for arm in chosen:
            r = arm.result
            gone = sum(fixture.terms[t] for t in arm.terms)
            saved = base.ms - r.ms
            rate = gone / saved * 1e-9 if saved > 0.05 and gone else float("nan")
            flag = "" if abs(r.gain_pct) >= FLOOR_PCT else "  <- under the floor"
            print(f"{arm.name:{NAME_W}s} {r.ms:9.2f} {r.gain_pct:7.2f} {r.cov_pct:6.2f}"
                  f" {r.peak_gib:6.2f} {r.compile_s:8.2f} {gone / 1e12:11.3f}"
                  f" {rate:10.1f} {r.maxabs:9.2e} {r.cos:8.5f}{flag}")
        print()
        for arm in chosen:
            print(f"  {arm.name:{NAME_W}s} {arm.note}")
            if args.verbose:
                print(f"  {'':{NAME_W}s} groups: {arm.result.groups}")
        print("\n  rate is what the removed FLOPs were being charged at; compare it with "
              "the block's own (a floor: the terms omit the projections and the rope) "
              + (f"{total / base.ms * 1e-9:.1f} TF/s" if total else
                 "no FLOPs model for this block"))
        graph_check(fixture.run, fixture.attentions, "after")

    if args.section in ("sdpa", "all"):
        if known:
            sdpa_section(ctx, args.tokens, max(3, args.iters), args.verbose)
        else:
            print("\nsdpa: which attention operands a block offers is part of its fixture, "
                  f"and {args.block} has none. The backends answer the same questions "
                  "there as everywhere; someone has to write the list down.")

    if args.section in ("gemm", "all"):
        tuning = args.tune_out or Path(tempfile.gettempdir()) / (
            f"tunableop-{torch.cuda.get_device_name(0).lower().replace(' ', '_')}.csv")
        gemm_section(ctx, fixture.run, args.iters, args.repeats,
                     None if (args.child or args.no_tune) else tuning, args.lt_algos,
                     LTX_GEMMS if known else {})

    if args.section in ("shapes", "all"):
        with patches(arms[args.profile_arm].patches):
            shape_section(fixture.run, args.profile_arm, args.profile_iters,
                          1 if args.weights == "int8" else 2)

    if args.section in ("buckets", "all") and chosen:
        buckets_section(chosen, runs, args.profile_iters)

    if args.section in ("profile", "all"):
        arm = arms[args.profile_arm]
        with patches(arm.patches):
            profile_section(fixture.run, arm.name, args.profile_iters)


if __name__ == "__main__":
    main()
