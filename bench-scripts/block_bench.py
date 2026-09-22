#!/usr/bin/env python
"""Photograph the runtime cost of every DiT transformer block in the repo.

This is a *camera*, not a lab: it measures what the blocks in the working tree do
right now, prints a table and writes a JSON snapshot. There are deliberately no
A/B switches — nothing here patches a model, swaps an attention backend or
toggles compilation. Snapshot before a change, snapshot after it, and a regression is
a row: ``diff_bench.py`` is that diff, and it is the only way to read a snapshot.

What gets photographed
----------------------
One case per ``(block, scenario, token count)``:

* **block** — every transformer block class the engine ships, at its real
  production width, with random (seeded) weights. No checkpoints, no server.
* **scenario** — the paths a block actually runs: ``plain`` (no KV cache) and
  ``fill`` / ``read`` (the reference-token KV cache of ``thenoise.dit.kvcache``,
  which changes the sequence a block attends over and its modulation layout).
* **tokens** — the image-token ladder from ``--tokens``, which is the token grid the
  block sees (the default 4096, 9216, 16384 are 64x64, 96x96 and 128x128). What a
  token is worth in pixels depends on the model's VAE and patch size, so the ladder
  is in tokens only. Entries should be perfect squares, or ``grid_side`` snaps them
  to the nearest one (with a warning).

Inputs are built the way the owning model builds them — the same RoPE builders,
position ids, key-padding masks, modulation rows, 256-token padding and KV
buffers — so each block sees the shapes, layouts and strides it sees in the
server. Text/context length is ``--txt``; editing cases get ``--refs`` reference
images at the target's resolution (``--refs 0`` drops those scenarios). Blocks whose
sequence does not depend on the image ladder (text-only ones) run once, at the first
rung, rather than once per rung under a different key.

Measurement protocol
--------------------
Fixed, and recorded in the JSON: numbers are only comparable between runs of the
same protocol *and* environment, which the JSON records next to them.

* batch 1, ``bfloat16``, ``torch.no_grad()``, the block's own ``@torch.compile``
  forward, in the order printed: every block at one token count, then the next
  count. The first count compiles statically and later ones exercise Dynamo's
  dynamic promotion — the order a server sees.
* weights are seeded from the block's name (so a block gets identical weights at
  every token count and scenario), activations from the case key.
* one call to compile (timed as ``comp``), ``--warmup`` warmup calls, then
  ``--repeats`` groups of ``--iters`` timed calls. ``ms`` is the median of the
  groups, ``±%`` their coefficient of variation — read it before believing a small
  delta. The raw groups go into the JSON too, so a comparison can judge a delta
  against the two runs' own scatter instead of trusting one summary number.
* ``GiB`` is the allocator's peak for that case alone (stats reset per case), and
  ``t_s`` is when a case ran since the start of the run: the ladder runs in order,
  so anything drifting over the several minutes a run takes (clocks, thermals, a
  neighbour) would otherwise be indistinguishable from a token-count effect.
* a case that raises is reported and skipped, so one broken block still leaves a
  snapshot of everything else (the exit code is 1).
* at the end, a *watchlist*: cases over ``±1%``, and blocks that end the ladder
  costing more per FLOP than they started it. In a single snapshot a block that
  decays while its neighbours stay flat is the interesting row, and spotting that
  costs no arithmetic.

A cold ``TORCHINDUCTOR_CACHE_DIR`` costs compile time, never measured time.

Usage
-----
    .venv/bin/python bench-scripts/block_bench.py --list
    .venv/bin/python bench-scripts/block_bench.py                       # everything
    .venv/bin/python bench-scripts/block_bench.py --models krea2,zimage
    .venv/bin/python bench-scripts/block_bench.py --tokens 4096 --refs 0
    .venv/bin/python bench-scripts/block_bench.py --out bench-scripts/snapshots/head.json
    .venv/bin/python bench-scripts/diff_bench.py before.json after.json   # what moved
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

SCHEMA = 2
DEFAULT_TOKENS = "4096,9216,16384"
CACHE_SCENARIOS = ("plain", "fill", "read")

# Environment that silently changes which kernels get measured. Recorded so two
# snapshots that disagree can still be told apart from a real change.
BENCH_VARS = ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "MIOPEN_FIND_MODE",
              "TORCH_BLAS_PREFER_HIPBLASLT", "HSA_OVERRIDE_GFX_VERSION",
              "TORCH_COMPILE_DISABLE")
PLAIN = ("plain",)


# ------------------------------------------------------------------------ cases
@dataclass
class Case:
    """One block's production forward at one shape, plus the FLOPs it should cost."""

    run: Callable[[], object]
    flops: float
    seq: int                 # tokens the block actually processes
    detail: str = ""         # free-form: cache capacity, stream layout, ...


@dataclass(frozen=True)
class Fixture:
    model: str
    block: str
    scenarios: tuple[str, ...]
    build: Callable[["Ctx", int, str], Case]
    fixed_seq: bool = False          # its sequence never depends on --tokens


FIXTURES: list[Fixture] = []


def register(model: str, block: str, build, scenarios: tuple[str, ...] = PLAIN,
             fixed_seq: bool = False) -> None:
    FIXTURES.append(Fixture(model, block, scenarios, build, fixed_seq))


def fixture(model: str, block: str, scenarios: tuple[str, ...] = PLAIN,
            fixed_seq: bool = False):
    def deco(fn):
        register(model, block, fn, scenarios, fixed_seq)
        return fn
    return deco


@dataclass
class Ctx:
    device: str
    txt: int = 512
    refs: int = 1
    warmup: int = 3
    iters: int = 5
    repeats: int = 3
    dtype: torch.dtype = torch.bfloat16

    @property
    def tdev(self) -> torch.device:
        return torch.device(self.device)


# ---------------------------------------------------------------------- helpers
def seed_of(key: str) -> int:
    return zlib.crc32(key.encode()) & 0x7FFF_FFFF


def new_block(ctx: Ctx, name: str, cls, *args, **kwargs):
    """Seeded random weights, on device, compute dtype, eval — as a loaded block is."""
    torch.manual_seed(seed_of(name))
    with torch.device(ctx.device):
        blk = cls(*args, **kwargs).to(ctx.dtype).eval()
    blk.requires_grad_(False)
    return blk


def seed_inputs(name: str, scenario: str, tokens: int) -> None:
    torch.manual_seed(seed_of(f"{name}/{scenario}@{tokens}"))


def randn(ctx: Ctx, *shape) -> torch.Tensor:
    """Seeded activations drawn in fp32 then cast, so values are dtype-independent."""
    return torch.randn(*shape, device=ctx.tdev, dtype=torch.float32).to(ctx.dtype)


def n_refs(ctx: Ctx, scenario: str) -> int:
    """Reference images this scenario runs with (each at the target's resolution)."""
    return ctx.refs if scenario != "plain" else 0


def grid_side(tokens: int) -> int:
    """Square latent side for a token count (snapped, since grids are square)."""
    side = int(round(math.sqrt(max(tokens, 1))))
    if side * side != tokens:
        print(f"  (snapping {tokens} image tokens to {side * side})", file=sys.stderr)
    return side


def gemm_params(module: nn.Module) -> int:
    """Weights of the projections under ``module`` — 2 FLOPs per token per weight."""
    from thenoise.dit.quantized import QuantizedLinear

    return sum(m.weight.numel() + (0 if m.bias is None else m.bias.numel())
               for m in module.modules() if isinstance(m, (nn.Linear, QuantizedLinear)))


def gemm_flops(*pairs: tuple[int, int]) -> float:
    """``(projection weights, tokens fed to them)`` pairs -> FLOPs."""
    return sum(2 * weights * tokens for weights, tokens in pairs)


def attn_flops(heads: int, head_dim: int, pairs: tuple[tuple[int, int], ...]) -> float:
    """``(query tokens, key tokens)`` per attention call -> FLOPs (QK^T + AV, forward)."""
    return sum(4 * heads * head_dim * lq * lk for lq, lk in pairs)


def rope_cache(ctx: Ctx, axes: list[int], theta: float):
    from thenoise.utils.rope import RopeCache, matrix_rope

    return RopeCache(matrix_rope(list(axes), theta))


def pe_one(ctx: Ctx, axes: list[int], theta: float, ids: torch.Tensor) -> torch.Tensor:
    """The model's own RoPE path for a single frequency table."""
    return rope_cache(ctx, axes, theta).store("bench", ids, dtype=ctx.dtype)


def new_cache(cached_slice: str):
    from thenoise.dit.kvcache import KVCache

    return KVCache("bench", cached_slice)


def cache_buffers(cache, key, mode: str, shape, ctx: Ctx, prime: bool = False):
    """``KVCache.buffers`` exactly as a model calls it (it marks the token axis too).

    ``prime`` fills the buffers: the frozen slice a ``read`` attends over holds
    reference K/V in production, and uninitialized memory would hand the attention
    denormals and silently lie about its speed.
    """
    bufs = cache.buffers(key, mode, shape, ctx.dtype, ctx.tdev)
    if prime:
        with torch.no_grad():
            bufs.k.normal_()
            bufs.v.normal_()
    return bufs


def causal_prefix_mask(rows: int, cols: int, ctx: Ctx) -> torch.Tensor:
    """``[1, 1, rows, cols]`` bool (True = attend): row i reads keys ``0..i + (cols-rows)``.

    The shape Qwen-Image 2.1 gives a text segment that joins ``cols - rows`` tokens it
    can already see (``torch.ones(n, length + n).tril(length)`` in ``build_sequence``).
    """
    mask = torch.ones(rows, cols, dtype=torch.bool, device=ctx.tdev).tril(cols - rows)
    return mask[None, None]


# --------------------------------------------------------------------- fixtures
@fixture("qwen_image21", "block", CACHE_SCENARIOS)
def qwen_image21_block(ctx: Ctx, tokens: int, scenario: str) -> Case:
    """Qwen-Image 2.1: single-stream block over a block-causal [text, refs, target]."""
    from thenoise.dit.qwen_image21.models import (
        AttentionPlan,
        QwenImage21TransformerBlock,
        modulation_rows,
    )
    from thenoise.utils.positions import broadcast_positions, grid_from_axes

    name = "qwen_image21/block"
    heads, head_dim = 32, 128                       # QwenImage21Params defaults
    dim = heads * head_dim
    side = grid_side(tokens)
    tokens = side * side
    txt = ctx.txt
    refs = n_refs(ctx, scenario)
    read = scenario == "read"

    blk = new_block(ctx, name, QwenImage21TransformerBlock, dim, heads, head_dim, mlp_ratio=3)
    seed_inputs(name, scenario, tokens)

    # Positions as build_sequence lays them out: text first, then one centred grid
    # per reference, then the target grid, each image advancing the t axis by max(h,w).
    def image_ids(pos: int) -> torch.Tensor:
        axis = torch.arange(side, device=ctx.tdev, dtype=torch.float32) - (side - side // 2)
        return grid_from_axes([torch.full((1,), float(pos), device=ctx.tdev), axis, axis])

    pos, ids = txt, [broadcast_positions(txt, 3, device=ctx.tdev)]
    for _ in range(refs):
        ids.append(image_ids(pos))
        pos += side
    ids.append(image_ids(pos))
    pe_full = pe_one(ctx, [16, 56, 56], 10000.0, torch.cat(ids, dim=0)[None])

    ref_tokens = refs * tokens
    full = txt + ref_tokens + tokens
    prefix = txt + ref_tokens if refs else 0
    n = tokens if read else full
    x = randn(ctx, 1, n, dim)
    pe = pe_full[:, prefix:] if read else pe_full

    # Block-causal segments: the text reads causally, each image segment reads
    # everything up to its own end (so no mask is needed).
    segments: list[tuple[int, int, Optional[torch.Tensor]]] = [
        (0, txt, causal_prefix_mask(txt, txt, ctx))]
    keys: list[tuple[int, int]] = [(txt, txt)]
    cursor = txt
    for _ in range(refs):
        segments.append((cursor, cursor + tokens, None))
        keys.append((tokens, cursor + tokens))
        cursor += tokens
    segments.append((cursor, full, None))

    rows = randn(ctx, 2, dim)
    mods = tuple(modulation_rows(r, 0 if read else prefix, n)
                 for r in (rows, rows.tanh(), rows, rows.tanh()))

    plan_args: dict = {"mode": scenario, "segments": () if read else tuple(segments)}
    if scenario != "plain":
        cache = new_cache("prefix")                 # the cached slice is the text+ref prefix
        cache_buffers(cache, ("block", 0), "fill", (1, heads, full, head_dim), ctx)
        if read:
            cache.set_filled()
        plan_args["bufs"] = cache_buffers(cache, ("block", 0), scenario,
                                          (1, heads, n, head_dim), ctx, prime=read)
    plan = AttentionPlan(**plan_args)

    def run():
        return blk(x, *mods, pe, plan)

    # The read pass asks the target's questions of the cached prefix instead of the
    # live tokens, and skips the text segment (its K/V is what was cached).
    if read:
        keys = [(tokens, full)]
    flops = gemm_flops((gemm_params(blk), n)) + attn_flops(heads, head_dim, tuple(keys))
    return Case(run, flops, n, detail=f"prefix={prefix} cache={full if refs else 0}")


@fixture("qwen_image", "block", CACHE_SCENARIOS)
def qwen_image_block(ctx: Ctx, tokens: int, scenario: str) -> Case:
    """Qwen-Image dual-stream block: joint attention over [text, target, refs]."""
    from thenoise.dit.qwen_image.models import (
        QwenImageTransformerBlock,
        build_txt_positions,
        build_video_positions,
    )
    from thenoise.utils.dynamo import mark_token_axis

    name = "qwen_image/block"
    heads, head_dim = 24, 128                       # create_model(): 24 x 128 = 3072
    dim = heads * head_dim
    side = grid_side(tokens)
    tokens = side * side
    txt = ctx.txt
    nref = n_refs(ctx, scenario)
    refs = nref * tokens
    read = scenario == "read"

    blk = new_block(ctx, name, QwenImageTransformerBlock, dim=dim, heads=heads,
                    attention_head_dim=head_dim, eps=1e-5)
    seed_inputs(name, scenario, tokens)

    # Target and reference positions are stored apart, which is what lets ``read``
    # drop the references from the sequence while their cached K/V stays put.
    shapes = [(1, side, side)] * (1 + nref)
    img_pos = build_video_positions(shapes, device=ctx.tdev)
    max_vid_index = max(max(h // 2, w // 2) for _, h, w in shapes)
    rope = rope_cache(ctx, [16, 56, 56], 10000.0)
    img_pe = rope.store("img", img_pos[:, :tokens], dtype=ctx.dtype)
    if refs and not read:
        img_pe = torch.cat([img_pe, rope.store("ref", img_pos[:, tokens:], dtype=ctx.dtype)],
                           dim=1)
    txt_pe = rope.store("txt", build_txt_positions(max_vid_index, txt, device=ctx.tdev),
                        dtype=ctx.dtype)

    img_tokens = tokens if read else tokens + refs
    x = randn(ctx, 1, img_tokens, dim)
    txt_x = randn(ctx, 1, txt, dim)
    if refs and not read:
        # Timestep-zero conditioning: temb carries the t row then the t=0 row, and the
        # token mask picks which one each image token modulates from.
        temb = randn(ctx, 2, dim)
        token_mask = (torch.arange(img_tokens, device=ctx.tdev) < tokens)[None, :, None]
    else:
        temb, token_mask = randn(ctx, 1, dim), None

    bufs = None
    if scenario != "plain":
        cache = new_cache("suffix")                 # layout is text, target, references
        cache_buffers(cache, 0, "fill", (1, heads, txt + tokens + refs, head_dim), ctx)
        if read:
            cache.set_filled()
        bufs = cache_buffers(cache, 0, scenario, (1, heads, txt + img_tokens, head_dim),
                             ctx, prime=read)

    def run():
        mark_token_axis(x, txt_x, img_pe, txt_pe, token_mask)
        return blk(x, txt_x, temb, img_pe, txt_pe, token_mask, bufs)

    img_side = gemm_params(blk.img_mlp) + sum(gemm_params(getattr(blk.attn, p))
                                              for p in ("to_q", "to_k", "to_v", "to_out"))
    txt_side = gemm_params(blk.txt_mlp) + sum(gemm_params(getattr(blk.attn, p))
                                              for p in ("add_q_proj", "add_k_proj",
                                                        "add_v_proj", "to_add_out"))
    joint = txt + tokens + refs                     # ``read`` attends the cached refs too
    flops = gemm_flops((img_side, img_tokens), (txt_side, txt)) + attn_flops(
        heads, head_dim, ((joint, joint),))
    return Case(run, flops, img_tokens + txt,
                detail=f"img={img_tokens} txt={txt} refs={refs}")


def _flux2_pe(ctx: Ctx, params, side: int, nref: int, read: bool):
    """Target(+reference) and text frequencies through the model's own ``prc_*`` ids."""
    from thenoise.dit.flux2.sampling import prc_img, prc_txt

    latent = torch.zeros(1, 4, side, side, device=ctx.tdev)
    x_ids = prc_img(latent)[1]
    pe_ctx = pe_one(ctx, params.axes_dim, params.theta,
                    prc_txt(torch.zeros(1, ctx.txt, 4, device=ctx.tdev))[1])
    if nref and not read:
        ref_ids = torch.cat([
            prc_img(latent, t_coord=torch.tensor([i + 1], device=ctx.tdev))[1]
            for i in range(nref)
        ], dim=1)
        x_ids = torch.cat([x_ids, ref_ids], dim=1)
    return pe_one(ctx, params.axes_dim, params.theta, x_ids), pe_ctx


def _flux2_mod(groups: int, dim: int, ctx: Ctx, tokens: int = 0, refs: int = 0,
               zero: bool = False):
    """``groups`` ``(shift, scale, gate)`` triples: broadcast rows, or per-token ones.

    ``zero`` is ``zero_cond_t``: the reference slice modulates from the ``t = 0`` row,
    which the model materialises as per-token rows over the image stream. Without it
    one broadcast row covers every token, so ``tokens``/``refs`` are unused.
    """
    out = []
    for _ in range(groups):
        if not zero:
            out.append(tuple(randn(ctx, 1, 1, dim) for _ in range(3)))
            continue
        out.append(tuple(
            torch.cat([randn(ctx, 1, 1, dim).expand(1, tokens, -1),
                       randn(ctx, 1, 1, dim).expand(1, refs, -1)], dim=1)
            for _ in range(3)))
    return tuple(out)


def _flux2_double(ctx: Ctx, tokens: int, scenario: str, params, model: str) -> Case:
    from thenoise.dit.flux2.models import DoubleStreamBlock
    from thenoise.utils.dynamo import mark_token_axis

    name = f"{model}/double"
    heads, dim = params.num_heads, params.hidden_size
    head_dim = dim // heads
    side = grid_side(tokens)
    tokens = side * side
    txt = ctx.txt
    nref = n_refs(ctx, scenario)
    refs = nref * tokens
    read = scenario == "read"

    blk = new_block(ctx, name, DoubleStreamBlock, dim, heads, mlp_ratio=params.mlp_ratio)
    seed_inputs(name, scenario, tokens)
    pe_x, pe_ctx = _flux2_pe(ctx, params, side, nref, read)

    img_tokens = tokens if read else tokens + refs
    img, txt_x = randn(ctx, 1, img_tokens, dim), randn(ctx, 1, txt, dim)
    zero = bool(refs) and not read
    mod_img = _flux2_mod(2, dim, ctx, tokens, refs, zero)
    mod_txt = _flux2_mod(2, dim, ctx)                 # the text stream has no ref slice

    bufs = None
    if scenario != "plain":
        cache = new_cache("suffix")
        cache_buffers(cache, ("double", 0), "fill",
                      (1, heads, txt + tokens + refs, head_dim), ctx)
        if read:
            cache.set_filled()
        bufs = cache_buffers(cache, ("double", 0), scenario,
                             (1, heads, txt + img_tokens, head_dim), ctx, prime=read)

    def run():
        mark_token_axis(img, txt_x, pe_x, pe_ctx)
        return blk(img, txt_x, pe_x, pe_ctx, mod_img, mod_txt, bufs)

    joint = txt + tokens + refs                     # ``read`` attends the cached refs too
    flops = gemm_flops(
        (gemm_params(blk.img_attn) + gemm_params(blk.img_mlp), img_tokens),
        (gemm_params(blk.txt_attn) + gemm_params(blk.txt_mlp), txt),
    ) + attn_flops(heads, head_dim, ((joint, joint),))
    return Case(run, flops, img_tokens + txt,
                detail=f"img={img_tokens} txt={txt} refs={refs}")


def _flux2_single(ctx: Ctx, tokens: int, scenario: str, params, model: str) -> Case:
    from thenoise.dit.flux2.models import SingleStreamBlock
    from thenoise.utils.dynamo import mark_token_axis

    name = f"{model}/single"
    heads, dim = params.num_heads, params.hidden_size
    head_dim = dim // heads
    side = grid_side(tokens)
    tokens = side * side
    txt = ctx.txt
    nref = n_refs(ctx, scenario)
    refs = nref * tokens
    read = scenario == "read"

    blk = new_block(ctx, name, SingleStreamBlock, dim, heads, mlp_ratio=params.mlp_ratio)
    seed_inputs(name, scenario, tokens)
    pe_x, pe_ctx = _flux2_pe(ctx, params, side, nref, read)
    pe = torch.cat([pe_ctx, pe_x], dim=1)          # the single stream runs on [txt, img]

    img_tokens = tokens if read else tokens + refs
    x = randn(ctx, 1, txt + img_tokens, dim)
    mod = _flux2_mod(1, dim, ctx, txt + tokens, refs, bool(refs) and not read)[0]

    bufs = None
    if scenario != "plain":
        cache = new_cache("suffix")
        cache_buffers(cache, ("single", 0), "fill",
                      (1, heads, txt + tokens + refs, head_dim), ctx)
        if read:
            cache.set_filled()
        bufs = cache_buffers(cache, ("single", 0), scenario,
                             (1, heads, txt + img_tokens, head_dim), ctx, prime=read)

    def run():
        mark_token_axis(x, pe)
        return blk(x, pe, mod, bufs)

    joint = txt + tokens + refs
    flops = gemm_flops((gemm_params(blk), txt + img_tokens)) + attn_flops(
        heads, head_dim, ((joint, joint),))
    return Case(run, flops, txt + img_tokens, detail=f"txt+img={txt + img_tokens}")


@fixture("krea2", "block")
def krea2_block(ctx: Ctx, tokens: int, scenario: str) -> Case:
    """Krea 2 single-stream block: GQA attention, gated output, [image, text] stream."""
    import torch.nn.functional as F
    from thenoise.dit.krea2.mmdit import SingleStreamBlock
    from thenoise.dit.krea2.sampling import prepare
    from thenoise.dit.krea2.utils import single_mmdit_large_wide as cfg
    from thenoise.utils.attention import AttentionParams

    name = "krea2/block"
    heads, kv_heads = cfg.heads, cfg.kvheads or cfg.heads
    features, head_dim = cfg.features, cfg.features // cfg.heads
    side = grid_side(tokens)
    tokens = side * side
    txt = ctx.txt

    blk = new_block(ctx, name, SingleStreamBlock, features, heads, cfg.multiplier,
                    cfg.bias, cfg.kvheads)
    seed_inputs(name, scenario, tokens)

    # Positions and key-padding mask from the model's own ``prepare`` (image tokens
    # first, so a sample's valid tokens are a contiguous prefix), then the same
    # pad-to-256 the DiT applies around its block loop.
    txtmask = torch.ones(1, txt, dtype=torch.bool, device=ctx.tdev)
    _, pos, full_mask = prepare(
        torch.zeros(1, cfg.channels, 2 * side, 2 * side, device=ctx.tdev), txt, cfg.patch,
        txtmask)
    axes = [head_dim - 12 * (head_dim // 16), 6 * (head_dim // 16), 6 * (head_dim // 16)]
    freqs = pe_one(ctx, axes, cfg.theta, pos)

    x = torch.cat([randn(ctx, 1, tokens, features), randn(ctx, 1, txt, features)], dim=1)
    txt_mask = full_mask[:, tokens:]                # the text tail, as the DiT slices it
    padlen = (-x.shape[1]) % 256
    if padlen:
        x = F.pad(x, (0, 0, 0, padlen))
        txt_mask = F.pad(txt_mask, (0, padlen), value=False)
        freqs = F.pad(freqs, (0, 0, 0, 0, 0, 0, 0, padlen, 0, 0))
    vec = randn(ctx, 1, 6 * features)
    params = AttentionParams.create_attention_params_from_mask(tokens, txt_mask)

    def run():
        return blk(x, vec, freqs, params)

    seq = x.shape[1]
    flops = gemm_flops((gemm_params(blk.attn) + gemm_params(blk.mlp), seq)) + attn_flops(
        heads, head_dim, ((seq, seq),))            # k/v are expanded to q's head count
    return Case(run, flops, seq,
                detail=f"img={tokens} txt={txt} pad={padlen} kv_heads={kv_heads}")


@fixture("krea2", "text_fusion", ("nomask", "masked"), fixed_seq=True)
def krea2_text_fusion(ctx: Ctx, tokens: int, scenario: str) -> Case:
    """Krea 2's text-fusion block: the layerwise ones run unmasked, the refiners masked."""
    from thenoise.dit.krea2.mmdit import TextFusionBlock
    from thenoise.dit.krea2.utils import single_mmdit_large_wide as cfg
    from thenoise.utils.attention import AttentionParams

    name = "krea2/text_fusion"
    heads, features = cfg.txtheads, cfg.txtdim
    head_dim = features // heads
    blk = new_block(ctx, name, TextFusionBlock, features, heads, cfg.multiplier,
                    cfg.bias, cfg.txtkvheads)
    seed_inputs(name, scenario, tokens)
    x = randn(ctx, 1, ctx.txt, features)
    if scenario == "nomask":
        params = AttentionParams.create_attention_params()
    else:
        mask = torch.ones(1, ctx.txt, dtype=torch.bool, device=ctx.tdev)
        mask[:, -64:] = False                      # the prompt's padded tail
        params = AttentionParams.create_attention_params_from_mask(0, mask)

    def run():
        return blk(x, params)

    seq = ctx.txt
    flops = gemm_flops((gemm_params(blk.attn) + gemm_params(blk.mlp), seq)) + attn_flops(
        heads, head_dim, ((seq, seq),))
    return Case(run, flops, seq, detail=f"txt={ctx.txt}")


@fixture("anima", "block")
def anima_block(ctx: Ctx, tokens: int, scenario: str) -> Case:
    """Anima (Cosmos-Predict2) block: self-attn + cross-attn + MLP, AdaLN-LoRA, 3D rope."""
    from thenoise.dit.anima.models import Block
    from thenoise.utils.attention import AttentionParams
    from thenoise.utils.rope import split_half_rope_3d

    name = "anima/block"
    x_dim, heads, context_dim = 2048, 16, 1024     # the config anima/utils.py pins
    head_dim = x_dim // heads
    side = grid_side(tokens)
    tokens = side * side

    blk = new_block(ctx, name, Block, x_dim=x_dim, context_dim=context_dim,
                    num_heads=heads, mlp_ratio=4.0, use_adaln_lora=True, adaln_lora_dim=256)
    seed_inputs(name, scenario, tokens)

    x = randn(ctx, 1, 1, side, side, x_dim)        # B, T, H, W, D with T = 1 for images
    emb = randn(ctx, 1, 1, x_dim)
    adaln_lora = randn(ctx, 1, 1, 3 * x_dim)
    cross = randn(ctx, 1, ctx.txt, context_dim)
    # ``pos_embedder.store("emb", latents.shape, device)``: the raw latent grid, which
    # the rope builder divides by the patch size to land on side * side tokens.
    cos, sin = (t.to(ctx.dtype) for t in split_half_rope_3d(head_dim, 2, 1, 4.0, 4.0, 1.0)(
        (1, 16, 1, 2 * side, 2 * side), ctx.tdev))
    params = AttentionParams.create_attention_params()

    def run():
        return blk(x, emb, cross, params, (cos, sin), adaln_lora, None)

    adaln = sum(gemm_params(getattr(blk, f"adaln_modulation_{n}"))
                for n in ("self_attn", "cross_attn", "mlp"))
    flops = gemm_flops(
        (gemm_params(blk.self_attn) + gemm_params(blk.mlp), tokens),
        (gemm_params(blk.cross_attn.q_proj) + gemm_params(blk.cross_attn.output_proj), tokens),
        (gemm_params(blk.cross_attn.k_proj) + gemm_params(blk.cross_attn.v_proj), ctx.txt),
        (adaln, 1),                                 # one modulation row per batch item
    ) + attn_flops(heads, head_dim, ((tokens, tokens), (tokens, ctx.txt)))
    return Case(run, flops, tokens, detail=f"ctx={ctx.txt}")


def _zimage_ids(ctx: Ctx, sizes: list[int], start: int) -> torch.Tensor:
    """Z-Image position ids for one stream, including the padding's zero positions.

    Mirrors ``_pad_with_ids``: the real grid first, then the (0,0,0) position repeated
    out to ``SEQ_MULTI_OF``, and the image grid starts after the padded caption.
    """
    from thenoise.dit.zimage.models import SEQ_MULTI_OF
    from thenoise.utils.positions import grid_positions

    ids = grid_positions(sizes, start=[start, 0, 0], dtype=torch.int32, device=ctx.tdev)
    pad = (-ids.shape[0]) % SEQ_MULTI_OF
    return torch.cat([ids, ids[:1].repeat(pad, 1)], dim=0)


def _zimage_case(ctx: Ctx, tokens: int, block: str, modulation: bool) -> Case:
    from thenoise.dit.zimage.models import ZImageTransformerBlock

    name = f"zimage/{block}"
    dim, heads, head_dim = 3840, 30, 128           # the config zimage/utils.py pins
    axes, theta = (32, 48, 48), 256.0
    side = grid_side(tokens)
    tokens = side * side

    blk = new_block(ctx, name, ZImageTransformerBlock, 0, dim, heads, heads, 1e-5,
                    modulation=modulation)
    seed_inputs(name, block, tokens)

    # Each stream is padded to a multiple of 32 and the image grid starts after the
    # caption; a batch of one needs no key-padding mask (``make_key_padding_mask``
    # returns None when every sequence is the batch max).
    cap_ids = _zimage_ids(ctx, [ctx.txt, 1, 1], 1)
    img_ids = _zimage_ids(ctx, [1, side, side], cap_ids.shape[0] + 1)
    if block == "context_refiner":
        ids = cap_ids
    elif block == "noise_refiner":
        ids = img_ids
    else:                                          # the unified [image, caption] stream
        ids = torch.cat([img_ids, cap_ids], dim=0)
    freqs = pe_one(ctx, axes, theta, ids[None])
    n = ids.shape[0]
    x = randn(ctx, 1, n, dim)
    adaln = randn(ctx, 1, min(dim, 256)) if modulation else None

    def run():
        return blk(x, None, freqs, adaln)

    flops = gemm_flops(
        (gemm_params(blk.attention) + gemm_params(blk.feed_forward), n),
        (gemm_params(blk.adaLN_modulation) if modulation else 0, 1),
    ) + attn_flops(heads, head_dim, ((n, n),))
    return Case(run, flops, n, detail=f"stream={block} modulation={modulation}")


@fixture("zimage", "layer")
def zimage_layer(ctx: Ctx, tokens: int, scenario: str) -> Case:
    return _zimage_case(ctx, tokens, "layer", modulation=True)


@fixture("zimage", "noise_refiner")
def zimage_noise_refiner(ctx: Ctx, tokens: int, scenario: str) -> Case:
    return _zimage_case(ctx, tokens, "noise_refiner", modulation=True)


@fixture("zimage", "context_refiner", fixed_seq=True)
def zimage_context_refiner(ctx: Ctx, tokens: int, scenario: str) -> Case:
    return _zimage_case(ctx, tokens, "context_refiner", modulation=False)


def register_flux2() -> None:
    """One pair of fixtures per shipped Flux.2 width (``detect_klein_params`` picks these)."""
    from thenoise.dit.flux2.models import Klein4BParams, Klein9BParams

    for model, params in (("flux2_klein9b", Klein9BParams()), ("flux2_klein4b", Klein4BParams())):
        register(model, "double",
                 lambda ctx, tokens, scenario, p=params, m=model:
                 _flux2_double(ctx, tokens, scenario, p, m), CACHE_SCENARIOS)
        register(model, "single",
                 lambda ctx, tokens, scenario, p=params, m=model:
                 _flux2_single(ctx, tokens, scenario, p, m), CACHE_SCENARIOS)


register_flux2()


# ------------------------------------------------------------------ measurement
@dataclass
class Timing:
    """What one case cost. ``groups`` is the raw evidence behind ``ms`` and ``cov_pct``."""

    ms: float
    cov_pct: float
    groups: list[float]
    peak_gib: float
    compile_s: float


def measure(run, ctx: Ctx) -> Timing:
    """One call to compile, ``ctx.warmup`` warmups, then ``repeats`` x ``iters`` timed."""
    sync = torch.cuda.synchronize if ctx.device == "cuda" else (lambda: None)
    with torch.no_grad():
        t0 = time.perf_counter()
        run()                                      # compiles
        sync()
        compile_s = time.perf_counter() - t0
        for _ in range(ctx.warmup):
            run()
        sync()
        if ctx.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        groups = []
        for _ in range(ctx.repeats):
            t0 = time.perf_counter()
            for _ in range(ctx.iters):
                run()
            sync()
            groups.append((time.perf_counter() - t0) * 1000 / ctx.iters)
    groups.sort()
    mean = sum(groups) / len(groups)
    var = sum((g - mean) ** 2 for g in groups) / len(groups)
    peak = torch.cuda.max_memory_allocated() / 2**30 if ctx.device == "cuda" else 0.0
    return Timing(groups[len(groups) // 2], 100 * math.sqrt(var) / mean,
                  [round(g, 4) for g in groups], peak, compile_s)


def watchlist(results: list[dict], cov_warn: float = 1.0, decay_pct: float = 25.0,
              show: int = 5) -> None:
    """Rows worth a second look inside *this* snapshot, printed as they are found.

    Two cheap rules, no judgement: a case that was not stable (``±%`` over
    ``cov_warn``), and a block that ends the ladder costing more per FLOP than it
    started it. Blocks that get *better* along the ladder are only counted, never
    listed — that is normal, since attention's share of the FLOPs grows with length —
    so what is left is decay, the one thing a single snapshot cannot show otherwise.
    """
    ok = [r for r in results if r["status"] == "ok"]
    noisy = sorted((r for r in ok if r["cov_pct"] > cov_warn), key=lambda r: -r["cov_pct"])
    series: dict[tuple[str, str, str], list[dict]] = {}
    for r in ok:
        series.setdefault((r["model"], r["block"], r["scenario"]), []).append(r)
    decaying, gaining = [], 0
    for key, rows in series.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: r["image_tokens"])
        lost = 100 * (1 - rows[-1]["tflops"] / max(r["tflops"] for r in rows))
        if lost > decay_pct:
            decaying.append((lost, key, rows))
        elif 100 * (rows[-1]["tflops"] / rows[0]["tflops"] - 1) > decay_pct:
            gaining += 1
    if not decaying and not noisy:
        print(f"\nwatchlist: clean (no block losing {decay_pct:.0f}% of its TFLOPS along "
              f"the ladder, every case under ±{cov_warn}%)")
        return
    print("\nwatchlist")
    for lost, key, rows in sorted(decaying, key=lambda d: -d[0]):
        track = "  ".join(f"{r['image_tokens']}:{r['tflops']:.1f}" for r in rows)
        print(f"  {'/'.join(key):38s} loses {lost:3.0f}% of its TFLOPS  {track}")
    if gaining:
        print(f"  {'':38s} ({gaining} block(s) gain {decay_pct:.0f}% along the ladder "
              f"instead — normal, attention's share grows)")
    for r in noisy[:show]:
        print(f"  {'/'.join((r['model'], r['block'], r['scenario'])):38s} "
              f"unstable: ±{r['cov_pct']:.1f}% at {r['image_tokens']} tokens")
    if len(noisy) > show:
        print(f"  {'':38s} (+{len(noisy) - show} more over ±{cov_warn}%)")


def git_info() -> dict:
    def git(*cmd) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True,
                                  timeout=10).stdout.strip()
        except Exception:                                       # noqa: BLE001
            return ""

    return {"sha": git("git", "rev-parse", "HEAD"),
            "describe": git("git", "describe", "--always", "--dirty"),
            "dirty": bool(git("git", "status", "--porcelain"))}


def env_info(device: str) -> dict:
    info = {"python": sys.version.split()[0],
            "torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
            "device": device}
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["arch"] = getattr(props, "gcnArchName", "") or ""
        info["total_gib"] = round(props.total_memory / 2**30, 1)
    info["vars"] = {k: os.environ[k] for k in BENCH_VARS if k in os.environ}
    return info


def device_slug(info: dict) -> str:
    slug = (info.get("arch") or info.get("gpu") or info["device"]).split("(")[0].strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", slug) or "device"


# -------------------------------------------------------------------------- CLI
def select(models: str) -> list[Fixture]:
    """Fixtures whose ``model/block`` key matches a comma-separated prefix filter."""
    wanted = [w.strip() for w in models.split(",") if w.strip()]

    def matches(f: Fixture, w: str) -> bool:
        key = f"{f.model}/{f.block}"
        return key.startswith(w) or w.startswith(key) or w == f.block or w == f.model

    keep = [f for f in FIXTURES if any(matches(f, w) for w in wanted)]
    if not keep:
        raise SystemExit(f"no block matches {models}\nknown: "
                         f"{', '.join(f.model + '/' + f.block for f in FIXTURES)}")
    return keep


def case_list(fixtures: list[Fixture], tokens: list[int], refs: int):
    """Canonical order: every block at one token count, then the next count.

    A block whose sequence does not depend on the ladder is text-only, so the other
    rungs would repeat the same measurement under a different key: it runs once.
    """
    return [(f, t, s) for t in tokens for f in fixtures
            if t == tokens[0] or not f.fixed_seq
            for s in (f.scenarios if refs else ("plain",))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Snapshot the cost of every DiT transformer block.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="blocks: " + ", ".join(f"{f.model}/{f.block}" for f in FIXTURES))
    ap.add_argument("--models", help="comma-separated model or block filter, e.g. krea2,zimage")
    ap.add_argument("--tokens", default=DEFAULT_TOKENS,
                    help="image-token ladder (default %(default)s)")
    ap.add_argument("--txt", type=int, default=512, help="text / context tokens")
    ap.add_argument("--refs", type=int, default=1,
                    help="reference images per editing case; 0 drops fill and read")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=3, help="timed calls per group")
    ap.add_argument("--repeats", type=int, default=3, help="groups per case (median reported)")
    ap.add_argument("--warmup", type=int, default=1, help="calls after the compiling one")
    ap.add_argument("--out", help="JSON snapshot path (default bench-scripts/snapshots/"
                                  "<device>-<timestamp>.json)")
    ap.add_argument("--list", action="store_true", help="print the case matrix and exit")
    args = ap.parse_args()

    tokens = sorted(int(t) for t in args.tokens.split(",") if t.strip())
    if args.refs < 0 or not tokens:
        raise SystemExit("--refs must be >= 0 and --tokens must not be empty")
    fixtures = select(args.models) if args.models else list(FIXTURES)
    cases = case_list(fixtures, tokens, args.refs)
    if args.list:
        print("\n".join(f"{f.model}/{f.block}/{s}@{t}" for f, t, s in cases))
        print(f"\n{len(cases)} cases: {len(fixtures)} blocks over "
              f"{len(tokens)} ladder rungs")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no cuda/rocm device visible — run this on the GPU box")
    ctx = Ctx(device=args.device, txt=args.txt, refs=args.refs, warmup=args.warmup,
              iters=args.iters, repeats=args.repeats)

    env = env_info(args.device)
    print(f"{env.get('gpu', args.device)} | torch {torch.__version__} | hip {env.get('hip')}")
    print(f"bf16 | batch 1 | ladder {','.join(map(str, tokens))} | txt {ctx.txt} "
          f"| refs {ctx.refs} | warmup {ctx.warmup} + {ctx.repeats}x{ctx.iters} timed "
          f"| {len(cases)} cases")

    results, failed, last_tokens, case = [], 0, None, None
    started = time.perf_counter()
    for f, t, scenario in cases:
        if t != last_tokens:
            print(f"\n=== {t} image tokens ===")
            print(f"  {'block/scenario':32s} {'seq':>7s} {'ms':>8s} {'±%':>5s} "
                  f"{'TFLOPS':>7s} {'GiB':>6s} {'comp':>6s}")
            last_tokens = t
        label = f"{f.model}/{f.block}/{scenario}"
        entry = {"key": f"{label}@{t}", "model": f.model, "block": f.block,
                 "scenario": scenario, "image_tokens": t}
        # The first call of a case compiles it, which can take a minute: print the row
        # label first so a long compile is visibly progress rather than a hang.
        print(f"  {label:32s}", end="", flush=True)
        try:
            case = f.build(ctx, t, scenario)
            tim = measure(case.run, ctx)
            tflops = case.flops / (tim.ms / 1000) / 1e12
            print(f" {case.seq:7d} {tim.ms:8.2f} {tim.cov_pct:5.1f} {tflops:7.1f} "
                  f"{tim.peak_gib:6.2f} {tim.compile_s:6.1f}", flush=True)
            entry.update({"status": "ok", "seq": case.seq, "ms": round(tim.ms, 4),
                          "cov_pct": round(tim.cov_pct, 3), "groups": tim.groups,
                          "compile_s": round(tim.compile_s, 2),
                          "tflop": round(case.flops / 1e12, 4),
                          "tflops": round(tflops, 2), "peak_gib": round(tim.peak_gib, 3),
                          "detail": case.detail})
        except Exception as exc:                                # noqa: BLE001
            print(f" {'FAILED':>7s}  {type(exc).__name__}: {str(exc)[:110]}", flush=True)
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            failed += 1
        if f.fixed_seq:
            entry["fixed_seq"] = True                             # run at one rung only
        entry["t_s"] = round(time.perf_counter() - started, 1)     # drift is not a shape
        results.append(entry)
        case = None
        gc.collect()
        if ctx.device == "cuda":
            torch.cuda.empty_cache()

    watchlist(results)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots",
                                   f"{device_slug(env)}-{time.strftime('%Y%m%d-%H%M')}.json")
    snapshot = {
        "tool": "block_bench",
        "schema": SCHEMA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "git": git_info(),
        "env": env,
        "protocol": {"dtype": "bfloat16", "batch": 1, "grad": False,
                     "compile": "as-shipped", "tokens": tokens, "txt_tokens": ctx.txt,
                     "refs": ctx.refs, "warmup": ctx.warmup, "iters": ctx.iters,
                     "repeats": ctx.repeats,
                     "order": "token count ascending, fixtures in registry order"},
        "cases": results,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(snapshot, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {out}  ({len(results) - failed}/{len(results)} cases ok, "
          f"{time.perf_counter() - started:.0f} s)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
