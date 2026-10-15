#!/usr/bin/env python
"""Photograph the runtime cost of every VAE codec in the repo.

This is a *camera*, not a lab: it measures what the codecs in the working tree do right
now, prints a table and writes a JSON snapshot. There are deliberately no A/B switches —
nothing here patches a codec, swaps a conv backend or toggles compilation. Snapshot
before a change, snapshot after it, and a regression is a row: ``diff_bench.py`` is that
diff, and it is the only way to read a snapshot.

What gets photographed
----------------------
One case per ``(codec, op, resolution)``:

* **codec** — every VAE the engine ships, at its production width, with random (seeded)
  weights. No checkpoints, no server: what is under test is the module tree and the
  shapes that go through it, both read off the classes the loaders build (each fixture
  says where its knobs come from, and the two families are built from the loader's own
  architecture constants so this cannot drift from them).
* **op** — ``encode`` (``encode_pixels_to_latents``, the editing input path), ``decode``
  (``decode_to_pixels``, the last step of every generation) and ``features``
  (``decode_features``, the early-exit decode the Qwen-Image 2.1 transcoder conditions
  on). All three are called exactly as the models call them, latent normalisation
  included, so a row is what a request pays rather than what a conv stack costs.
  ``flux`` is decode-only because the engine ships no encoder for the Flux AE;
  ``features`` exists only on the Wan 2.2 family, which is the only decoder that can
  stop early.
* **resolution** — the pixel ladder from ``--res``: 1024x1024, 1536x2048 and 2048x2048
  by default, i.e. the three rungs a real request lands on. Both sides snap up to the
  codec's own compression, so every rung is a shape a model could actually ask for.

Out on purpose: SesquiLSR and the pixel upscalers. Their cost is a network *around* a
codec, and one row that is really two models makes a regression impossible to attribute.

Measurement protocol
--------------------
Fixed, and recorded in the JSON: numbers are only comparable between runs of the same
protocol *and* environment, which the JSON records next to them.

* batch 1, ``bfloat16`` (the dtype every model ends up running its codec in: the loaders
  either cast to the model dtype or ship bf16 files — a fp32 checkpoint nobody cast is a
  file choice, and ``--dtype fp32`` is there to price it), ``torch.no_grad()``, the
  codec's own eager forward — nothing here is compiled, and no codec ships a
  ``@torch.compile``.
* weights are seeded from the codec's name (so a codec gets identical weights at every
  rung and op), activations from the case key. Pixels are drawn in fp32 then cast, and
  latents are a unit-normal draw, which is what a denoised canonical latent looks like.
* one untimed first call — which also counts the case's FLOPs with PyTorch's flop
  counter, so ``TFLOPS`` is measured geometry (convs and matmuls, the way a peak is
  quoted) rather than an assumed one — then ``--warmup`` calls, then ``--repeats``
  groups of ``--iters`` timed calls. ``ms`` is the
  median of the groups, ``±%`` their coefficient of variation; read it before believing
  a small delta. A VAE pass is long enough that ``--iters 1`` costs nothing in launch
  noise, and the scatter across groups is what exposes drift. The raw groups go into the
  JSON, so a comparison can judge a delta against the two runs' own spread.
* ``GiB`` is the allocator's peak *allocated* for that case alone (stats reset after each
  case's warmup) and ``resv`` the peak it had *reserved*: at 2048x2048 a decoder's
  activations are gigabytes wide and the gap between those two columns is fragmentation,
  which is what actually bites on unified memory. ``Mpx/s`` is the throughput a user
  feels; ``TFLOPS`` is the one to compare across rungs, since it is the only column that
  is scale-free.
* ``1st`` is the untimed first call: MIOpen picks a conv algorithm per shape there, so
  it is the cold-shape tax a server pays and never a measured number.
* a case that raises — a 16 GB card will lose the top rung of the ladder — is reported
  and skipped, so one broken codec still leaves a snapshot of everything else (the exit
  code is 1).
* at the end, a *watchlist*: cases over ``±1%``, and codecs that end the ladder costing
  more per FLOP than they started it. Codec rows that get *better* along the ladder are
  only counted, never listed — that is normal, since a bigger conv is a fatter GEMM —
  so what is left is decay: the score-matrix or strip tiling in ``utils.attention`` and
  ``vae.wan22`` switching on, or the allocator starting to thrash.

A cold ``MIOPEN_USER_DB_PATH`` costs find time in ``1st``, never measured time.

Usage
-----
    .venv/bin/python bench-scripts/vae_bench.py --list
    .venv/bin/python bench-scripts/vae_bench.py                      # everything
    .venv/bin/python bench-scripts/vae_bench.py --models flux2,qwen_image21
    .venv/bin/python bench-scripts/vae_bench.py --ops decode --res 2048x2048
    .venv/bin/python bench-scripts/vae_bench.py --out bench-scripts/snapshots/head.json
    .venv/bin/python bench-scripts/diff_bench.py before.json after.json   # what moved
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Callable

import torch

SCHEMA = 1
DEFAULT_RES = "1024x1024,1536x2048,2048x2048"
ENCODE, DECODE, FEATURES = "encode", "decode", "features"
BOTH = (ENCODE, DECODE)
FEATURED = (ENCODE, DECODE, FEATURES)      # decoders that can stop early
OPS = (ENCODE, DECODE, FEATURES)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Environment that silently changes which kernels get measured. Convs are what a VAE is,
# so the MIOpen find mode and the allocator are as much part of the protocol here as the
# attention backend is in block_bench. Recorded so two snapshots that disagree can still
# be told apart from a real change.
BENCH_VARS = ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "MIOPEN_FIND_MODE",
              "MIOPEN_FIND_MAX_FINDS_PER_SEARCH", "TORCH_BLAS_PREFER_HIPBLASLT",
              "HSA_OVERRIDE_GFX_VERSION", "TORCH_COMPILE_DISABLE",
              "TORCH_CUDNN_V8_API_DISABLED", "PYTORCH_CUDA_ALLOC_CONF")


# ------------------------------------------------------------------------ cases
@dataclass
class Case:
    """One codec's production op at one shape, plus what it is worth per second."""

    run: Callable[[], object]
    mpx: float             # pixels this op moves, for the Mpx/s column
    latent: str = ""       # the canonical latent it produces or consumes
    detail: str = ""       # free-form: who loads this codec, what the op stops at


@dataclass(frozen=True)
class Fixture:
    model: str
    ops: tuple[str, ...]
    build: Callable[["Ctx", "Res", str], Case]


FIXTURES: list[Fixture] = []


def register(model: str, ops: tuple[str, ...], build) -> None:
    FIXTURES.append(Fixture(model, ops, build))


def fixture(model: str, ops: tuple[str, ...]):
    def deco(fn):
        register(model, ops, fn)
        return fn
    return deco


@dataclass
class Ctx:
    device: str
    warmup: int = 1
    iters: int = 1
    repeats: int = 3
    dtype: torch.dtype = torch.bfloat16

    @property
    def tdev(self) -> torch.device:
        return torch.device(self.device)

    @property
    def cuda(self) -> bool:
        return self.device == "cuda" and torch.cuda.is_available()


@dataclass(frozen=True)
class Res:
    """A pixel resolution the codec is asked to move, width x height."""

    w: int
    h: int

    @staticmethod
    def parse(text: str) -> "Res":
        """``1536x2048`` (or a bare ``1024`` for a square)."""
        parts = text.lower().replace(" ", "").split("x")
        if len(parts) == 1:
            parts = parts * 2
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise SystemExit(f"--res wants WxH (or N), got {text!r}")
        w, h = int(parts[0]), int(parts[1])
        if w < 16 or h < 16:
            raise SystemExit(f"--res {text!r} is smaller than any codec can hold")
        return Res(w, h)

    def align(self, comp: int) -> "Res":
        """Both sides up to the codec's compression: below it there are no latents."""
        from thenoise.utils.math import round_up

        w, h = round_up(self.w, comp), round_up(self.h, comp)
        if (w, h) != (self.w, self.h):
            print(f"  ({self} snapped to {w}x{h} for a {comp}x codec)", file=sys.stderr)
        return Res(w, h)

    @property
    def mpx(self) -> float:
        return self.w * self.h / 1e6

    def __str__(self) -> str:
        return f"{self.w}x{self.h}"


# ---------------------------------------------------------------------- helpers
def seed_of(key: str) -> int:
    return zlib.crc32(key.encode()) & 0x7FFF_FFFF


def new_vae(ctx: Ctx, name: str, cls, *args, **kwargs):
    """Seeded random weights, on device, compute dtype, eval — as a loaded codec is."""
    torch.manual_seed(seed_of(name))
    with torch.device(ctx.device):
        vae = cls(*args, **kwargs).to(ctx.dtype).eval()
    vae.requires_grad_(False)
    return vae


def seed_inputs(name: str, op: str, res: Res) -> None:
    torch.manual_seed(seed_of(f"{name}/{op}@{res}"))


def randn(ctx: Ctx, *shape) -> torch.Tensor:
    """Seeded values drawn in fp32 then cast, so they are dtype-independent."""
    return torch.randn(*shape, device=ctx.tdev, dtype=torch.float32).to(ctx.dtype)


def rand_pixels(ctx: Ctx, res: Res, channels: int) -> torch.Tensor:
    """Uniform [-1, 1] noise: what a decoded image is bounded by, both ways."""
    x = torch.rand(1, channels, res.h, res.w, device=ctx.tdev, dtype=torch.float32)
    return (x * 2.0 - 1.0).to(ctx.dtype)


def vae_case(ctx: Ctx, vae, name: str, res: Res, op: str, note: str = "") -> Case:
    """Wire one op of ``vae`` up at ``res``, exactly as a model calls it.

    Every codec states its own geometry (``z_dim``, ``spatial_compression``,
    ``pixel_channels``), so the latent and pixel tensors here are built the way the
    pipeline builds them rather than being hardcoded per codec.
    """
    comp = vae.spatial_compression
    res = res.align(comp)
    lh, lw = res.h // comp, res.w // comp
    seed_inputs(name, op, res)
    latent = f"{vae.z_dim}x{lh}x{lw}"

    if op == ENCODE:
        x = rand_pixels(ctx, res, vae.pixel_channels)

        def run():
            return vae.encode_pixels_to_latents(x)
    elif op in (DECODE, FEATURES):
        x = randn(ctx, 1, vae.z_dim, lh, lw)
        # ``decode_features`` is a decode that stops after its first upsample stage, so
        # it is fed the same canonical latent the full decode is.
        step = vae.decode_to_pixels if op == DECODE else vae.decode_features

        def run():
            return step(x)
    else:
        raise ValueError(f"unknown op {op!r}")

    detail = f"{vae.pixel_channels}ch px {res}; {note}"
    if op == FEATURES:
        detail += f", stops after upsample 1/{len(vae.decoder.upsamples)}"
    return Case(run, res.mpx, latent, detail)


# --------------------------------------------------------------------- fixtures
# The knobs below are the ones the loaders pass, not measurements of the checkpoints:
# ``_WAN21_FAMILY_ARCH`` is that loader's own constant, and the Wan 2.2 family numbers
# are what ``_infer_arch`` recovers from the shipped files.
@fixture("qwen_image", BOTH)
def qwen_image_vae(ctx: Ctx, res: Res, op: str) -> Case:
    """The Qwen-Image codec (Wan 2.1 family): 16ch latent at 8x, RGB."""
    from thenoise.vae import AutoencoderKLQwenImage
    from thenoise.vae.qwen_image import _WAN21_FAMILY_ARCH

    vae = new_vae(ctx, "qwen_image", AutoencoderKLQwenImage, **_WAN21_FAMILY_ARCH)
    return vae_case(ctx, vae, "qwen_image", res, op, "16ch/8x rgb; anima+krea2 too")


@fixture("ming_image", BOTH)
def ming_image_vae(ctx: Ctx, res: Res, op: str) -> Case:
    """The Ming-Image codec: the same tree, RGBA pixels and a scalar-scaled latent."""
    from thenoise.vae import AutoencoderKLQwenImage
    from thenoise.vae.qwen_image import (
        MING_IMAGE_SCALE_FACTOR,
        MING_IMAGE_SHIFT_FACTOR,
        _WAN21_FAMILY_ARCH,
    )

    vae = new_vae(ctx, "ming_image", AutoencoderKLQwenImage,
                  input_channels=4, scale_factor=MING_IMAGE_SCALE_FACTOR,
                  shift_factor=MING_IMAGE_SHIFT_FACTOR, **_WAN21_FAMILY_ARCH)
    return vae_case(ctx, vae, "ming_image", res, op, "16ch/8x rgba, scalar scale")


# Qwen-Image 2.1: the 64ch/16x RGBA member of the Wan 2.2 family, five stages so the
# stages alone make 16x and patchify is 1 (``qwen_image_2.1_vae``, via _infer_arch).
QI21_ARCH = dict(dim=96, dec_dim=144, z_dim=64, dim_mult=[1, 2, 4, 8, 8],
                 num_res_blocks=2, temperal_downsample=[False, True, True, True],
                 image_channels=4, patch_size=1)


@fixture("qwen_image21", FEATURED)
def qwen_image21_vae(ctx: Ctx, res: Res, op: str) -> Case:
    """The Qwen-Image 2.1 codec: 64ch latent at 16x, RGBA, row-strip convs."""
    from thenoise.vae import AutoencoderKLWan22

    vae = new_vae(ctx, "qwen_image21", AutoencoderKLWan22, **QI21_ARCH)
    return vae_case(ctx, vae, "qwen_image21", res, op, "64ch/16x rgba")


@fixture("wan22", FEATURED)
def wan22_vae(ctx: Ctx, res: Res, op: str) -> Case:
    """The stock Wan 2.2 codec (48ch at 16x, RGB) the same module serves.

    No shipped adapter loads a 48ch file, so this row is the family's other shape —
    the one ``AutoencoderKLWan22``'s own defaults describe, and the baseline the
    Qwen-Image 2.1 variant above is a deviation from.
    """
    from thenoise.vae import AutoencoderKLWan22

    vae = new_vae(ctx, "wan22", AutoencoderKLWan22)
    return vae_case(ctx, vae, "wan22", res, op, "48ch/16x rgb, patchify 2")


@fixture("flux", (DECODE,))
def flux_vae(ctx: Ctx, res: Res, op: str) -> Case:
    """The Flux AE, decode only: 16ch at 8x, and no encoder in the engine at all."""
    from thenoise.vae import AutoencoderKLFlux

    vae = new_vae(ctx, "flux", AutoencoderKLFlux)
    return vae_case(ctx, vae, "flux", res, op, "16ch/8x rgb, decoder only")


@fixture("flux2", BOTH)
def flux2_vae(ctx: Ctx, res: Res, op: str) -> Case:
    """The Flux.2 codec: 32ch at 8x packed 2x2 into a 128ch/16x BatchNorm'd latent."""
    from thenoise.vae import AutoencoderKLFlux2

    vae = new_vae(ctx, "flux2", AutoencoderKLFlux2)
    return vae_case(ctx, vae, "flux2", res, op, "128ch/16x packed, bn normalised")


@fixture("mage_flow", BOTH)
def mage_flow_vae(ctx: Ctx, res: Res, op: str) -> Case:
    """Mage-VAE: a one-step flow codec, so both directions are a small DiT."""
    from thenoise.vae import AutoencoderKLMageFlow

    vae = new_vae(ctx, "mage_flow", AutoencoderKLMageFlow)
    return vae_case(ctx, vae, "mage_flow", res, op, "128ch/16x, one-step denoiser")


# ------------------------------------------------------------------ measurement
@dataclass
class Timing:
    """What one case cost. ``groups`` is the raw evidence behind ``ms`` and ``cov_pct``."""

    ms: float
    cov_pct: float
    groups: list[float]
    flops: float
    peak_gib: float
    peak_res_gib: float
    first_s: float


def measure(run, ctx: Ctx) -> Timing:
    """Count the FLOPs on an untimed call, warm up, then ``repeats`` x ``iters`` timed.

    The counting pass is free in wall-clock terms (it *is* the call every case makes
    anyway, wrapped in a flop counter) and it is the only honest source of FLOPs for a
    convnet: a decoder's cost per pixel depends on which stage runs at which width, and
    on attention switching to its tiled path. If the counter itself is what fails the
    case is still measured, with ``TFLOPS`` left blank.
    """
    sync = torch.cuda.synchronize if ctx.cuda else (lambda: None)
    with torch.no_grad():
        t0 = time.perf_counter()
        flops = 0.0
        try:
            from torch.utils.flop_counter import FlopCounterMode

            with FlopCounterMode(display=False) as counter:
                run()
            flops = float(counter.get_total_flops())
        except Exception as exc:                                # noqa: BLE001
            print(f"\n    (no FLOP count: {type(exc).__name__}: {str(exc)[:80]})",
                  file=sys.stderr)
            run()                                               # re-raises a real failure
        sync()
        first_s = time.perf_counter() - t0
        for _ in range(ctx.warmup):
            run()
        sync()
        if ctx.cuda:
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
    peak = torch.cuda.max_memory_allocated() / 2**30 if ctx.cuda else 0.0
    peak_res = torch.cuda.max_memory_reserved() / 2**30 if ctx.cuda else 0.0
    return Timing(groups[len(groups) // 2], 100 * math.sqrt(var) / mean,
                  [round(g, 4) for g in groups], flops, peak, peak_res, first_s)


def watchlist(results: list[dict], cov_warn: float = 1.0, decay_pct: float = 25.0,
              show: int = 5) -> None:
    """Rows worth a second look inside *this* snapshot, printed as they are found.

    Two cheap rules, no judgement: a case that was not stable (``±%`` over
    ``cov_warn``), and a codec op that ends the ladder costing more per FLOP than it
    started it. Rows that get *better* along the ladder are only counted, never listed —
    a bigger conv is a fatter GEMM, so that is the normal shape of the column — so what
    is left is decay, which for a VAE has three usual suspects: the score-matrix tiling
    in ``utils.attention`` engaging on a mid-block attention, the row-strip convs in
    ``vae.wan22`` crossing their element threshold, and the allocator.
    """
    ok = [r for r in results if r["status"] == "ok" and r.get("tflops")]
    noisy = sorted((r for r in results if r["status"] == "ok" and r["cov_pct"] > cov_warn),
                   key=lambda r: -r["cov_pct"])
    series: dict[tuple[str, str], list[dict]] = {}
    for r in ok:
        series.setdefault((r["model"], r["op"]), []).append(r)
    decaying, gaining = [], 0
    for key, rows in series.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: r["pixels"])
        lost = 100 * (1 - rows[-1]["tflops"] / max(r["tflops"] for r in rows))
        if lost > decay_pct:
            decaying.append((lost, key, rows))
        elif 100 * (rows[-1]["tflops"] / rows[0]["tflops"] - 1) > decay_pct:
            gaining += 1
    if not decaying and not noisy:
        print(f"\nwatchlist: clean (no codec op losing {decay_pct:.0f}% of its TFLOPS "
              f"along the ladder, every case under ±{cov_warn}%)")
        return
    print("\nwatchlist")
    for lost, key, rows in sorted(decaying, key=lambda d: -d[0]):
        track = "  ".join(f"{r['resolution']}:{r['tflops']:.1f}" for r in rows)
        print(f"  {'/'.join(key):26s} loses {lost:3.0f}% of its TFLOPS  {track}")
    if gaining:
        print(f"  {'':26s} ({gaining} codec op(s) gain {decay_pct:.0f}% along the ladder "
              f"instead — normal, a bigger conv is a fatter GEMM)")
    for r in noisy[:show]:
        print(f"  {'/'.join((r['model'], r['op'])):26s} "
              f"unstable: ±{r['cov_pct']:.1f}% at {r['resolution']}")
    if len(noisy) > show:
        print(f"  {'':26s} (+{len(noisy) - show} more over ±{cov_warn}%)")


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
    """Fixtures whose codec name starts with one of the comma-separated filters.

    Prefix in one direction only on purpose: ``qwen`` covers ``qwen_image`` and
    ``qwen_image21``, while ``qwen_image21`` means that codec alone.
    """
    wanted = [w.strip() for w in models.split(",") if w.strip()]
    keep = [f for f in FIXTURES if any(f.model.startswith(w) for w in wanted)]
    if not keep:
        raise SystemExit(f"no codec matches {models}\nknown: "
                         f"{', '.join(f.model for f in FIXTURES)}")
    return keep


def case_list(fixtures: list[Fixture], ladder: list[Res], ops: tuple[str, ...]):
    """Canonical order: every codec at one rung, then the next rung, encode first.

    Rungs go up by pixel count, so the ladder reads as one machine getting hotter
    rather than as three unrelated tables.
    """
    return [(f, res, op) for res in ladder for f in fixtures for op in f.ops
            if op in ops]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Snapshot the cost of every VAE codec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="codecs: " + ", ".join(f.model for f in FIXTURES))
    ap.add_argument("--models", help="comma-separated codec filter, e.g. flux2,qwen_image21")
    ap.add_argument("--res", default=DEFAULT_RES,
                    help="pixel ladder as WxH (bare N = square), default %(default)s")
    ap.add_argument("--ops", default=",".join(OPS),
                    help="comma-separated subset of encode,decode,features")
    ap.add_argument("--dtype", default="bf16", choices=sorted(DTYPES),
                    help="compute dtype, as every model loads it (default %(default)s)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=1, help="timed calls per group")
    ap.add_argument("--repeats", type=int, default=3, help="groups per case (median reported)")
    ap.add_argument("--warmup", type=int, default=1, help="calls after the untimed one")
    ap.add_argument("--out", help="JSON snapshot path (default bench-scripts/snapshots/"
                                  "vae-<device>-<timestamp>.json)")
    ap.add_argument("--list", action="store_true", help="print the case matrix and exit")
    args = ap.parse_args()

    ladder = sorted((Res.parse(r) for r in args.res.split(",") if r.strip()),
                    key=lambda r: (r.mpx, r.w, r.h))
    ops = tuple(o.strip() for o in args.ops.split(",") if o.strip())
    if not ladder or any(o not in OPS for o in ops) or args.repeats < 1 or args.iters < 1:
        raise SystemExit(f"--res must not be empty and --ops must be from {','.join(OPS)}")
    fixtures = select(args.models) if args.models else list(FIXTURES)
    cases = case_list(fixtures, ladder, ops)
    if args.list:
        print("\n".join(f"{f.model}/{op}@{res}" for f, res, op in cases))
        print(f"\n{len(cases)} cases: {len(fixtures)} codecs x {len(ladder)} rungs "
              f"x ops {','.join(ops)}")
        return
    if not cases:
        raise SystemExit("nothing selected: no listed codec runs the ops you asked for")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no cuda/rocm device visible — run this on the GPU box")
    ctx = Ctx(device=args.device, warmup=args.warmup, iters=args.iters,
              repeats=args.repeats, dtype=DTYPES[args.dtype])

    env = env_info(args.device)
    print(f"{env.get('gpu', args.device)} | torch {torch.__version__} | hip {env.get('hip')}")
    print(f"{args.dtype} | batch 1 | ladder {', '.join(map(str, ladder))} | ops "
          f"{','.join(ops)} | warmup {ctx.warmup} + {ctx.repeats}x{ctx.iters} timed "
          f"| {len(cases)} cases")

    results, failed, last_res, case = [], 0, None, None
    started = time.perf_counter()
    for f, res, op in cases:
        if res != last_res:
            print(f"\n=== {res} ({res.mpx:.1f} Mpx) ===")
            print(f"  {'codec/op':26s} {'latent':>13s} {'ms':>9s} {'±%':>5s} {'Mpx/s':>7s} "
                  f"{'TFLOPS':>7s} {'GiB':>6s} {'resv':>6s} {'1st':>6s}")
            last_res = res
        label = f"{f.model}/{op}"
        entry = {"key": f"{label}@{res}", "model": f.model, "op": op,
                 "resolution": str(res), "width": res.w, "height": res.h,
                 "pixels": res.w * res.h}
        # The first call of a case is untimed but slow (MIOpen searches a conv
        # algorithm per shape): print the row label first so a long search is visibly
        # progress rather than a hang.
        print(f"  {label:26s}", end="", flush=True)
        try:
            case = f.build(ctx, res, op)
            tim = measure(case.run, ctx)
            tflops = tim.flops / (tim.ms / 1000) / 1e12 if tim.flops else 0.0
            mpx = case.mpx / (tim.ms / 1000)
            tf = f"{tflops:7.1f}" if tim.flops else "      -"
            print(f" {case.latent:>13s} {tim.ms:9.1f} {tim.cov_pct:5.1f} {mpx:7.2f} {tf} "
                  f"{tim.peak_gib:6.2f} {tim.peak_res_gib:6.2f} {tim.first_s:6.1f}",
                  flush=True)
            entry.update({"status": "ok", "latent": case.latent,
                          "ms": round(tim.ms, 4), "cov_pct": round(tim.cov_pct, 3),
                          "groups": tim.groups, "first_s": round(tim.first_s, 2),
                          "tflop": round(tim.flops / 1e12, 4) if tim.flops else None,
                          "tflops": round(tflops, 2) if tim.flops else None,
                          "mpx_per_s": round(mpx, 4),
                          "peak_gib": round(tim.peak_gib, 3),
                          "peak_reserved_gib": round(tim.peak_res_gib, 3),
                          "detail": case.detail})
        except Exception as exc:                                # noqa: BLE001
            print(f" {'FAILED':>13s}  {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            failed += 1
        entry["t_s"] = round(time.perf_counter() - started, 1)   # drift is not a shape
        results.append(entry)
        case = None
        gc.collect()
        if ctx.cuda:
            torch.cuda.empty_cache()

    watchlist(results)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots",
                                   f"vae-{device_slug(env)}-{time.strftime('%Y%m%d-%H%M')}.json")
    snapshot = {
        "tool": "vae_bench",
        "schema": SCHEMA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "git": git_info(),
        "env": env,
        "protocol": {"dtype": args.dtype, "batch": 1, "grad": False,
                     "compile": "none (no codec ships one)",
                     "resolutions": [str(r) for r in ladder], "ops": list(ops),
                     "warmup": ctx.warmup, "iters": ctx.iters, "repeats": ctx.repeats,
                     "flops": "torch FlopCounterMode, counted on the untimed first call",
                     "order": "pixel count ascending, codecs in registry order, "
                              "encode before decode"},
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
