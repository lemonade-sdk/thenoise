#!/usr/bin/env python
"""One picture, the shape knobs, and what each setting costs and produces.

    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 MIOPEN_FIND_MODE=FAST \
      .venv/bin/python bench-scripts/shape_sweep.py \
      --dit models/hunyuan3d/hunyuan_3d_v2.1.safetensors \
      --image chair.png --out /tmp/shape-sweep.json

The model loads once and every cell renders against it, as a server would. Each cell reports
wall time, the sampling window inside it, peak allocator memory, and what the cut produced:
triangles, signed volume, flat faces, boundary edges. Snapshots have the shape
``block_bench.py`` writes, so ``diff_bench.py`` reads a pair of sweeps — one grid against
another grid, or eager against compiled.

Those two exports are part of the experiment: they are what the app runs with, and
``env.vars`` records them. Cells are the model's own defaults plus one change per knob;
``--factorial`` crosses them, and ``auto`` in a list means the shipped value of that knob.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from block_bench import device_slug, env_info, git_info  # noqa: E402

SCHEMA = 1
TOOL = "shape_sweep"
KNOBS = ("density", "steps", "guidance", "level")
AXES = {"density": "auto,768", "steps": "auto,20", "guidance": "auto",
        "level": "auto,1/1024"}
FIELD = {"density": "octree_resolution", "steps": "steps",
         "guidance": "guidance_scale", "level": "mc_level"}
WARM = {"octree_resolution": 64, "steps": 2, "guidance_scale": 1.0}


def parse_list(knob: str, text: str) -> list:
    """``auto,768`` or ``auto,1/1024``; ``auto`` is the model's own value."""
    values = [None if one.strip() in ("", "auto")
              else float(Fraction(one.strip())) for one in text.split(",") if one.strip()]
    return [None if value is None else int(value) for value in values] \
        if knob in ("density", "steps") else values


@dataclass(frozen=True)
class Cell:
    """One render. ``None`` asks the model for its own value of that knob."""

    density: int | None = None
    steps: int | None = None
    guidance: float | None = None
    level: float | None = None

    def spec_overrides(self) -> dict:
        return {FIELD[knob]: value for knob in KNOBS
                if (value := getattr(self, knob)) is not None}

    def key(self, resolved=None) -> str:
        """Named by what was actually used, so a cell that asked for ``auto`` and one that
        asked for the value ``auto`` resolves to are the same experiment."""
        named = []
        for knob in KNOBS:
            value = None if resolved is None else getattr(resolved, FIELD[knob], None)
            if value is None:
                value = getattr(self, knob)
            named.append(written_as(knob, value))
        return "d%s/s%s/g%s/c%s" % tuple(named)


def written_as(knob: str, value) -> str:
    """A cut level reads as its reciprocal when it is one: ``1/512`` says more than
    ``0.001953125``, and these strings are what a comparison prints row by row."""
    if value is None:
        return "auto"
    if knob == "level" and value:
        over = 1.0 / value
        if abs(over - round(over)) < 1e-6:
            return f"1/{int(round(over))}"
    return f"{value:g}"


def grid(axis_values: dict[str, list], factorial: bool) -> list[Cell]:
    """The defaults first, then one knob at a time unless the knobs are crossed."""
    if not factorial:
        cells = [Cell()]
        for knob in KNOBS:
            for value in axis_values[knob]:
                if value is not None:
                    cells.append(Cell(**{knob: value}))
        return cells
    rows = [()]
    for knob in KNOBS:
        rows = [row + (value,) for row in rows for value in axis_values[knob]]
    return [Cell()] + [Cell(*row) for row in rows if any(value is not None for value in row)]


def glb_stats(path: Path) -> dict:
    """Positions and indices read back off the file the render wrote, so the numbers describe
    the artifact rather than what the pipeline believed it made."""
    raw = path.read_bytes()
    json_length = int.from_bytes(raw[12:16], "little")
    document = json.loads(raw[20:20 + json_length])
    bin_start = 20 + json_length + 8
    kinds = {5126: "<f4", 5125: "<u4", 5123: "<u2"}
    fields = {"VEC3": 3, "SCALAR": 1}
    primitive = document["meshes"][0]["primitives"][0]

    def column(number: int) -> np.ndarray:
        accessor = document["accessors"][number]
        view = document["bufferViews"][accessor["bufferView"]]
        offset = bin_start + view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        return np.frombuffer(raw, dtype=kinds[accessor["componentType"]],
                             count=accessor["count"] * fields[accessor["type"]], offset=offset)

    positions = column(primitive["attributes"]["POSITION"]).reshape(-1, 3).astype(np.float64)
    faces = column(primitive["indices"]).reshape(-1, 3)
    corners = positions[faces]
    span, reach = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(span, reach), axis=-1)
    volume = float(np.einsum("ij,ij->i", corners[:, 0],
                             np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys = edges.min(axis=1).astype(np.int64) * (positions.shape[0] + 1) \
        + edges.max(axis=1).astype(np.int64)
    multiplicity = np.bincount(np.unique(keys, return_inverse=True)[1])
    return {"vertices": int(positions.shape[0]), "faces": int(faces.shape[0]),
            "volume": volume, "flat_faces": int((area < 1e-12).sum()),
            "boundary_edges": int((multiplicity == 1).sum()),
            "nonmanifold_edges": int((multiplicity > 2).sum()),
            "bytes": len(raw), "finite": bool(np.isfinite(positions).all())}


class SamplingWindow:
    """Sampling is the only stage the mesh controller counts, so it is the one window cut out
    of the total; what is left is conditioning, the volume and the write."""

    def __init__(self) -> None:
        self.spans: list[float] = []
        self._first: float | None = None

    def __call__(self, stage: str, step: int, total: int) -> None:
        if stage != "sampling":
            return
        now = time.perf_counter()
        self._first = now if self._first is None else self._first
        if step >= total:
            self.spans.append(now - self._first)
            self._first = None

    @property
    def seconds(self) -> float:
        return statistics.median(self.spans) if self.spans else 0.0


def render(runtime, spec, path: Path) -> tuple[float, float]:
    """One render: seconds, and the sampling window inside them."""
    from thenoise.jobs import Uncancellable

    window = SamplingWindow()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runtime.run(lambda controller: controller.render(spec, Uncancellable(), window, path))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - started, window.seconds


def peak_gib() -> float:
    return 0.0 if not torch.cuda.is_available() else torch.cuda.max_memory_allocated() / 2**30


def run_cell(runtime, cell: Cell, spec_kwargs: dict, args, out_dir: Path) -> dict:
    from thenoise.mesh.spec import MeshSpec

    resolved = runtime.run(lambda controller: controller.resolve(
        MeshSpec(**spec_kwargs, **cell.spec_overrides())))
    path = out_dir / (cell.key(resolved).replace("/", "_") + ".glb")
    timings, sampling, error = [], [], None
    peak = 0.0
    for _ in range(args.reps):
        try:
            seconds, window = render(runtime, resolved, path)
        except Exception as caught:                             # noqa: BLE001
            error = f"{type(caught).__name__}: {caught}"
            break
        timings.append(seconds)
        sampling.append(window)
        peak = max(peak, peak_gib())
    record = {"key": cell.key(resolved), "dtype": "bf16", "peak_gib": peak,
              "sampling_s": statistics.median(sampling) if sampling else 0.0,
              "latents": resolved.latents, "bounds": resolved.bounds,
              "sampler": resolved.sampler}
    if error:
        record.update({"status": "error", "ms": 0.0, "groups": [], "cov_pct": 0.0,
                       "error": error})
        return record
    record.update({"status": "ok", "ms": statistics.median(timings) * 1000,
                   "groups": timings,
                   "cov_pct": 0.0 if len(timings) < 2 else
                   100 * statistics.pstdev(timings) / statistics.fmean(timings)})
    try:
        record.update(glb_stats(path))
    finally:
        if args.keep is None:
            path.unlink(missing_ok=True)
    return record


def warm(runtime, spec_kwargs: dict, out_dir: Path, keep: bool) -> float:
    """One short render before any cell, so no row pays Dynamo's tracing and the cheapest
    cell is not the one that ran second. Guidance 1.0 keeps it cheap; the transformer's graph
    is the same one a guided cell walks, only travelled twice there."""
    from thenoise.mesh.spec import MeshSpec

    path = out_dir / "warm.glb"
    seconds, _ = render(runtime, MeshSpec(**spec_kwargs, **WARM), path)
    if not keep:
        path.unlink(missing_ok=True)
    return seconds


def watch(cases: list[dict]) -> list[str]:
    notes = [f"{one['key']}: {one.get('error', 'failed')}" for one in cases
             if one["status"] != "ok"]
    notes += [f"{one['key']}: {one['boundary_edges']} boundary edges, the cut is not closed"
              for one in cases if one["status"] == "ok" and one["boundary_edges"]]
    baseline = next((one for one in cases if one["status"] == "ok"), None)
    if baseline and baseline["volume"]:
        for one in cases:
            if one["status"] != "ok" or one is baseline:
                continue
            change = 100 * (one["volume"] - baseline["volume"]) / baseline["volume"]
            if abs(change) > 10.0:
                notes.append(f"{one['key']}: volume {change:+.0f}% against {baseline['key']} "
                             "— a different shape, not a finer one")
    good = [one for one in cases if one["status"] == "ok" and one["ms"] and one["faces"]]
    if len(good) > 1:
        cheapest = min(good, key=lambda one: one["ms"] / one["faces"])
        dearest = max(good, key=lambda one: one["ms"] / one["faces"])
        notes.append(f"cheapest per triangle {cheapest['key']}, dearest {dearest['key']}")
    return notes


def table(cases: list[dict]) -> None:
    print(f"\n{'cell':30s} {'s':>7} {'samp':>7} {'GiB':>6} {'tris':>9} "
          f"{'volume':>9} {'flat':>6} {'bnd':>5}")
    for case in cases:
        if case["status"] != "ok":
            print(f"{case['key']:30s} {'-':>7} {'-':>7} {'-':>6} {'-':>9} {'-':>9} "
                  f"{'-':>6} {'-':>5}  {case.get('error', '')[:56]}")
            continue
        print(f"{case['key']:30s} {case['ms'] / 1000:7.1f} {case['sampling_s']:7.1f} "
              f"{case['peak_gib']:6.2f} {case['faces']:9d} {case['volume']:9.4f} "
              f"{case['flat_faces']:6d} {case['boundary_edges']:5d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dit", required=True, metavar="PATH")
    ap.add_argument("--image", action="append", required=True, metavar="PATH",
                    help="the picture every cell renders from; repeat for views")
    ap.add_argument("--densities", default=AXES["density"])
    ap.add_argument("--steps", default=AXES["steps"])
    ap.add_argument("--guidance", default=AXES["guidance"])
    ap.add_argument("--levels", default=AXES["level"], help="cut levels, e.g. auto,1/1024")
    ap.add_argument("--bounds", type=float, default=None, help="fixed for every cell")
    ap.add_argument("--sampler", default=None, help="fixed for every cell")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lora", action="append", default=[], metavar="FILE[:WEIGHT]")
    ap.add_argument("--lora-dir", default="", metavar="PATH")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=2, help="renders per cell (default 2)")
    ap.add_argument("--factorial", action="store_true", help="cross the knobs")
    ap.add_argument("--keep", metavar="DIR", help="keep each cell's .glb here")
    ap.add_argument("--out", default="", metavar="PATH", help="snapshot to write")
    ap.add_argument("--dry-run", action="store_true", help="print the cells and stop")
    args = ap.parse_args()
    if args.reps < 1:
        raise SystemExit("--reps must be at least 1")

    axis_values = {"density": parse_list("density", args.densities),
                   "steps": parse_list("steps", args.steps),
                   "guidance": parse_list("guidance", args.guidance),
                   "level": parse_list("level", args.levels)}
    cells = grid(axis_values, args.factorial)
    print(f"{len(cells)} cells x {args.reps} reps")
    for cell in cells:
        print(f"  {cell.key()}")
    if args.dry_run:
        return

    from thenoise.one_shot import load
    from thenoise.utils.media import read_media

    pictures = [read_media(path, "mesh") for path in args.image]
    spec_kwargs = {"image": pictures[0] if len(pictures) == 1 else pictures,
                   "seed": args.seed, "output_format": "glb",
                   "lora_specs": args.lora or None}
    if args.bounds is not None:
        spec_kwargs["bounds"] = args.bounds
    if args.sampler is not None:
        spec_kwargs["sampler"] = args.sampler

    runtime = load(args, "mesh", "shape-sweep")
    info = env_info(args.device)
    out_dir = Path(args.keep or f"{Path(args.out or 'shape-sweep').stem}.cells")
    out_dir.mkdir(parents=True, exist_ok=True)

    warm_s = warm(runtime, spec_kwargs, out_dir, args.keep is not None)
    print(f"warm render {warm_s:.1f}s\n")

    started = time.perf_counter()
    cases, measured = [], set()
    for cell in cells:
        record = run_cell(runtime, cell, spec_kwargs, args, out_dir)
        if record["key"] in measured:
            print(f"  {record['key']:30s} already measured: another cell asks for the same")
            continue
        measured.add(record["key"])
        cases.append(record)
        print(f"  {record['key']:30s} "
              + (f"{record['ms'] / 1000:.1f}s, {record.get('faces', 0)} tris, "
                 f"{record['peak_gib']:.2f} GiB" if record["status"] == "ok"
                 else record.get("error", "")[:80]))

    table(cases)
    notes = watch(cases)
    if notes:
        print("\nworth a look:")
        for note in notes:
            print(f"  {note}")

    if args.out:
        snapshot = {"tool": TOOL, "schema": SCHEMA, "git": git_info(), "env": info,
                    "protocol": {"device": args.device, "slug": device_slug(info),
                                 "seed": args.seed, "reps": args.reps,
                                 "factorial": args.factorial, "bounds": args.bounds,
                                 "sampler": args.sampler, "images": len(args.image),
                                 "warm_s": round(warm_s, 1),
                                 "knobs": {one: axis_values[one] for one in KNOBS}},
                    "cases": cases}
        Path(args.out).write_text(json.dumps(snapshot, indent=1))
        print(f"\nwrote {args.out} in {time.perf_counter() - started:.0f}s "
              f"on {device_slug(info)}")
    sys.exit(1 if any(one["status"] != "ok" for one in cases) else 0)


if __name__ == "__main__":
    main()
