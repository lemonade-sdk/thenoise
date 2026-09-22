#!/usr/bin/env python
"""Say what actually moved between two ``block_bench`` snapshots.

``block_bench.py`` photographs; this develops a pair of photographs and hands back the
rows that changed. Nothing is measured here, so it runs anywhere, in seconds, on two
JSON files.

The design is all about noise. Every case carries its own scatter (``groups``, or
``cov_pct`` in a schema-1 snapshot), so a row only counts as slower or faster when the
delta clears ``--min`` sigmas of the two runs' own spread. Everything below that is
counted, not printed: the useful answer from comparing two equal runs is the sentence
"nothing moved", not 75 rows of decimals that each say ±0.3%.

The headers are compared first. Two snapshots taken with a different ladder, torch
build, dtype or ``--refs`` are not the same experiment, and the deltas between them are
decoration — so they are still printed, under a warning that says as much.

Usage
-----
    .venv/bin/python bench-scripts/diff_bench.py before.json after.json
    .venv/bin/python bench-scripts/diff_bench.py a.json b.json --all      # every row
    .venv/bin/python bench-scripts/diff_bench.py a.json b.json --min 2    # looser test
    .venv/bin/python bench-scripts/diff_bench.py a.json b.json --gib 5    # memory flag

Exit code is 1 when a case got slower or one that used to work now fails. A header
difference alone does not fail: those numbers often are still comparable.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

SIGMA_FLOOR = 0.005          # a groups set with zero scatter would flag a 1 ns change
MIN_COV_FLOOR = 0.1          # ditto for a snapshot that recorded cov_pct == 0.0


def load(path: str) -> dict:
    try:
        with open(path) as fh:
            snap = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path}: {exc}")
    if snap.get("tool") != "block_bench" or not isinstance(snap.get("cases"), list):
        raise SystemExit(f"{path}: not a block_bench snapshot")
    return snap


def flat(section: dict, name: str) -> dict:
    """One level deep, so ``env.vars`` differs per variable rather than per dict."""
    out = {}
    for key, value in section.items():
        if isinstance(value, dict):
            out.update({f"{name}.{key}.{k}": v for k, v in value.items()})
        else:
            out[f"{name}.{key}"] = value
    return out


def shown(value) -> str:
    text = ",".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    return text if len(text) <= 44 else text[:41] + "..."


def header_diff(a: dict, b: dict) -> list[tuple[str, str, str]]:
    """Everything the two runs recorded about themselves that is not the same."""
    left, right = {}, {}
    for section in ("git", "env", "protocol"):
        left.update(flat(a.get(section) or {}, section))
        right.update(flat(b.get(section) or {}, section))
    if a.get("schema") != b.get("schema"):
        left["schema"], right["schema"] = a.get("schema"), b.get("schema")
    out = []
    for key in sorted(set(left) | set(right)):
        va, vb = left.get(key, "-"), right.get(key, "-")
        if va != vb:
            out.append((key, shown(va), shown(vb)))
    return out


def floor_pct(a: dict, b: dict, k: float) -> float:
    """Percent change this pair of cases could not have told apart from its own noise."""
    ga, gb = a.get("groups"), b.get("groups")
    if ga and gb and len(ga) > 1 and len(gb) > 1:
        ma, mb = statistics.fmean(ga), statistics.fmean(gb)
        sa = max(statistics.pstdev(ga), SIGMA_FLOOR * ma)
        sb = max(statistics.pstdev(gb), SIGMA_FLOOR * mb)
        return 100 * k * math.sqrt(sa * sa / len(ga) + sb * sb / len(gb)) / ma
    return max(k * max(a.get("cov_pct", 0.0), b.get("cov_pct", 0.0)), MIN_COV_FLOOR)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diff two block_bench snapshots.")
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--min", type=float, default=3.0, dest="sigma", metavar="K",
                    help="sigmas of the two runs' spread a delta must clear "
                         "(default %(default)s)")
    ap.add_argument("--gib", type=float, default=10.0, metavar="PCT",
                    help="report peak-memory changes over this %% (default %(default)s)")
    ap.add_argument("--all", action="store_true",
                    help="print every compared case, not only the movers")
    args = ap.parse_args()

    a, b = load(args.before), load(args.after)
    diffs = header_diff(a, b)
    if diffs:
        print("these snapshots were taken differently:")
        for key, va, vb in diffs:
            print(f"  {key}: {va}  ->  {vb}")
        print("  ...so read the numbers below with that in mind")

    A = {c["key"]: c for c in a["cases"]}
    B = {c["key"]: c for c in b["cases"]}
    common = sorted(set(A) & set(B))
    moved, quiet, memory, broke, fixed = [], [], [], [], []
    for key in common:
        ra, rb = A[key], B[key]
        if ra["status"] != rb["status"]:
            (broke if rb["status"] != "ok" else fixed).append((key, rb.get("error", "")))
            continue
        if ra["status"] != "ok" or not ra.get("ms"):
            continue
        delta = 100 * (rb["ms"] - ra["ms"]) / ra["ms"]
        noise = floor_pct(ra, rb, args.sigma)
        dgib = (100 * (rb["peak_gib"] - ra["peak_gib"]) / ra["peak_gib"]
                if ra.get("peak_gib") and rb.get("peak_gib") else 0.0)
        row = (delta, noise, key, ra, rb)
        (moved if abs(delta) > noise else quiet).append(row)
        if abs(dgib) > args.gib:
            memory.append((dgib, key, ra, rb))

    def line(delta: float, noise: float, key: str, ra: dict, rb: dict, mark: str = "") -> None:
        print(f"  {key:44s} {ra['ms']:9.2f} {rb['ms']:9.2f} {delta:+8.1f} {noise:5.1f} "
              f"{ra['tflops']:6.1f} -> {rb['tflops']:<6.1f} "
              f"{ra['peak_gib']:5.2f} -> {rb['peak_gib']:<5.2f} {mark}".rstrip())

    def header() -> None:
        print(f"  {'case':44s} {'before':>9s} {'after':>9s} {'Δ%':>8s} {'floor':>5s} "
              f"{'TFLOPS':>16s} {'GiB':>14s}")

    if args.all:
        print(f"\nevery compared case ({len(moved) + len(quiet)}), [*] clears "
              f"{args.sigma:g} sigma")
        header()
        for delta, noise, key, ra, rb in sorted(moved + quiet, key=lambda r: -r[0]):
            line(delta, noise, key, ra, rb, "*" if abs(delta) > noise else "")
    else:
        for title, rows in (("slower", sorted([r for r in moved if r[0] > 0],
                                              key=lambda r: -r[0])),
                            ("faster", sorted([r for r in moved if r[0] < 0],
                                              key=lambda r: r[0]))):
            if not rows:
                continue
            print(f"\n{title} ({len(rows)})")
            header()
            for delta, noise, key, ra, rb in rows:
                line(delta, noise, key, ra, rb)

    if memory:
        print(f"\npeak memory moved more than {args.gib:g}%")
        for dgib, key, ra, rb in sorted(memory, key=lambda r: -abs(r[0])):
            print(f"  {key:44s} {ra['peak_gib']:6.2f} -> {rb['peak_gib']:6.2f} GiB "
                  f"{dgib:+6.1f}%")

    for title, rows in (("now failing", broke), ("no longer failing", fixed)):
        if rows:
            print(f"\n{title} ({len(rows)})")
            for key, err in rows:
                print(f"  {key:44s} {err[:80]}")
    for title, keys in (("only in " + args.before, sorted(set(A) - set(B))),
                        ("only in " + args.after, sorted(set(B) - set(A)))):
        if keys:
            print(f"\n{title} ({len(keys)})")
            for key in keys:
                print(f"  {key}")

    slow = sum(1 for r in moved if r[0] > 0)
    summary = (f"\n{len(common)} cases compared: {slow} slower, "
               f"{len(moved) - slow} faster, {len(quiet)} under the noise floor "
               f"({args.sigma:g} sigma)")
    if memory:
        summary += f", {len(memory)} memory"
    if broke:
        summary += f", {len(broke)} broken"
    print(summary)
    sys.exit(1 if slow or broke else 0)


if __name__ == "__main__":
    main()
