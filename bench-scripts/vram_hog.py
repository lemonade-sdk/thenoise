#!/usr/bin/env python
"""Hold device memory so a big card behaves like a small one.

The engine plans every stage against what the driver says is free, so a card with room
for everything exercises none of the interesting code: nothing is admitted to CPU,
nothing is evicted, and the out-of-memory retry never runs. Sitting on a few GiB from a
second process makes the server see a smaller card, and sees it in the driver's own
figures, so admission, eviction order, the squeeze and a genuine OOM all behave as they
would on real hardware.

Why a second process rather than a switch in the engine: memory held here is invisible to
the server's caching allocator and unreclaimable by its ``squeeze()``, which is what a
smaller card is. And it stays out of the CLI, where it would be a puzzle to everyone who
did not come here to break things.

Run it before the server and leave it running::

    .venv/bin/python bench-scripts/vram_hog.py --gib 40
    .venv/bin/python -m thenoise serve ...

On the terminal it started in it takes commands: ``+4`` hold four GiB more, ``-4`` give
four back, ``=20`` hold twenty in total, anything else prints, ``q`` quits. Started with
no terminal it holds until ``SIGINT``; ``SIGUSR1`` prints the state.

Two things to know before aiming high on an APU. The pool is the GTT, which the
compositor also allocates from, so a desktop box can lose its display -- use an headless
one. And one big block is not a card full of interleaved allocations: this simulates a
shortage, not a fragmentation.
"""
from __future__ import annotations

import argparse
import signal
import sys

import torch

GIB = 1 << 30


def _free(device: torch.device) -> tuple[int, int]:
    """(free, total) bytes, as the driver reports them."""
    return torch.cuda.mem_get_info(device)


def _shared_pool_warning(device: torch.device) -> str:
    """A warning when the device's pool is system memory rather than VRAM."""
    try:
        import glob

        carve_outs = [int(open(p).read()) for p in glob.glob(
            "/sys/class/drm/card*/device/mem_info_vis_vram_total")]
    except OSError:
        return ""
    total = torch.cuda.get_device_properties(device).total_memory
    if not carve_outs or max(carve_outs) * 8 >= total:
        return ""
    return ("this device's memory is mostly shared with the system, so on a box with a "
            "desktop, holding this much can take the compositor with it")


class Balloon:
    """Device memory taken in blocks, so the hold can grow and shrink on request."""

    def __init__(self, device: torch.device, block_bytes: int):
        self.device = device
        self.block_bytes = block_bytes
        self.blocks: list[torch.Tensor] = []

    @property
    def held(self) -> int:
        return len(self.blocks) * self.block_bytes

    def grow(self, gib: float) -> None:
        self._take(self._blocks(gib))

    def shrink(self, gib: float) -> None:
        self._release(self._blocks(gib))

    def set(self, gib: float) -> None:
        target = self._blocks(gib)
        if target > len(self.blocks):
            self._take(target - len(self.blocks))
        else:
            self._release(len(self.blocks) - target)

    def report(self) -> None:
        free, total = _free(self.device)
        print(f"held {self.held / GIB:6.1f} GiB | {free / GIB:6.1f} GiB free of "
              f"{total / GIB:.1f} on {self.device}", flush=True)

    def _blocks(self, gib: float) -> int:
        return max(round(gib * GIB / self.block_bytes), 0)

    def _take(self, blocks: int) -> None:
        for _ in range(blocks):
            # Leave the driver a block beyond ours: that last stretch is what lets the
            # server fail cleanly instead of wedging.
            if _free(self.device)[0] < self.block_bytes * 2:
                print("the driver will give no more", flush=True)
                break
            self.blocks.append(
                torch.empty(self.block_bytes, dtype=torch.uint8, device=self.device))

    def _release(self, blocks: int) -> None:
        for _ in range(min(blocks, len(self.blocks))):
            self.blocks.pop()
        # Otherwise the bytes sit in this process's cache, invisible to the server,
        # which is the opposite of what was asked for.
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hold device memory so a big card behaves like a small one.")
    ap.add_argument("--gib", type=float, default=16.0,
                    help="GiB to hold (default: %(default)s)")
    ap.add_argument("--device", default="cuda",
                    help="compute device to hold on (default: %(default)s)")
    ap.add_argument("--block-gib", type=float, default=1.0,
                    help="granularity of the hold (default: %(default)s)")
    ap.add_argument("--keep-free", type=float, default=2.0,
                    help="leave this many GiB for the server (default: %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="hold even if that leaves less than --keep-free")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        print(f"{device}: no device memory to hold", file=sys.stderr)
        return 2
    if args.block_gib <= 0:
        print("--block-gib must be positive", file=sys.stderr)
        return 2

    balloon = Balloon(device, int(args.block_gib * GIB))
    if warning := _shared_pool_warning(device):
        print(f"note: {warning}", flush=True)

    free, total = _free(device)
    if not args.force and free - args.gib * GIB < args.keep_free * GIB:
        print(f"refusing: {args.gib:g} GiB leaves {free / GIB:.1f} of {total / GIB:.1f} "
              f"GiB free, under --keep-free {args.keep_free:g}. --force to insist.",
              file=sys.stderr)
        return 1

    balloon.set(args.gib)
    balloon.report()

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: balloon.report())
    if sys.stdin.isatty():
        print("+N more, -N back, =N total, anything else to print, q to quit", flush=True)

    try:
        for line in sys.stdin:      # already at EOF under nohup: signals take over
            command = line.strip()
            if command in ("q", "quit"):
                break
            try:
                if command[:1] == "+":
                    balloon.grow(float(command[1:]))
                elif command[:1] == "-":
                    balloon.shrink(float(command[1:]))
                elif command[:1] == "=":
                    balloon.set(float(command[1:]))
            except ValueError as bad:
                print(f"not a number: {bad}", file=sys.stderr)
            balloon.report()
        else:
            _wait()                 # stdin ran dry without a quit: keep holding
    except KeyboardInterrupt:
        pass
    return 0


def _wait() -> None:
    """Hold until a signal arrives."""
    if hasattr(signal, "pause"):
        while True:
            signal.pause()
    else:                                   # Windows
        import time

        while True:
            time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
