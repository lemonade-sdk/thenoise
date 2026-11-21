"""Small filesystem path helpers shared across CLI entry points."""
from __future__ import annotations

import os
from pathlib import Path


def ensure_png_extension(path: str) -> str:
    """Append ``.png`` when *path* has no file extension (PIL cannot infer the
    output format from a bare filename).
    """
    if not os.path.splitext(path)[1]:
        return path + ".png"
    return path


def named_output(name: str, output_format: str, noun: str, command: str) -> Path:
    """Where a rendered file goes: ``name`` extended with the container actually written,
    a name promising a different one refused rather than renamed."""
    suffix = Path(name).suffix
    if not suffix:
        return Path(f"{name}.{output_format}")
    if suffix != f".{output_format}":
        raise SystemExit(
            f"{command}: --out {name} ends in {suffix} but this {noun} is {output_format}")
    return Path(name)


__all__ = ["ensure_png_extension", "named_output"]
