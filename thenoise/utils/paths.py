"""Small filesystem path helpers shared across CLI entry points."""
from __future__ import annotations

import os


def ensure_png_extension(path: str) -> str:
    """Append ``.png`` when *path* has no file extension (PIL cannot infer the
    output format from a bare filename).
    """
    if not os.path.splitext(path)[1]:
        return path + ".png"
    return path


__all__ = ["ensure_png_extension"]
