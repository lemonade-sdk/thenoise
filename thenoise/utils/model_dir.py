"""Shared helpers for model files living in a directory.

Both LoRAs and pixel upscalers are selected by name from a configured base
directory. The name-parsing, path-resolution, and directory-listing logic is
identical, so it lives here and is called by the model with the relevant base
path (``lora_dir`` or ``upscaler_dir``).
"""
from __future__ import annotations


def ensure_safetensors(name: str) -> str:
    """Return ``name`` with a trailing ``.safetensors`` appended if missing."""
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    return name


def strip_safetensors(name: str) -> str:
    """Return ``name`` with a trailing ``.safetensors`` removed if present."""
    if name.endswith(".safetensors"):
        name = name[: -len(".safetensors")]
    return name


def resolve_in_dir(base_dir: str, filename: str) -> str:
    """Resolve ``filename`` to an absolute path within ``base_dir``.

    Subdirectories are allowed, but ``..`` components that would escape
    ``base_dir`` raise ``ValueError``.
    """
    if not base_dir:
        raise ValueError("base directory is not set")
    import os

    base = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.join(base_dir, filename))
    if not candidate.startswith(base + os.sep) and candidate != base:
        raise ValueError("path escapes base directory")
    return candidate


def list_safetensors(base_dir: str) -> list[str]:
    """Recursively list ``.safetensors`` names relative to ``base_dir``.

    Names are relative paths with the ``.safetensors`` suffix stripped (e.g.
    ``"12345_something"`` or ``"sub/style"``). Returns ``[]`` when ``base_dir``
    is empty.
    """
    if not base_dir:
        return []
    import os

    names = []
    for root, _dirs, files in os.walk(base_dir):
        for name in sorted(files):
            if not name.endswith(".safetensors"):
                continue
            rel = os.path.relpath(os.path.join(root, name), base_dir)
            names.append(strip_safetensors(rel))
    return sorted(names)
