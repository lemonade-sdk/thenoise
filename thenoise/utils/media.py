"""Reading media from a path, and from base64 on the wire."""
from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path

from PIL import Image


def read_media(path: str, command: str, flag: str = "--image") -> bytes:
    """One input file's bytes, refusal phrased as the command that asked for it."""
    try:
        return Path(path).read_bytes()
    except OSError as e:
        raise SystemExit(f"{command}: {flag}: {e}") from e


def verified_picture(field: str, encoded: str) -> bytes:
    """One base64 picture from the wire, checked before a multi-minute job is accepted;
    PIL's dimension check happens before the image is fully decompressed."""
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"{field} must be valid base64 bytes") from e
    if not data:
        raise ValueError(f"{field} must not be empty")
    try:
        with Image.open(io.BytesIO(data)) as picture:
            picture.verify()
    except (OSError, Image.DecompressionBombError) as e:
        raise ValueError(f"{field} must be a valid image") from e
    return data


__all__ = ["read_media", "verified_picture"]
