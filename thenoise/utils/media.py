"""Reading media from a path, and from base64 on the wire."""
from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
from typing import Optional, Union

import av
import numpy as np
import torch
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


def pcm_array(pcm: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """PCM as ``[channels, samples]`` float32 in ``[-1, 1]``, finite on every lane.

    A model's tensor and a decoder's array mean the same thing, so neither is moved by
    hand at each call site; a non-finite sample would poison a latent and every frame
    decoded from it, which is worth refusing where it is still cheap.
    """
    if hasattr(pcm, "detach"):
        pcm = pcm.detach().float().cpu().numpy()
    samples = np.asarray(pcm, dtype=np.float32)
    if samples.ndim == 3 and samples.shape[0] == 1:
        samples = samples[0]
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[0] not in (1, 2) or samples.shape[1] == 0:
        raise ValueError("audio must be (channels, samples), mono or stereo")
    if not np.isfinite(samples).all():
        raise ValueError("audio has non-finite samples")
    return np.ascontiguousarray(np.clip(samples, -1, 1))


def pcm_resample(pcm: Union[torch.Tensor, np.ndarray], src_rate: int, dst_rate: int,
                 layout: Optional[str] = None) -> np.ndarray:
    """Move PCM to another rate through libswresample, whole array in and whole array out.

    Sample positions are what a timeline and a latent grid are anchored by, so a resampler
    that streams would hand back a clock of its own. ``layout`` is the channel count's,
    unless a caller already named it for the container it is writing.
    """
    samples = pcm_array(pcm)
    if layout is None:
        layout = "mono" if samples.shape[0] == 1 else "stereo"
    if src_rate == dst_rate:
        return samples
    frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
    frame.sample_rate = src_rate
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=dst_rate)
    parts = [moved.to_ndarray()
             for chunk in (resampler.resample(frame), resampler.resample(None))
             for moved in chunk]
    return np.ascontiguousarray(np.concatenate(parts, axis=1))


__all__ = ["pcm_array", "pcm_resample", "read_media", "verified_picture"]
