"""Reading media from a path, and from base64 on the wire: pictures, and the clips and
sounds a reference condition can be."""
from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
from typing import NamedTuple, Optional, Union

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


#: How much of a reference file is read. A reference longer than the clip it conditions
#: is cropped to it anyway, so reading further would only be decoding what gets thrown
#: away -- and the file is whoever sent it, at whatever length they recorded.
MAX_REFERENCE_SECONDS = 60.0


class ReferenceMedia(NamedTuple):
    """What one reference file turned out to be.

    ``frames`` is a single image for a picture and the clip's own for a video, in the
    order it plays; ``pcm`` is its sound, if it carried one, at the rate the file
    carries it rather than the rate whoever reads it wants.
    """

    kind: str
    frames: tuple[Image.Image, ...]
    pcm: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    fps: Optional[float] = None


def verified_media(field: str, encoded: str) -> bytes:
    """One base64 reference from the wire, checked before a multi-minute job is accepted.

    The header of a picture or of a container is all that is read: whether the bytes are
    a picture, a clip or a sound is asked of them again when they are decoded, and a
    client is told about a bad file now rather than after a queue slot.
    """
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"{field} must be valid base64 bytes") from e
    if not data:
        raise ValueError(f"{field} must not be empty")
    try:
        with Image.open(io.BytesIO(data)):
            return data
    except (OSError, Image.DecompressionBombError):
        pass
    try:
        with av.open(io.BytesIO(data)) as container:
            if container.streams.video or container.streams.audio:
                return data
    except (av.FFmpegError, OSError, ValueError):
        pass
    raise ValueError(f"{field} must be a picture, a clip or a sound")


def read_reference(data: bytes, *, name: str = "reference",
                   max_seconds: float = MAX_REFERENCE_SECONDS) -> ReferenceMedia:
    """One reference file as frames and sound, whatever kind of file it was.

    The kind is asked of the bytes and not of a name a client chose, because a picture,
    a clip and a sound all arrive through one field. Nothing is resized and no frame
    count is snapped here: what shape a reference takes is the model's answer and which
    counts the codec runs is the codec's.
    """
    try:
        with Image.open(io.BytesIO(data)) as picture:
            return ReferenceMedia("picture", (picture.convert("RGB"),))
    except (OSError, Image.DecompressionBombError):
        pass

    try:
        with av.open(io.BytesIO(data)) as container:
            frames, fps = _read_frames(container, max_seconds)
            sound, rate = _read_sound(data, max_seconds)
    except av.FFmpegError as e:
        raise ValueError(f"{name} is not a picture, a clip or a sound ({e.strerror or e})"
                         ) from e
    if not frames and sound is None:
        raise ValueError(f"{name} carries no video or audio stream")
    return ReferenceMedia("clip" if frames else "audio", tuple(frames), sound, rate, fps)


def _read_frames(container, max_seconds: float) -> tuple[list[Image.Image], Optional[float]]:
    if not container.streams.video:
        return [], None
    stream = container.streams.video[0]
    rate = stream.average_rate or stream.base_rate
    frames = []
    for frame in container.decode(stream):
        if frame.time is not None and frame.time >= max_seconds:
            break
        frames.append(frame.to_image())
    return frames, (float(rate) if rate else None)


def _read_sound(data: bytes, max_seconds: float) -> tuple[Optional[np.ndarray],
                                                          Optional[int]]:
    """The file's sound as ``[channels, samples]`` float32, at the rate it was recorded."""
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            return None, None
        stream = container.streams.audio[0]
        resampler = None
        parts: list[np.ndarray] = []
        for frame in container.decode(stream):
            if resampler is None:
                resampler = av.AudioResampler(format="fltp", layout=frame.layout,
                                              rate=frame.sample_rate)
            parts.extend(moved.to_ndarray() for moved in resampler.resample(frame))
            if frame.time is not None and frame.time >= max_seconds:
                break
        if resampler is not None:
            parts.extend(moved.to_ndarray() for moved in resampler.resample(None))
        if not parts:
            return None, stream.rate
        return pcm_array(np.concatenate(parts, axis=1)), stream.rate


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
