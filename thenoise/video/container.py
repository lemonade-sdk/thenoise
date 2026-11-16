"""Video container encoding via PyAV.

Frames are fed individually; no giant RGBA video array is allocated by the muxer.
The output becomes visible only after close/trailer and an atomic rename.
"""
from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import av
import numpy as np

from thenoise.jobs import Cancellation
from .spec import ResolvedVideo


def _prepare_audio(audio, sample_rate, np):
    if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
        raise ValueError("audio sample rate must be between 8000 and 192000 Hz")
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 3 and samples.shape[0] == 1:
        samples = samples[0]
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[0] not in (1, 2) or samples.shape[1] == 0:
        raise ValueError("audio must be (channels, samples), mono or stereo")
    if not np.isfinite(samples).all():
        raise ValueError("generated audio has non-finite samples")
    return np.ascontiguousarray(np.clip(samples, -1, 1))


def encode_video(frames: Iterable, spec: ResolvedVideo, token: Cancellation, out: Path,
                 progress, *, audio=None, sample_rate: int | None = None) -> None:
    if audio is not None and spec.output_format != "mp4":
        raise ValueError("synchronized audio currently requires MP4")
    samples = _prepare_audio(audio, sample_rate, np) if audio is not None else None
    if samples is not None:
        # Pad/trim the generated audio to the requested video timeline.
        wanted = round(spec.frame_count * sample_rate / spec.fps)
        if samples.shape[1] < wanted:
            samples = np.pad(samples, ((0, 0), (0, wanted - samples.shape[1])))
        else:
            samples = np.ascontiguousarray(samples[:, :wanted])
    codec = "libx264" if spec.output_format == "mp4" else "libvpx-vp9"
    tmp = out.with_name(out.name + ".part")
    frame_count = 0
    try:
        with av.open(str(tmp), mode="w", format=spec.output_format) as container:
            stream = container.add_stream(codec, rate=spec.fps)
            stream.width, stream.height = spec.width, spec.height
            stream.pix_fmt = "yuv420p"
            if codec == "libx264":
                stream.options = {"crf": "17"}
            audio_stream = None
            audio_pos = 0
            if samples is not None:
                audio_stream = container.add_stream("aac", rate=sample_rate)
                audio_layout = "mono" if samples.shape[0] == 1 else "stereo"
                audio_stream.layout = audio_layout
                audio_timebase = Fraction(1, sample_rate)

            def encode_audio_until(limit):
                nonlocal audio_pos
                if audio_stream is None:
                    return
                limit = min(limit, samples.shape[1])
                while audio_pos < limit:
                    token.check()
                    remaining = samples.shape[1] - audio_pos
                    if remaining < 1024:
                        chunk = np.zeros((samples.shape[0], 1024), dtype=np.float32)
                        chunk[:, :remaining] = samples[:, audio_pos:]
                        count = remaining
                    elif limit - audio_pos >= 1024:
                        chunk = np.ascontiguousarray(samples[:, audio_pos:audio_pos + 1024])
                        count = 1024
                    else:
                        break
                    aframe = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=audio_layout)
                    aframe.sample_rate = sample_rate
                    aframe.time_base = audio_timebase
                    aframe.pts = audio_pos
                    for packet in audio_stream.encode(aframe):
                        container.mux(packet)
                    audio_pos += count

            for frame in frames:
                token.check()
                if hasattr(frame, "convert"):
                    frame = np.asarray(frame.convert("RGB"))
                else:
                    frame = np.asarray(frame)
                if frame.shape != (spec.height, spec.width, 3):
                    raise ValueError(f"decoded video frame has unexpected shape {frame.shape}")
                av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)
                frame_count += 1
                if audio_stream is not None:
                    encode_audio_until(round(frame_count * sample_rate / spec.fps))
                progress("encoding", frame_count, spec.frame_count)
            if frame_count != spec.frame_count:
                raise ValueError(f"decoder yielded {frame_count} frames, expected {spec.frame_count}")
            token.check()
            for packet in stream.encode():
                container.mux(packet)
            if audio_stream is not None:
                encode_audio_until(samples.shape[1])
                for packet in audio_stream.encode():
                    container.mux(packet)
        token.check()
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
