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
from thenoise.utils.media import pcm_array, pcm_resample
from .spec import ResolvedVideo

#: The audio codec each container carries, the rate its encoder is opened at, and the
#: frame length that encoder consumes. Opus opens only at 8, 12, 16, 24 or 48 kHz and eats
#: 20 ms at a time, so WebM encodes at 48 kHz whatever rate the model's codec answered
#: (H3's is 32 kHz) and the samples are moved there first; AAC takes the rate it is handed
#: and frames itself at 1024 samples.
AUDIO_CODECS = {"mp4": ("aac", None, 1024), "webm": ("libopus", 48000, 960)}


def _prepare_audio(audio, sample_rate):
    if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
        raise ValueError("audio sample rate must be between 8000 and 192000 Hz")
    return pcm_array(audio)


def encode_video(frames: Iterable, spec: ResolvedVideo, token: Cancellation, out: Path,
                 progress, *, audio=None, sample_rate: int | None = None) -> None:
    samples = _prepare_audio(audio, sample_rate) if audio is not None else None
    audio_codec, audio_rate, audio_frame = AUDIO_CODECS[spec.output_format]
    layout = ""
    if samples is not None:
        layout = "mono" if samples.shape[0] == 1 else "stereo"
        if audio_rate and sample_rate != audio_rate:
            samples = pcm_resample(samples, sample_rate, audio_rate, layout)
            sample_rate = audio_rate
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
                audio_stream = container.add_stream(audio_codec, rate=sample_rate)
                audio_stream.layout = layout
                audio_timebase = Fraction(1, sample_rate)

            def encode_audio_until(limit):
                nonlocal audio_pos
                if audio_stream is None:
                    return
                limit = min(limit, samples.shape[1])
                while audio_pos < limit:
                    token.check()
                    remaining = samples.shape[1] - audio_pos
                    if remaining < audio_frame:
                        chunk = np.zeros((samples.shape[0], audio_frame), dtype=np.float32)
                        chunk[:, :remaining] = samples[:, audio_pos:]
                        count = remaining
                    elif limit - audio_pos >= audio_frame:
                        chunk = np.ascontiguousarray(samples[:, audio_pos:audio_pos + audio_frame])
                        count = audio_frame
                    else:
                        break
                    aframe = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
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
