"""Video container encoding via PyAV.

Frames are fed individually; no giant RGBA video array is allocated by the muxer.
The output becomes visible only after close/trailer and an atomic rename.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .jobs import Cancellation
from .spec import ResolvedVideo


def encode_video(frames: Iterable, spec: ResolvedVideo, token: Cancellation, out: Path,
                 progress) -> None:
    try:
        import av
        import numpy as np
    except ImportError as e:
        raise RuntimeError("video encoding needs the optional 'video' extras (PyAV)") from e

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
                progress("encoding", frame_count, spec.frame_count)
            if frame_count != spec.frame_count:
                raise ValueError(f"decoder yielded {frame_count} frames, expected {spec.frame_count}")
            token.check()
            for packet in stream.encode():
                container.mux(packet)
        token.check()
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
