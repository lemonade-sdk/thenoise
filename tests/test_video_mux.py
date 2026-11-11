"""Cancellation and atomic file publishing in the video encoder, with fake AV."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from thenoise.video.container import encode_video
from thenoise.video.jobs import Cancellation, VideoCancelled
from thenoise.video.spec import VideoSpec


def fake_av(monkeypatch):
    class Stream:
        def __init__(self):
            self.width = 0
            self.height = 0
            self.pix_fmt = ""

        def encode(self, frame=None):
            return [b"f"] if frame is not None else [b"e"]

    class Container:
        def __init__(self, path):
            self.path = Path(path)
            self.file = None

        def __enter__(self):
            self.file = self.path.open("wb")
            return self

        def __exit__(self, *_):
            self.file.close()

        def add_stream(self, codec, rate):
            assert codec in ("libx264", "libvpx-vp9")
            assert rate == 8
            return Stream()

        def mux(self, packet):
            self.file.write(packet)

    fake = types.ModuleType("av")
    fake.open = lambda path, mode, format: Container(path)
    fake.VideoFrame = types.SimpleNamespace(from_ndarray=lambda data, format: data)
    monkeypatch.setitem(sys.modules, "av", fake)


def spec():
    return VideoSpec(prompt="fox", width=480, height=320, frames=9,
                     fps=8, output_format="mp4").resolve()


def test_encoded_clip_is_published_only_after_mux_is_closed(tmp_path, monkeypatch):
    fake_av(monkeypatch)
    target = tmp_path / "clip.mp4"
    clip = spec()
    frames = [np.zeros((clip.height, clip.width, 3), dtype=np.uint8)] * clip.frame_count
    checks = []

    def progress(stage, step, count):
        assert not target.exists(), "partial clips must never appear as complete files"
        checks.append((stage, step, count))

    encode_video(frames, clip, Cancellation(), target, progress)
    assert target.read_bytes() == b"f" * 9 + b"e"
    assert checks[-1] == ("encoding", 9, 9)
    assert not (tmp_path / "clip.mp4.part").exists()


def test_encoder_cancel_removes_partial_file_and_preserves_existing_output(tmp_path, monkeypatch):
    fake_av(monkeypatch)
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"old-completed-clip")
    clip = spec()
    frames = [np.zeros((clip.height, clip.width, 3), dtype=np.uint8)] * clip.frame_count
    cancel = Cancellation()

    def stop_after_one(stage, step, count):
        cancel.cancel()

    with pytest.raises(VideoCancelled):
        encode_video(frames, clip, cancel, target, stop_after_one)
    assert target.read_bytes() == b"old-completed-clip"
    assert not (tmp_path / "clip.mp4.part").exists()
