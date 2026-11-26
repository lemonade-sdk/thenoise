"""Cancellation and atomic file publishing in the video encoder, with fake AV."""
from __future__ import annotations

import types
from pathlib import Path

import av
import numpy as np
import pytest

from thenoise.utils.media import pcm_resample
from thenoise.video.container import AUDIO_CODECS, encode_video
from thenoise.jobs import Cancellation, JobCancelled
from thenoise.video.spec import OUTPUT_FORMATS, VideoSpec
from conftest import wan_prefs


def fake_av(monkeypatch, *, with_audio=False):
    class Stream:
        def __init__(self, codec):
            self.codec = codec
            self.width = 0
            self.height = 0
            self.pix_fmt = ""

        def encode(self, frame=None):
            if self.codec == "aac":
                return [b"a"] if frame is not None else [b"z"]
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
            if codec == "aac":
                assert with_audio and rate == 48000
            else:
                assert codec in ("libx264", "libvpx-vp9")
                assert rate == (24 if with_audio else 8)
            return Stream(codec)

        def mux(self, packet):
            self.file.write(packet)

    fake = types.ModuleType("av")
    fake.open = lambda path, mode, format: Container(path)
    fake.VideoFrame = types.SimpleNamespace(from_ndarray=lambda data, format: data)
    fake.AudioFrame = types.SimpleNamespace(
        from_ndarray=lambda data, format, layout: types.SimpleNamespace(
            data=data, format=format, layout=layout))
    monkeypatch.setattr("thenoise.video.container.av", fake)


def spec():
    return VideoSpec(prompt="fox", width=480, height=320, frames=9,
                     fps=8, output_format="mp4").resolve(wan_prefs())


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

    with pytest.raises(JobCancelled):
        encode_video(frames, clip, cancel, target, stop_after_one)
    assert target.read_bytes() == b"old-completed-clip"
    assert not (tmp_path / "clip.mp4.part").exists()


def test_mp4_audio_video_mux_and_atomic_publish(tmp_path, monkeypatch):
    """AAC shares the existing PyAV muxer and atomic output publication."""
    fake_av(monkeypatch, with_audio=True)
    clip = VideoSpec(prompt="fox", width=64, height=64, frames=9,
                     fps=24, output_format="mp4").resolve(wan_prefs())
    target = tmp_path / "av-clip.mp4"
    frames = (np.zeros((clip.height, clip.width, 3), dtype=np.uint8)
              for _ in range(clip.frame_count))
    pcm = np.zeros((1, round(clip.frame_count * 48000 / clip.fps)), dtype=np.float32)
    progress = []
    encode_video(frames, clip, Cancellation(), target,
                 lambda *args: progress.append(args), audio=pcm, sample_rate=48000)

    data = target.read_bytes()
    assert data.count(b"f") == clip.frame_count
    assert b"a" in data  # AAC frames were passed to the same container
    assert data.endswith(b"ez")  # both encoder streams flushed before publication
    assert not target.with_name(target.name + ".part").exists()
    assert progress[-1] == ("encoding", clip.frame_count, clip.frame_count)


def test_every_container_the_request_allows_carries_sound():
    """The codec table must cover every container the request allows."""
    assert set(OUTPUT_FORMATS) == set(AUDIO_CODECS)


#: 32 kHz is a model's own rate (H3's audio codec) rather than one every container's codec
#: will open at, so the table's rate is what lands in the file.
@pytest.mark.parametrize("container", OUTPUT_FORMATS)
@pytest.mark.parametrize("source_rate", [48000, 32000])
def test_generated_sound_is_demuxable_from_every_container(tmp_path, container, source_rate):
    """Only a real container proves the codec the table names exists in this build, at the
    rate it will actually open."""
    clip = VideoSpec(prompt="fox", width=64, height=64, frames=9, fps=24,
                     output_format=container).resolve(wan_prefs())
    target = tmp_path / f"clip.{container}"
    frames = (np.zeros((clip.height, clip.width, 3), dtype=np.uint8)
              for _ in range(clip.frame_count))
    ramp = np.arange(int(clip.frame_count / clip.fps * source_rate), dtype=np.float32)
    sound = np.sin(2 * np.pi * 440 * ramp / source_rate)[None, :]
    encode_video(frames, clip, Cancellation(), target, lambda *_: None,
                 audio=sound, sample_rate=source_rate)

    expected, container_rate, codec_frame = AUDIO_CODECS[container]
    rate = container_rate or source_rate
    with av.open(str(target)) as media:
        assert sorted(stream.type for stream in media.streams) == ["audio", "video"]
        audio, = [stream for stream in media.streams if stream.type == "audio"]
        assert audio.codec_context.sample_rate == rate
        # The decoder names Opus without the encoder's libprefix.
        assert audio.codec_context.name in (expected, expected.removeprefix("lib"))
        demuxed = sum(frame.samples for frame in media.decode(audio))
    # The tail is padded out to a whole codec frame, never trimmed short of the clip.
    wanted = clip.frame_count * rate / clip.fps
    assert wanted <= demuxed < wanted + codec_frame


def test_moving_audio_to_another_rate_keeps_it_on_the_clock():
    """A resample that slid the sound sideways would put the lips ahead of the words, and
    a correct sample count would hide it. The peak is where the ratio puts it."""
    n = 1600
    click = np.zeros((2, n), dtype=np.float32)
    click[:, n // 2] = 1.0

    moved = pcm_resample(click, 32000, 48000)

    assert moved.shape == (2, n * 3 // 2)
    assert int(np.argmax(moved[0])) == n // 2 * 3 // 2
    assert float(moved[0].max()) == pytest.approx(1.0, abs=1e-3)


def test_bad_audio_does_not_replace_existing_video(tmp_path):
    """Invalid PCM is rejected before any container or partial file is created."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"existing-complete-video")
    clip = spec()
    frames = (np.zeros((clip.height, clip.width, 3), dtype=np.uint8)
              for _ in range(clip.frame_count))
    with pytest.raises(ValueError, match="mono or stereo"):
        encode_video(frames, clip, Cancellation(), target, lambda *_: None,
                     audio=np.zeros((3, 16), dtype=np.float32), sample_rate=48000)
    assert target.read_bytes() == b"existing-complete-video"
    assert not target.with_name(target.name + ".part").exists()
