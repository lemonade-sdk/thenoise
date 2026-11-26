"""A reference file read back: what kind it was, its frames, and the sound that came with
them — from real files written through the same library that reads them."""
from __future__ import annotations

import base64
import io

import av
import numpy as np
import pytest
from PIL import Image

from thenoise.utils.media import (MAX_REFERENCE_SECONDS, read_reference, verified_media,
                                  verified_picture)

FRAME_RATE = 24
RATE = 32000


def _png(width: int = 64, height: int = 32) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (10, 200, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _write(*, frames: int, rate: int, sound: bool, format: str,
           codec: str | None = "libx264") -> bytes:
    """Solid frames, each darker than the last, with a 440 Hz tone as their soundtrack.

    One audio chunk per video frame, so the sound lasts as long as the pictures whatever
    rate it was written at.
    """
    buffer = io.BytesIO()
    with av.open(buffer, "w", format=format) as container:
        video = None
        if codec is not None:
            video = container.add_stream(codec, rate=FRAME_RATE)
            video.width = video.height = 32
            video.pix_fmt = "yuv420p"
        audio = container.add_stream("pcm_s16le", rate=rate) if sound else None
        if audio is not None:
            # Asked for explicitly: a container that is left to choose answers stereo,
            # and the reader believes the layout the file states.
            audio.layout = "mono"
        chunk = max(1, round(rate / FRAME_RATE))
        for index in range(frames):
            if video is not None:
                shade = np.full((32, 32, 3), max(8, 200 - index * 4), dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(shade, format="rgb24")
                for packet in video.encode(frame):
                    container.mux(packet)
            if audio is not None:
                step = np.arange(chunk) + index * chunk
                tone = (np.sin(step * 2 * np.pi * 440.0 / rate) * 12000).astype(np.int16)
                frame = av.AudioFrame.from_ndarray(tone.reshape(1, -1), format="s16",
                                                   layout="mono")
                frame.sample_rate = rate
                for packet in audio.encode(frame):
                    container.mux(packet)
        for stream in (video, audio):
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
    return buffer.getvalue()


def _clip(frames: int = 12, *, rate: int = RATE, sound: bool = True) -> bytes:
    return _write(frames=frames, rate=rate, sound=sound, format="mp4")


def _wav(seconds: float = 0.5, *, rate: int = RATE) -> bytes:
    return _write(frames=int(seconds * FRAME_RATE), rate=rate, sound=True,
                  format="wav", codec=None)


# --- what the file was --------------------------------------------------------


def test_a_picture_arrives_as_one_frame_and_no_sound():
    read = read_reference(_png())
    assert read.kind == "picture"
    assert [image.size for image in read.frames] == [(64, 32)]
    assert read.pcm is None and read.sample_rate is None and read.fps is None


def test_a_clip_arrives_with_its_frames_in_the_order_they_play():
    read = read_reference(_clip(frames=12))
    assert read.kind == "clip"
    assert len(read.frames) == 12
    assert read.fps == pytest.approx(FRAME_RATE)
    brightness = [float(np.asarray(image).mean()) for image in read.frames]
    assert brightness == sorted(brightness, reverse=True)


def test_a_clips_sound_arrives_with_it_at_the_length_of_the_frames():
    read = read_reference(_clip(frames=12))
    assert read.sample_rate == RATE
    assert read.pcm.shape[0] == 1
    wanted = 12 / FRAME_RATE * RATE
    assert abs(read.pcm.shape[1] - wanted) < 0.2 * wanted


def test_the_sound_of_a_file_at_another_rate_is_left_at_that_rate():
    """The codec answers for the rate it encodes at; a reader that quietly moved the
    samples would leave the frames and the sound disagreeing about the timeline."""
    read = read_reference(_wav(0.5, rate=44100))
    assert read.kind == "audio"
    assert read.sample_rate == 44100
    assert read.frames == ()
    assert abs(read.pcm.shape[1] - 0.5 * 44100) < 0.2 * 0.5 * 44100


def test_a_sound_on_its_own_is_a_reference_too():
    read = read_reference(_wav())
    assert read.kind == "audio"
    assert read.frames == ()
    assert read.pcm is not None


def test_a_silent_clip_is_still_a_clip():
    read = read_reference(_clip(frames=8, sound=False))
    assert read.kind == "clip"
    assert len(read.frames) == 8
    assert read.pcm is None


def test_a_long_file_is_read_only_as_far_as_it_could_be_used():
    long = _clip(frames=48)
    assert len(read_reference(long).frames) == 48
    short = read_reference(long, max_seconds=0.25)
    assert len(short.frames) == 6
    assert short.pcm.shape[1] < 0.35 * RATE
    assert MAX_REFERENCE_SECONDS == 60.0


def test_a_file_that_is_neither_picture_nor_sound_is_refused_by_name():
    with pytest.raises(ValueError, match="my-ref.*picture, a clip or a sound"):
        read_reference(b"not a media file at all", name="my-ref")


# --- the wire's promise -------------------------------------------------------


def test_the_wire_takes_a_clip_where_it_used_to_want_a_picture():
    clip = base64.b64encode(_clip(frames=6)).decode()
    assert verified_media("references", clip)
    assert verified_media("references", base64.b64encode(_png()).decode())


def test_the_wire_still_refuses_what_is_no_media_at_all():
    with pytest.raises(ValueError, match="picture, a clip or a sound"):
        verified_media("references", base64.b64encode(b"nonsense").decode())
    with pytest.raises(ValueError, match="valid base64"):
        verified_media("references", "not base64!")


def test_a_picture_field_still_wants_only_a_picture():
    """`verified_picture` guards the keyframe fields, which are frames of the clip and
    have nowhere a container's length would mean anything."""
    with pytest.raises(ValueError, match="valid image"):
        verified_picture("image", base64.b64encode(_clip(frames=6)).decode())
