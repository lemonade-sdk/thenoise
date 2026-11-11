"""Weight-free video job regression tests: the actual cancellation contract."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from thenoise.video.jobs import VideoJobs
from thenoise.video.spec import VideoSpec


def wait_until(predicate, timeout=3):
    limit = time.monotonic() + timeout
    while time.monotonic() < limit:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timeout waiting for video state")


def spec(**kwargs):
    return VideoSpec(prompt="a fox in snow", width=480, height=320,
                     frames=9, **kwargs).resolve()


def test_wan_frame_grid_and_duration():
    a = spec(duration=1.0, fps=8)
    # Exact counts win, even when duration was also supplied.
    assert (a.frame_count, a.fps) == (9, 8)
    assert a.public()["duration"] == 9 / 8
    assert VideoSpec(prompt="x", frames=10).resolve().frame_count == 13
    assert VideoSpec(prompt="x", frames=17).resolve().frame_count == 17
    # Precisely halfway: Lemonade's std::round picks the larger 4n+1 count.
    assert VideoSpec(prompt="x", duration=11/8, fps=8).resolve().frame_count == 13


def test_video_validation_fails_before_job_creation():
    for bad in (dict(width=4096), dict(fps=0), dict(guidance_scale=float("nan")),
                dict(output_format="avi")):
        with pytest.raises(ValueError):
            VideoSpec(prompt="valid", **bad).resolve()


def test_cancel_running_job_does_not_kill_worker_or_next_job(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def render(clip, token, progress, path):
        calls.append(clip.seed)
        if len(calls) == 1:
            entered.set()
            release.wait(2)
            token.check()
        progress("sampling", clip.steps, clip.steps)
        path.write_bytes(b"WEBM")

    jobs = VideoJobs(render, tmp_path, model="Wan2.2-TI2V-5B")
    try:
        first = jobs.submit(spec(seed=10))["id"]
        assert entered.wait(2)
        second = jobs.submit(spec(seed=11))["id"]
        assert jobs.cancel(first) is True
        assert jobs.status(first)["status"] == "running", "running cancel needs acknowledgment"
        release.set()
        wait_until(lambda: jobs.status(first)["status"] == "canceled")
        wait_until(lambda: jobs.status(second)["status"] == "completed")
        assert jobs.content(first) == ("canceled", None)
        assert jobs.content(second)[1].read_bytes() == b"WEBM"
        assert calls == [10, 11]
    finally:
        jobs.close()


def test_queued_cancel_skips_inference(tmp_path):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def render(clip, token, progress, path):
        calls.append(clip.seed)
        started.set()
        release.wait(2)
        token.check()
        path.write_bytes(b"WEBM")

    jobs = VideoJobs(render, tmp_path, model="wan")
    try:
        first = jobs.submit(spec(seed=1))["id"]
        assert started.wait(2)
        second = jobs.submit(spec(seed=2))["id"]
        assert jobs.cancel(second) is True
        assert jobs.status(second)["status"] == "canceled"
        assert jobs.cancel(second) is False
        release.set()
        wait_until(lambda: jobs.status(first)["status"] == "completed")
        assert calls == [1]
    finally:
        jobs.close()


def test_failed_job_does_not_poison_next_job(tmp_path):
    calls = []

    def render(clip, token, progress, path):
        calls.append(clip.seed)
        if len(calls) == 1:
            raise RuntimeError("decode broke")
        path.write_bytes(b"WEBM")

    jobs = VideoJobs(render, tmp_path, model="wan")
    try:
        a = jobs.submit(spec(seed=1))["id"]
        b = jobs.submit(spec(seed=2))["id"]
        wait_until(lambda: jobs.status(b)["status"] == "completed")
        assert jobs.status(a)["status"] == "failed"
        assert "decode broke" in jobs.status(a)["error"]
        assert jobs.content(a) == ("failed", None)
    finally:
        jobs.close()


def test_bounded_queue_and_pruning(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def render(clip, token, progress, path):
        entered.set()
        release.wait(2)
        token.check()
        path.write_bytes(b"a")

    jobs = VideoJobs(render, tmp_path, model="wan", queue_limit=1, history_limit=2)
    try:
        a = jobs.submit(spec(seed=1))["id"]
        assert entered.wait(2)
        b = jobs.submit(spec(seed=2))["id"]
        with pytest.raises(OverflowError):
            jobs.submit(spec(seed=3))
        release.set()
        wait_until(lambda: jobs.status(b)["status"] == "completed")
        c = jobs.submit(spec(seed=3))["id"]
        assert jobs.status(a) is None
        assert not (tmp_path / (a + ".webm")).exists()
        wait_until(lambda: jobs.status(c)["status"] == "completed")
    finally:
        jobs.close()


def test_completed_i2v_releases_input_media_but_retains_mode(tmp_path):
    jobs = VideoJobs(lambda clip, cancel, progress, path: path.write_bytes(b"webm"),
                     tmp_path, model="wan")
    try:
        clip = VideoSpec(prompt="a fox", image=b"large-binary-reference" * 10000,
                         width=480, height=320, frames=9).resolve()
        job_id = jobs.submit(clip)["id"]
        wait_until(lambda: jobs.status(job_id)["status"] == "completed")
        status = jobs.status(job_id)
        assert status["resolved"]["mode"] == "i2v"
        assert status["frame_count"] == 9
        assert jobs._jobs[job_id].spec is None
        assert jobs.content(job_id)[1].exists()
    finally:
        jobs.close()


def test_frame_count_does_not_have_a_static_limit():
    assert VideoSpec(prompt="fox", frames=1000).resolve().frame_count == 1001
