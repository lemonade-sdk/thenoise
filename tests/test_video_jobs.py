"""Weight-free video job regression tests: the actual cancellation contract."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from thenoise.video.jobs import VideoJobs
from thenoise.models.wan_video import WanVideo
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
                     frames=9, **kwargs).resolve(WanVideo.DEFAULT_PREFS)


def test_wan_frame_grid_and_duration():
    a = spec(duration=1.0, fps=8)
    # Exact counts win, even when duration was also supplied.
    assert (a.frame_count, a.fps) == (9, 8)
    assert a.public()["duration"] == 9 / 8
    # Request parsing is model-neutral. The Wan model applies its temporal grid.
    from thenoise.models.wan_video import WanVideo
    assert WanVideo.resolve_video(VideoSpec(prompt="x", frames=10)).frame_count == 13
    assert WanVideo.resolve_video(VideoSpec(prompt="x", frames=17)).frame_count == 17
    # Precisely halfway: Lemonade's std::round picks the larger 4n+1 count.
    assert WanVideo.resolve_video(VideoSpec(prompt="x", duration=11/8, fps=8)).frame_count == 13


def test_video_validation_fails_before_job_creation():
    for bad in (dict(width=-1), dict(fps=0), dict(guidance_scale=float("nan")),
                dict(output_format="avi")):
        with pytest.raises(ValueError):
            VideoSpec(prompt="valid", **bad).resolve(WanVideo.DEFAULT_PREFS)


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
                         width=480, height=320, frames=9).resolve(WanVideo.DEFAULT_PREFS)
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
    from thenoise.models.wan_video import WanVideo
    assert VideoSpec(prompt="fox", frames=1000).resolve(WanVideo.DEFAULT_PREFS).frame_count == 1000
    assert WanVideo.resolve_video(VideoSpec(prompt="fox", frames=1000)).frame_count == 1001


def test_video_defaults_are_owned_by_the_model():
    from thenoise.video.http import VideoRequest

    request = VideoRequest(prompt="fox")
    assert request.width is None and request.steps is None
    assert request.fps is None and request.sampler is None
    resolved = WanVideo.resolve_video(request.to_spec())
    assert resolved.width == WanVideo.DEFAULT_PREFS["width"]
    assert resolved.steps == WanVideo.DEFAULT_PREFS["steps"]
    assert resolved.sampler == WanVideo.DEFAULT_PREFS["sampler"]

    # The generic contract follows model preferences, not HTTP-side literals.
    updated = {**WanVideo.DEFAULT_PREFS, "steps": 7, "sampler": "euler"}
    changed = request.to_spec().resolve(updated)
    assert (changed.steps, changed.sampler) == (7, "euler")

def test_exit_clears_managed_video_cache_but_preserves_other_files(tmp_path):
    cache = tmp_path / "video_cache"
    cache.mkdir()
    unrelated = cache / "important.mp4"
    unrelated.write_bytes(b"user file")
    stale = cache / ("vid_" + "0" * 32 + ".webm")
    stale.write_bytes(b"old video")

    def render(clip, token, progress, path):
        path.write_bytes(b"complete")

    jobs = VideoJobs(render, cache, model="wan")
    sync = jobs.render_sync(spec(seed=21))
    job_id = jobs.submit(spec(seed=22))["id"]
    wait_until(lambda: jobs.status(job_id)["status"] == "completed")
    async_path = jobs.content(job_id)[1]
    assert sync.exists() and async_path.exists() and stale.exists()

    jobs.close()
    assert not sync.exists() and not async_path.exists() and not stale.exists()
    assert unrelated.read_bytes() == b"user file"
    jobs.close()  # repeated shutdown is harmless


def test_exit_waits_for_running_worker_before_cache_cleanup(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def render(clip, token, progress, path):
        entered.set()
        assert release.wait(3)
        path.write_bytes(b"partial")
        token.check()

    jobs = VideoJobs(render, tmp_path, model="wan")
    job_id = jobs.submit(spec(seed=99))["id"]
    assert entered.wait(2)
    closer = threading.Thread(target=jobs.close)
    closer.start()
    try:
        wait_until(lambda: jobs._jobs[job_id].token.requested)
        assert closer.is_alive(), "shutdown must wait for the worker"
    finally:
        release.set()
        closer.join(timeout=4)
    assert not closer.is_alive()
    assert jobs.status(job_id)["status"] == "canceled"
    assert not list(tmp_path.glob("vid_*.webm"))


def test_sync_video_reuses_resolver_and_cleans_on_failure(tmp_path):
    seen = []

    def render(spec, token, progress, path):
        seen.append(spec)
        path.write_bytes(b"partial")
        if len(seen) == 1:
            raise RuntimeError("encoder failed")
        path.write_bytes(b"complete")

    jobs = VideoJobs(render, tmp_path, model="Wan2.2-TI2V-5B",
                     resolve=WanVideo.resolve_video)
    try:
        with pytest.raises(RuntimeError, match="encoder failed"):
            jobs.render_sync(VideoSpec(prompt="fox", frames=9))
        assert not list(tmp_path.glob("sync_*"))
        output = jobs.render_sync(VideoSpec(prompt="fox", frames=9,
                                            lora_specs=["style:0.5"]))
        assert output.read_bytes() == b"complete"
        assert seen[-1].lora_specs == ["style:0.5"]
        output.unlink()
    finally:
        jobs.close()



