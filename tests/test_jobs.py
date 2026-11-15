"""Weight-free job regression tests: the actual cancellation contract.

The queue knows nothing about mediums, so these run on a stand-in payload and renderer;
what a medium does with the same queue is the business of `test_video_http` and
`test_jobs_image`.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from thenoise.jobs import JobPlan, JobQueue


def wait_until(predicate, timeout=3):
    limit = time.monotonic() + timeout
    while time.monotonic() < limit:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timeout waiting for job state")


def write_out(payload, token, progress, path):
    progress("sampling", payload.steps, payload.steps)
    path.write_bytes(b"BYTES")


def plan(render=write_out, *, seed=1, steps=1, **kwargs):
    """One medium's worth of work: a payload the renderer reads, an echo to hand back."""
    defaults = {"kind": "clip", "model": "test", "suffix": ".bin",
                "payload": SimpleNamespace(seed=seed, steps=steps),
                "public": {"frames": 9}, "render": render}
    defaults.update(kwargs)
    return JobPlan(**defaults)


def test_cancel_running_job_does_not_kill_worker_or_next_job(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def render(payload, token, progress, path):
        calls.append(payload.seed)
        if len(calls) == 1:
            entered.set()
            release.wait(2)
            token.check()
        path.write_bytes(b"BYTES")

    queue = JobQueue(tmp_path)
    try:
        first = queue.submit(plan(render=render, seed=10))["id"]
        assert entered.wait(2)
        second = queue.submit(plan(render=render, seed=11))["id"]
        assert queue.cancel(first) is True
        assert queue.status(first)["status"] == "running", "running cancel needs acknowledgment"
        release.set()
        wait_until(lambda: queue.status(first)["status"] == "canceled")
        wait_until(lambda: queue.status(second)["status"] == "completed")
        assert queue.content(first) == ("canceled", None)
        assert queue.content(second)[1].read_bytes() == b"BYTES"
        assert calls == [10, 11]
    finally:
        queue.close()


def test_queued_cancel_skips_the_run(tmp_path):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def render(payload, token, progress, path):
        calls.append(payload.seed)
        started.set()
        release.wait(2)
        token.check()
        path.write_bytes(b"BYTES")

    queue = JobQueue(tmp_path)
    try:
        first = queue.submit(plan(render=render, seed=1))["id"]
        assert started.wait(2)
        second = queue.submit(plan(render=render, seed=2))["id"]
        assert queue.cancel(second) is True
        assert queue.status(second)["status"] == "canceled"
        assert queue.cancel(second) is False
        release.set()
        wait_until(lambda: queue.status(first)["status"] == "completed")
        assert calls == [1]
    finally:
        queue.close()


def test_a_canceled_render_is_a_cancellation_not_a_failure(tmp_path):
    """A renderer that was told to stop says so by raising; the error is not its own."""
    def render(payload, token, progress, path):
        token.cancel()
        token.check()

    queue = JobQueue(tmp_path)
    try:
        job_id = queue.submit(plan(render=render))["id"]
        wait_until(lambda: queue.status(job_id)["status"] == "canceled")
        assert queue.status(job_id).get("error") is None
    finally:
        queue.close()


def test_failed_job_does_not_poison_next_job(tmp_path):
    calls = []

    def render(payload, token, progress, path):
        calls.append(payload.seed)
        if len(calls) == 1:
            raise RuntimeError("decode broke")
        path.write_bytes(b"BYTES")

    queue = JobQueue(tmp_path)
    try:
        a = queue.submit(plan(render=render, seed=1))["id"]
        b = queue.submit(plan(render=render, seed=2))["id"]
        wait_until(lambda: queue.status(b)["status"] == "completed")
        assert queue.status(a)["status"] == "failed"
        assert "decode broke" in queue.status(a)["error"]
        assert queue.content(a) == ("failed", None)
    finally:
        queue.close()


def test_progress_is_the_renderers_stage_and_count(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def render(payload, token, progress, path):
        progress("conditioning")
        for step in range(payload.steps):
            progress("sampling", step + 1, payload.steps)
        entered.set()
        assert release.wait(2)
        path.write_bytes(b"BYTES")

    queue = JobQueue(tmp_path)
    try:
        job_id = queue.submit(plan(render=render, steps=4))["id"]
        assert entered.wait(2)
        running = queue.status(job_id)
        assert running["status"] == "running"
        assert running["progress"] == {"stage": "sampling", "step": 4, "of": 4}
        release.set()
        wait_until(lambda: queue.status(job_id)["status"] == "completed")
        status = queue.status(job_id)
        assert status["kind"] == "clip" and status["model"] == "test"
        assert status["resolved"] == {"frames": 9}
        assert status["mime_type"] == "application/octet-stream"
    finally:
        queue.close()


def test_the_listing_finds_a_job_whose_id_was_not_kept(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def render(payload, token, progress, path):
        entered.set()
        release.wait(2)
        path.write_bytes(b"BYTES")

    queue = JobQueue(tmp_path)
    try:
        first = queue.submit(plan(render=render, seed=1))["id"]
        assert entered.wait(2)
        second = queue.submit(plan(seed=2))["id"]

        listed = queue.listing()
        assert [job["id"] for job in listed] == [first, second]
        assert [job["status"] for job in listed] == ["running", "queued"]
        assert listed[0]["progress"]["stage"] == "loading"

        release.set()
        wait_until(lambda: all(job["status"] == "completed" for job in queue.listing()))
    finally:
        queue.close()


def test_bounded_queue_and_pruning(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def render(payload, token, progress, path):
        entered.set()
        release.wait(2)
        token.check()
        path.write_bytes(b"a")

    queue = JobQueue(tmp_path, queue_limit=1, history_limit=2)
    try:
        a = queue.submit(plan(render=render, seed=1))["id"]
        assert entered.wait(2)
        b = queue.submit(plan(render=render, seed=2))["id"]
        with pytest.raises(OverflowError):
            queue.submit(plan(render=render, seed=3))
        release.set()
        wait_until(lambda: queue.status(b)["status"] == "completed")
        c = queue.submit(plan(render=render, seed=3))["id"]
        assert queue.status(a) is None
        assert not (tmp_path / (a + ".bin")).exists()
        wait_until(lambda: queue.status(c)["status"] == "completed")
    finally:
        queue.close()


def test_a_finished_job_releases_its_input_but_keeps_its_echo(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        media = b"large-binary-reference" * 10000
        job_id = queue.submit(plan(payload=SimpleNamespace(seed=1, steps=1,
                                                           media=media)))["id"]
        wait_until(lambda: queue.status(job_id)["status"] == "completed")
        assert queue.status(job_id)["resolved"]["frames"] == 9
        assert queue._jobs[job_id].payload is None
        assert queue.content(job_id)[1].exists()
    finally:
        queue.close()


def test_exit_clears_managed_files_but_preserves_other_files(tmp_path):
    cache = tmp_path / "job_cache"
    cache.mkdir()
    unrelated = cache / "important.png"
    unrelated.write_bytes(b"user file")
    stale = cache / ("job_" + "0" * 32 + ".bin")
    stale.write_bytes(b"old output")

    queue = JobQueue(cache)
    try:
        sync = queue.render_sync(plan(seed=21))
        job_id = queue.submit(plan(seed=22))["id"]
        wait_until(lambda: queue.status(job_id)["status"] == "completed")
        async_path = queue.content(job_id)[1]
        assert sync.exists() and async_path.exists() and stale.exists()
    finally:
        queue.close()
    assert not sync.exists() and not async_path.exists() and not stale.exists()
    assert unrelated.read_bytes() == b"user file"
    queue.close()  # repeated shutdown is harmless


def test_exit_waits_for_running_worker_before_cache_cleanup(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def render(payload, token, progress, path):
        entered.set()
        assert release.wait(3)
        path.write_bytes(b"partial")
        token.check()

    queue = JobQueue(tmp_path)
    job_id = queue.submit(plan(seed=99, render=render))["id"]
    assert entered.wait(2)
    closer = threading.Thread(target=queue.close)
    closer.start()
    try:
        wait_until(lambda: queue._jobs[job_id].token.requested)
        assert closer.is_alive(), "shutdown must wait for the worker"
    finally:
        release.set()
        closer.join(timeout=4)
    assert not closer.is_alive()
    assert queue.status(job_id)["status"] == "canceled"
    assert not list(tmp_path.glob("job_*.bin"))


def test_submitting_after_shutdown_is_refused(tmp_path):
    queue = JobQueue(tmp_path)
    queue.close()
    with pytest.raises(RuntimeError, match="shutting down"):
        queue.submit(plan())
    with pytest.raises(RuntimeError, match="shutting down"):
        queue.render_sync(plan())


def test_render_sync_leaves_nothing_behind_when_the_render_fails(tmp_path):
    seen = []

    def render(payload, token, progress, path):
        seen.append(payload)
        path.write_bytes(b"partial")
        if len(seen) == 1:
            raise RuntimeError("encoder failed")
        path.write_bytes(b"complete")

    queue = JobQueue(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="encoder failed"):
            queue.render_sync(plan(render=render))
        assert not list(tmp_path.glob("sync_*"))
        output = queue.render_sync(plan(render=render))
        assert output.read_bytes() == b"complete"
        output.unlink()
    finally:
        queue.close()
