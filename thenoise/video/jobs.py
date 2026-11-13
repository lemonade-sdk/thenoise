"""Bounded, single-worker video jobs with cooperative, acknowledged cancellation.

The inference adapter receives a Cancellation object and a progress callback. It
must check the token in its denoising loop and after each potentially blocking
stage. This module intentionally has no FastAPI, torch or codec dependency.
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .spec import ResolvedVideo, VideoSpec

log = logging.getLogger("uvicorn.error")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VideoCancelled(Exception):
    """Expected cooperative interruption, not an inference failure."""


class Cancellation:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def check(self) -> None:
        if self._event.is_set():
            raise VideoCancelled()

    @property
    def requested(self) -> bool:
        return self._event.is_set()


@dataclass
class _Job:
    id: str
    model: str
    spec: Optional[ResolvedVideo]
    target: Path
    resolved: dict = field(init=False)
    output_format: str = field(init=False)
    token: Cancellation = field(default_factory=Cancellation)
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    stage: str = "queued"
    step: int = 0
    of: int = 0

    def __post_init__(self) -> None:
        self.resolved = self.spec.public()
        self.output_format = self.spec.output_format

    def snapshot(self) -> dict:
        record = {
            "id": self.id, "status": self.status, "model": self.model,
            "created_at": self.created_at, "resolved": dict(self.resolved),
            "progress": {"stage": self.stage, "step": self.step, "of": self.of},
        }
        for key in ("started_at", "completed_at", "error"):
            value = getattr(self, key)
            if value is not None:
                record[key] = value
        if self.status == "completed":
            record.update({
                "mime_type": "video/mp4" if self.output_format == "mp4" else "video/webm",
                **self.resolved,
            })
        return record


# renderer(spec, cancellation, progress(stage, step, total), destination)
Renderer = Callable[[ResolvedVideo, Cancellation, Callable[[str, int, int], None], Path], None]


class VideoJobs:
    def __init__(self, renderer: Renderer, root: Path, *, model: str,
                 queue_limit: int = 4, history_limit: int = 32,
                 resolve: Callable[[VideoSpec], ResolvedVideo] | None = None):
        self._render = renderer
        self._resolve = resolve
        self._root = Path(root)
        self._model = model
        self._queue_limit = queue_limit
        self._history_limit = history_limit
        self._lock = threading.Condition()
        self._jobs: dict[str, _Job] = {}
        self._queued: deque[str] = deque()
        self._worker: Optional[threading.Thread] = None
        self._closed = False

    def _resolve_spec(self, spec: VideoSpec | ResolvedVideo) -> ResolvedVideo:
        if isinstance(spec, VideoSpec):
            if self._resolve is None:
                raise ValueError("a video model resolver is required for unresolved requests")
            return self._resolve(spec)
        return spec

    def render_sync(self, spec: VideoSpec | ResolvedVideo) -> Path:
        """Render directly; the HTTP response removes the private output on completion."""
        spec = self._resolve_spec(spec)
        with self._lock:
            if self._closed:
                raise RuntimeError("video service is shutting down")
        self._root.mkdir(parents=True, exist_ok=True)
        target = self._root / ("sync_" + uuid.uuid4().hex + "." + spec.output_format)
        try:
            self._render(spec, Cancellation(), lambda *_: None, target)
            return target
        except BaseException:
            target.unlink(missing_ok=True)
            raise

    def submit(self, spec: VideoSpec | ResolvedVideo) -> dict:
        spec = self._resolve_spec(spec)
        with self._lock:
            if self._closed:
                raise RuntimeError("video service is shutting down")
            if len(self._queued) >= self._queue_limit:
                raise OverflowError("video queue is full")
            self._prune()
            job_id = "vid_" + uuid.uuid4().hex
            job = _Job(job_id, self._model, spec,
                       self._root / (job_id + "." + spec.output_format))
            self._jobs[job_id] = job
            self._queued.append(job_id)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="thenoise-video",
                                                daemon=True)
                self._worker.start()
            self._lock.notify()
            log.info("Video %s queued (%s, %d frames, %dx%d)",
                     job.id, job.resolved["mode"], spec.frame_count, spec.width, spec.height)
            return job.snapshot()

    def _prune(self) -> None:
        # Never evict active jobs or their clips. Delete only terminal records.
        for key, job in list(self._jobs.items()):
            if len(self._jobs) < self._history_limit:
                break
            if job.status in ("completed", "failed", "canceled"):
                self._jobs.pop(key)
                job.target.unlink(missing_ok=True)

    def status(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.snapshot() if job else None

    def content(self, job_id: str) -> tuple[str, Optional[Path]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return "not_found", None
            if job.status != "completed":
                return job.status, None
            return job.status, job.target

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in ("queued", "running") or job.token.requested:
                return False
            job.token.cancel()
            if job.status == "queued":
                self._queued.remove(job_id)
                self._finish(job, "canceled")
            return True

    @staticmethod
    def _finish(job: _Job, status: str) -> None:
        job.status = status
        job.stage = status
        job.completed_at = _now()
        # Raw I2V inputs may be 15 MiB each; do not retain them for the whole
        # history window after the renderer (or a queued cancellation) ends.
        job.spec = None
        log.info("Video %s %s%s", job.id, status,
                 f" ({job.target})" if status == "completed" else "")

    def _work(self) -> None:
        while True:
            with self._lock:
                self._lock.wait_for(lambda: self._closed or bool(self._queued))
                if self._closed and not self._queued:
                    return
                job = self._jobs[self._queued.popleft()]
                if job.token.requested:
                    self._finish(job, "canceled")
                    continue
                job.status = "running"
                job.started_at = _now()
                job.stage = "loading"
            log.info("Video %s started: loading model", job.id)

            def progress(stage: str, step: int = 0, total: int = 0) -> None:
                job.token.check()
                with self._lock:
                    changed = job.stage != stage
                    job.stage, job.step, job.of = stage, step, total
                if changed:
                    log.info("Video %s: %s", job.id, stage)

            try:
                job.token.check()
                self._root.mkdir(parents=True, exist_ok=True)
                self._render(job.spec, job.token, progress, job.target)
                job.token.check()
            except VideoCancelled:
                with self._lock:
                    self._finish(job, "canceled")
            except Exception as e:
                with self._lock:
                    if job.token.requested:
                        self._finish(job, "canceled")
                    else:
                        log.exception("video %s failed", job.id)
                        job.error = f"{type(e).__name__}: {e}"
                        self._finish(job, "failed")
            else:
                with self._lock:
                    self._finish(job, "completed")
            finally:
                if job.status != "completed":
                    job.target.unlink(missing_ok=True)

    def close(self) -> None:
        """Request cancellation; the worker acknowledges it at the next safe point."""
        with self._lock:
            self._closed = True
            for job in self._jobs.values():
                if job.status in ("queued", "running"):
                    job.token.cancel()
                    if job.status == "queued":
                        self._finish(job, "canceled")
            self._queued.clear()
            self._lock.notify_all()
