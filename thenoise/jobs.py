"""Bounded, single-worker jobs with cooperative, acknowledged cancellation.

A job carries its own medium: the payload a renderer consumes, the renderer, the file the
result lands in and the resolved request the status route echoes back. Nothing here knows
what an image or a video is, and the module intentionally has no FastAPI, torch or codec
dependency.

The renderer receives a Cancellation and a progress callback. It must check the token in
its denoising loop and after each potentially blocking stage.
"""
from __future__ import annotations

import logging
import mimetypes
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

# Only delete generated cache files, never unrelated files beside them.
_CACHE_NAME = re.compile(r"(?:sync_|job_)[0-9a-f]{32}\.\w+\Z")

#: progress(stage, step, of)
Progress = Callable[[str, int, int], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def idle(_stage: str, _step: int = 0, _of: int = 0) -> None:
    """The progress sink of a caller nobody is polling."""


def mime_type(path: Path) -> str:
    """What a finished artifact is, from the suffix its plan asked for."""
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


class JobCancelled(Exception):
    """Expected cooperative interruption, not an inference failure."""


class Cancellation:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def check(self) -> None:
        if self._event.is_set():
            raise JobCancelled()

    @property
    def requested(self) -> bool:
        return self._event.is_set()


class Uncancellable(Cancellation):
    """The token of a request nobody can interrupt: every synchronous caller."""

    def cancel(self) -> None:
        pass

    def check(self) -> None:
        pass

    @property
    def requested(self) -> bool:
        return False


#: One shared instance: it has no state to keep apart, and nothing can cancel it.
UNCANCELLABLE = Uncancellable()


def stepping(token: Cancellation, progress: Progress, stage: str = "sampling",
             done: int = 0, of: int = 0):
    """The sampler's per-step hook: one progress tick, one cancellation check.

    ``done`` is the steps an earlier span already took and ``of`` the steps the whole
    schedule has, so a request cut into spans still shows one clock. Left alone, the
    solver's own count and length are used.
    """
    def step(x, i, total):
        progress(stage, done + i, of or total)
        token.check()
        return x

    return step


#: render(payload, cancellation, progress, destination)
Renderer = Callable[[Any, Cancellation, Progress, Path], None]


@dataclass(frozen=True)
class JobPlan:
    """What a medium asks the queue to do, and what its caller is told about it.

    ``public`` is the resolved request as the status route echoes it; ``payload`` is the
    renderer's input, dropped as soon as the job settles because a request that conditions
    on media carries that media in it.
    """

    kind: str
    model: str
    payload: Any
    public: Mapping[str, Any]
    #: Including the dot: ".png", ".mp4".
    suffix: str
    render: Renderer


@dataclass
class _Job:
    id: str
    kind: str
    model: str
    public: Mapping[str, Any]
    payload: Any
    render: Renderer
    target: Path
    token: Cancellation = field(default_factory=Cancellation)
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    stage: str = "queued"
    step: int = 0
    of: int = 0

    def snapshot(self) -> dict:
        record = {
            "id": self.id, "kind": self.kind, "status": self.status,
            "model": self.model, "created_at": self.created_at,
            "resolved": dict(self.public),
            "progress": {"stage": self.stage, "step": self.step, "of": self.of},
        }
        for key in ("started_at", "completed_at", "error"):
            value = getattr(self, key)
            if value is not None:
                record[key] = value
        if self.status == "completed":
            record["mime_type"] = mime_type(self.target)
        return record


class JobQueue:
    """One worker thread, so whatever was queued the engine still runs one request at a
    time — which is what the single resident model and the inference lock require anyway.
    """

    def __init__(self, root: Path, *, queue_limit: int = 4, history_limit: int = 32):
        self._root = Path(root)
        self._queue_limit = queue_limit
        self._history_limit = history_limit
        self._lock = threading.Condition()
        self._jobs: dict[str, _Job] = {}
        self._queued: deque[str] = deque()
        self._worker: Optional[threading.Thread] = None
        self._closed = False

    def render_sync(self, plan: JobPlan) -> Path:
        """Render directly, without queueing; the HTTP response removes the output."""
        with self._lock:
            if self._closed:
                raise RuntimeError("job service is shutting down")
        self._root.mkdir(parents=True, exist_ok=True)
        target = self._root / ("sync_" + uuid.uuid4().hex + plan.suffix)
        try:
            plan.render(plan.payload, UNCANCELLABLE, idle, target)
            return target
        except BaseException:
            target.unlink(missing_ok=True)
            raise

    def submit(self, plan: JobPlan) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("job service is shutting down")
            if len(self._queued) >= self._queue_limit:
                raise OverflowError("job queue is full")
            self._prune()
            job_id = "job_" + uuid.uuid4().hex
            job = _Job(job_id, plan.kind, plan.model, dict(plan.public),
                       plan.payload, plan.render,
                       self._root / (job_id + plan.suffix))
            self._jobs[job.id] = job
            self._queued.append(job.id)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="thenoise-jobs",
                                                daemon=True)
                self._worker.start()
            self._lock.notify()
            logger.info("%s job %s queued for model %r", job.kind, job.id, job.model)
            return job.snapshot()

    def _prune(self) -> None:
        # Never evict active jobs or their artifacts. Delete only terminal records.
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
        # A reference image or clip input can be megabytes; do not retain it for the
        # whole history window after the renderer (or a queued cancellation) ends.
        job.payload = None
        logger.info("%s job %s %s%s", job.kind, job.id, status,
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
            logger.info("%s job %s started", job.kind, job.id)

            def progress(stage: str, step: int = 0, of: int = 0) -> None:
                job.token.check()
                with self._lock:
                    changed = job.stage != stage
                    job.stage, job.step, job.of = stage, step, of
                if changed:
                    logger.info("%s job %s: %s", job.kind, job.id, stage)

            try:
                job.token.check()
                self._root.mkdir(parents=True, exist_ok=True)
                job.render(job.payload, job.token, progress, job.target)
                job.token.check()
            except JobCancelled:
                with self._lock:
                    self._finish(job, "canceled")
            except Exception as e:
                with self._lock:
                    if job.token.requested:
                        self._finish(job, "canceled")
                    else:
                        logger.exception("%s job %s failed", job.kind, job.id)
                        job.error = f"{type(e).__name__}: {e}"
                        self._finish(job, "failed")
            else:
                with self._lock:
                    self._finish(job, "completed")
            finally:
                if job.status != "completed":
                    job.target.unlink(missing_ok=True)

    def close(self) -> None:
        """Stop the worker, then remove managed artifacts from the cache."""
        with self._lock:
            self._closed = True
            for job in self._jobs.values():
                if job.status in ("queued", "running"):
                    job.token.cancel()
                    if job.status == "queued":
                        self._finish(job, "canceled")
            self._queued.clear()
            self._lock.notify_all()

        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join()
        if self._root.is_dir():
            for path in self._root.iterdir():
                if _CACHE_NAME.fullmatch(path.name) and path.is_file():
                    path.unlink(missing_ok=True)
