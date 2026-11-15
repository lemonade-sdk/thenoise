"""The job API's HTTP shape: one submit route per medium, one read surface for all of them.

A submission answers 202 with the job it accepted; a request the loaded model cannot serve
is refused there and then, because resolving it is cheap. Everything past the accepted
answer is JSON, refusals included.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .api_errors import guarded, json_error
from .jobs import JobPlan, JobQueue, mime_type

#: A full queue and a model that cannot take the job are neither the caller's fault nor
#: a fault worth a traceback.
_REFUSALS = {OverflowError: 429, RuntimeError: 503}


def content_response(path: Path) -> FileResponse:
    """Hand a finished artifact out once — it exists to be downloaded, so it leaves with
    the response that carried it."""
    return FileResponse(path, media_type=mime_type(path),
                        background=BackgroundTask(path.unlink, missing_ok=True))


def accepted(snapshot: dict) -> dict:
    """What a caller needs to go poll: the job's name and nothing else."""
    return {key: snapshot[key]
            for key in ("id", "kind", "status", "model", "created_at")}


def submit_job(queue: JobQueue, failure: str, plan: Callable[[], JobPlan]):
    """Resolve and queue one request, answering 202 with the accepted job."""
    return guarded(failure, lambda: queue.submit(plan()), accepted,
                   envelope=json_error, statuses=_REFUSALS)


def install_job_routes(app: FastAPI, queue: JobQueue) -> None:
    """Every medium's reader: the list, status, content, cancel."""

    @app.get("/jobs")
    def list_jobs():
        return {"jobs": queue.listing()}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        result = queue.status(job_id)
        if result is None:
            return json_error(404, "unknown job")
        return result

    @app.get("/jobs/{job_id}/content")
    def get_job_content(job_id: str):
        status, path = queue.content(job_id)
        if status == "not_found":
            return json_error(404, "unknown job")
        if status != "completed":
            return json_error(409, f"job is {status}")
        if not path.is_file():
            return json_error(404, "job content missing")
        return content_response(path)

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if not queue.cancel(job_id):
            return json_error(409, "job not found or not cancellable")
        return {"status": "canceling"}
