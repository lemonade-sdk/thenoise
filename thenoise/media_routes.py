"""The write routes of a queued medium: the file at once, and the job."""

#: No `from __future__ import annotations` here: FastAPI must see the live request class.

from typing import Callable

from fastapi import FastAPI

from .api_errors import error, guarded
from .jobs import JobPlan, JobQueue
from .jobs_http import content_response, submit_job


def install_media_routes(app: FastAPI, queue: JobQueue, planner: Callable[..., JobPlan],
                         model_name: str, medium: str, request_model: type) -> None:
    """Mount ``/{medium}`` and ``/jobs/{medium}``: one plan, so both refuse a request alike."""
    def checked_plan(req) -> JobPlan:
        if req.model and req.model != model_name:
            raise ValueError(f"model {req.model!r} is not loaded")
        return planner(req.to_spec())

    @app.post(f"/{medium}")
    def generate(req: request_model):
        """Return the rendered file directly, using the same renderer as the medium's jobs."""
        return guarded(f"{medium} generation", lambda: queue.render_sync(checked_plan(req)),
                       content_response, envelope=error)

    @app.post(f"/jobs/{medium}", status_code=202)
    def create_job(req: request_model):
        return submit_job(queue, f"{medium} job", lambda: checked_plan(req))


__all__ = ["install_media_routes"]
