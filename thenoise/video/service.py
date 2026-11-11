"""Expose a loaded video model through asynchronous video jobs."""
from __future__ import annotations

from pathlib import Path

from .jobs import VideoJobs


def create_video_service(model, *, cache_dir: str) -> tuple[VideoJobs, str]:
    jobs = VideoJobs(model.generate, Path(cache_dir), model=model.name,
                     resolve=model.resolve_video)
    return jobs, model.name
