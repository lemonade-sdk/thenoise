"""Wire native Wan inference to asynchronous video jobs."""
from __future__ import annotations

from pathlib import Path

from .jobs import VideoJobs
from .wan import WanVideo


def create_video_service(model_dir: str, *, device: str, offload: bool,
                         cache_dir: str) -> tuple[VideoJobs, str]:
    model = WanVideo(model_dir, device=device, offload=offload)
    jobs = VideoJobs(model.generate, Path(cache_dir), model=model.name)
    return jobs, model.name
