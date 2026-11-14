"""Expose the loaded video model through asynchronous video jobs."""
from __future__ import annotations

from pathlib import Path

from .jobs import VideoJobs


def create_video_service(runtime, *, cache_dir: str) -> tuple[VideoJobs, str]:
    """Jobs bound to the runtime rather than to an adapter instance.

    A reload swaps the model under a running server, so a job asks the runtime for the
    controller when it runs — and one whose model reports itself unfit is rebuilt and
    retried by ``Runtime.run`` instead of being reloaded inside the run.

    ``cache_dir`` is the server's whole cache directory; clips get a ``videos`` subdirectory
    of it, which is this service's business rather than the caller's.
    """
    def render(spec, token, progress, output):
        runtime.run(lambda controller: controller.render(spec, token, progress, output))

    def resolve(spec):
        controller = runtime.pipeline
        if controller is None:
            raise RuntimeError("no video model is loaded")
        return controller.resolve(spec)

    jobs = VideoJobs(render, Path(cache_dir) / "videos", model=runtime.model_name,
                     resolve=resolve)
    return jobs, runtime.model_name
