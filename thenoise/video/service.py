"""Turn a video request into a job the queue can run against the loaded model."""
from __future__ import annotations

from typing import Callable

from ..jobs import JobPlan
from .spec import VideoSpec


def video_planner(runtime) -> Callable[[VideoSpec], JobPlan]:
    """Plan video jobs against the runtime rather than against an adapter instance.

    A reload swaps the model under a running server, so a job asks the runtime for the
    controller when it runs — and one whose model reports itself unfit is rebuilt and
    retried by ``Runtime.run`` instead of being reloaded inside the run.
    """
    def render(spec, token, progress, output):
        runtime.run(lambda controller: controller.render(spec, token, progress, output))

    def plan(spec: VideoSpec) -> JobPlan:
        controller = runtime.pipeline
        if controller is None:
            raise RuntimeError("no video model is loaded")
        resolved = controller.resolve(spec)
        return JobPlan(kind="video", model=runtime.model_name, payload=resolved,
                       public=resolved.public(), suffix=f".{resolved.output_format}",
                       render=render)

    return plan
