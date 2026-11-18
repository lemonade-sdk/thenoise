from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Medium:
    """What a medium runs through, and what it answers on the wire."""

    #: ``(model, pixel_upscalers)`` -> its controller.
    controller: Callable[[Any, Any], Any]
    #: ``(app, jobs, runtime)`` -> the job routes a queued server installs.
    routes: Callable[[Any, Any, Any], None]


def _image_controller(model, pixel_upscalers):
    from .pipeline import PipelineController

    return PipelineController(model, pixel_upscalers)


def _video_controller(model, pixel_upscalers):
    from .video.controller import VideoPipelineController

    return VideoPipelineController(model)


def _image_routes(app, jobs, runtime):
    from .api import install_image_job_routes

    install_image_job_routes(app, jobs, runtime)


def _video_routes(app, jobs, runtime):
    from .video.http import install_video_routes
    from .video.service import video_planner

    install_video_routes(app, jobs, video_planner(runtime), runtime.model_name)


#: Imports are inside the factories, so an image model never imports ``thenoise.video``.
MEDIUMS: dict[str, Medium] = {
    "image": Medium(controller=_image_controller, routes=_image_routes),
    "video": Medium(controller=_video_controller, routes=_video_routes),
}

#: What an unloaded runtime answers as, so the queue refuses a submit, not the router.
DEFAULT_MEDIUM = "image"


def controller_for(model, pixel_upscalers):
    """The controller of a loaded model's medium."""
    return _entry(model.MEDIUM).controller(model, pixel_upscalers)


def install_routes(app, jobs, runtime) -> None:
    """The job routes of the loaded model's medium."""
    _entry(runtime.medium or DEFAULT_MEDIUM).routes(app, jobs, runtime)


def _entry(medium: str) -> Medium:
    try:
        return MEDIUMS[medium]
    except KeyError:
        raise ValueError(f"no medium {medium!r} is registered "
                         f"(known: {', '.join(sorted(MEDIUMS))})") from None


__all__ = ["DEFAULT_MEDIUM", "MEDIUMS", "Medium", "controller_for", "install_routes"]
