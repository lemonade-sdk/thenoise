"""Video routes: the request body, and the two routes of the video medium."""
from __future__ import annotations

from typing import Callable, ClassVar, Optional

from fastapi import FastAPI

from ..jobs import JobPlan, JobQueue
from ..media_request import MediaRequest
from ..media_routes import install_media_routes
from .spec import VideoSpec


class VideoRequest(MediaRequest, VideoSpec):
    """The HTTP body: the video spec, plus what only the wire carries. Both frames arrive
    base64-encoded where the spec wants their bytes."""

    SPEC: ClassVar[type] = VideoSpec
    MEDIA_FIELDS: ClassVar[tuple[str, ...]] = ("image", "end_image")

    image: Optional[str] = None
    end_image: Optional[str] = None


def install_video_routes(app: FastAPI, queue: JobQueue,
                         planner: Callable[[VideoSpec], JobPlan],
                         model_name: str) -> None:
    install_media_routes(app, queue, planner, model_name, "video", VideoRequest)


__all__ = ["VideoRequest", "install_video_routes"]
