"""Video routes: the request body, and the two routes of the video medium."""
from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel

from ..jobs import JobPlan, JobQueue
from ..media_request import MediaRequest
from ..media_routes import install_media_routes
from ..utils.media import verified_picture
from .spec import Guide, VideoSpec


class GuideRequest(BaseModel):
    """One reference condition as the wire carries it: a picture, and its frame."""

    image: str
    frame: int = 0


class VideoRequest(MediaRequest, VideoSpec):
    """The HTTP body: the video spec, plus what only the wire carries. Frames arrive
    base64-encoded where the spec wants their bytes.

    ``references`` is a list whose items are either a bare base64 picture or
    ``{"image": ..., "frame": n}``, that frame being the one the reference belongs to and
    counted from the clip's end when negative.
    """

    SPEC: ClassVar[type] = VideoSpec
    MEDIA_FIELDS: ClassVar[tuple[str, ...]] = ("image", "end_image")

    image: Optional[str] = None
    end_image: Optional[str] = None
    references: Optional[list[Union[str, GuideRequest]]] = None

    def decode_media(self) -> dict:
        decoded = super().decode_media()
        decoded["references"] = [
            Guide(pixels=verified_picture(
                "references", item.image if isinstance(item, GuideRequest) else item),
                frame=item.frame if isinstance(item, GuideRequest) else 0)
            for item in (self.references or [])]
        return decoded


def install_video_routes(app: FastAPI, queue: JobQueue,
                         planner: Callable[[VideoSpec], JobPlan],
                         model_name: str) -> None:
    install_media_routes(app, queue, planner, model_name, "video", VideoRequest)


__all__ = ["VideoRequest", "install_video_routes"]
