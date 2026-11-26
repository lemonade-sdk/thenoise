"""Video routes: the request body, and the two routes of the video medium."""
from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel

from ..jobs import JobPlan, JobQueue
from ..media_request import MediaRequest
from ..media_routes import install_media_routes
from ..utils.media import verified_media
from .spec import Guide, VideoSpec


class GuideRequest(BaseModel):
    """One reference condition as the wire carries it: a file, and the frame of the clip
    it is one of -- omitted when it is none.

    The file is a picture, a clip or a sound; the bytes are asked which, so a client is
    not asked to declare a kind it may get wrong.
    """

    media: str
    frame: Optional[int] = None


class VideoRequest(MediaRequest, VideoSpec):
    """The HTTP body: the video spec, plus what only the wire carries. Frames arrive
    base64-encoded where the spec wants their bytes.

    ``references`` is a list whose items are either a bare base64 file or
    ``{"media": ..., "frame": n}``: with a frame the file conditions one of the clip's
    frames, counted from its end when negative, and without one the clip is only asked to
    resemble it.
    """

    SPEC: ClassVar[type] = VideoSpec
    MEDIA_FIELDS: ClassVar[tuple[str, ...]] = ("image", "end_image")

    image: Optional[str] = None
    end_image: Optional[str] = None
    references: Optional[list[Union[str, GuideRequest]]] = None

    def decode_media(self) -> dict:
        decoded = super().decode_media()
        decoded["references"] = [
            Guide(media=verified_media(
                "references", item.media if isinstance(item, GuideRequest) else item),
                frame=item.frame if isinstance(item, GuideRequest) else None)
            for item in (self.references or [])]
        return decoded


def install_video_routes(app: FastAPI, queue: JobQueue,
                         planner: Callable[[VideoSpec], JobPlan],
                         model_name: str) -> None:
    install_media_routes(app, queue, planner, model_name, "video", VideoRequest)


__all__ = ["VideoRequest", "install_video_routes"]
