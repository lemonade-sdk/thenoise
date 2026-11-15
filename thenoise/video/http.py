"""Video routes: the clip served directly, and the video submit of the job API."""
from __future__ import annotations

import base64
import binascii
import io
from typing import Callable, Optional

from fastapi import FastAPI

from ..api_errors import error, guarded
from ..jobs import JobPlan, JobQueue
from ..jobs_http import content_response, submit_job
from .spec import VideoSpec


class VideoRequest(VideoSpec):
    """The HTTP body: the video spec, plus what only the wire carries.

    ``model`` is checked against what the server actually loaded, and both frames arrive
    base64-encoded where the spec wants their bytes.
    """

    model: Optional[str] = None
    image: Optional[str] = None
    end_image: Optional[str] = None

    def to_spec(self) -> VideoSpec:
        data = self.model_dump(exclude={"model", "image", "end_image"})
        for name in ("image", "end_image"):
            encoded = getattr(self, name)
            if encoded is not None:
                data[name] = _frame(name, encoded)
        return VideoSpec(**data)


def _frame(field: str, encoded: str) -> bytes:
    """One base64 image from the wire, checked before a multi-minute job is accepted; PIL's
    dimension check happens before the image is fully decompressed."""
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"{field} must be valid base64 bytes") from e
    if not data:
        raise ValueError(f"{field} must not be empty")
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ("PNG", "JPEG"):
                raise ValueError(f"{field} must be PNG or JPEG")
            img.verify()
    except (OSError, Image.DecompressionBombError) as e:
        raise ValueError(f"{field} must be a valid PNG/JPEG") from e
    return data


def install_video_routes(app: FastAPI, queue: JobQueue, planner: Callable[[VideoSpec], JobPlan],
                         model_name: str) -> None:
    def checked_plan(req: VideoRequest) -> JobPlan:
        if req.model and req.model != model_name:
            raise ValueError(f"model {req.model!r} is not loaded")
        return planner(req.to_spec())

    @app.post("/video")
    def generate_video(req: VideoRequest):
        """Return the generated video directly, using the same renderer as video jobs."""
        return guarded("video generation", lambda: queue.render_sync(checked_plan(req)),
                       content_response, envelope=error)

    # The job API answers JSON -- refusals included -- and its reads are the shared
    # `/jobs/{id}` family, not video's own.
    @app.post("/jobs/video", status_code=202)
    def create_video_job(req: VideoRequest):
        return submit_job(queue, "video job", lambda: checked_plan(req))
