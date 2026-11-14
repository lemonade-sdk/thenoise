"""Asynchronous video job routes."""
from __future__ import annotations

import base64
import binascii
import io
import logging
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict

from ..api_errors import error, guarded, json_error
from ..utils.lora_roles import LoraSpecs
from .jobs import VideoJobs
from .spec import VideoSpec

logger = logging.getLogger(__name__)


class VideoRequest(BaseModel):
    # Unknown fields must NOT be silently discarded (e.g. an unsupported LoRA).
    model_config = ConfigDict(extra="forbid")
    model: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frames: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    output_format: Optional[Literal["mp4", "webm"]] = None
    sampler: Optional[str] = None
    # Either the specs for a model's single LoRA-able component, or them by role:
    # {"dit": [...], "dit_low": [...]} on a two-expert model.
    lora_specs: LoraSpecs = None
    image: Optional[str] = None  # base64-encoded PNG/JPEG initial frame for I2V

    def to_spec(self) -> VideoSpec:
        data = self.model_dump(exclude={"model", "image"})
        if self.image is not None:
            try:
                data["image"] = base64.b64decode(self.image, validate=True)
            except (binascii.Error, ValueError) as e:
                raise ValueError("image must be valid base64 bytes") from e
            if not data["image"]:
                raise ValueError("image must not be empty")
            # Validate the input image before accepting a multi-minute job. PIL's
            # dimension check happens before the image is fully decompressed.
            from PIL import Image
            try:
                with Image.open(io.BytesIO(data["image"])) as img:
                    if img.format not in ("PNG", "JPEG"):
                        raise ValueError("image must be PNG or JPEG")
                    img.verify()
            except (OSError, Image.DecompressionBombError) as e:
                raise ValueError("image must be a valid PNG/JPEG") from e
        return VideoSpec(**data)


def install_video_routes(app: FastAPI, jobs: VideoJobs, model_name: str) -> None:
    def checked_spec(req: VideoRequest) -> VideoSpec:
        if req.model and req.model != model_name:
            raise ValueError(f"model {req.model!r} is not loaded")
        return req.to_spec()

    def video_file(path):
        media = "video/mp4" if path.suffix == ".mp4" else "video/webm"
        # The file exists to be handed out once, so it goes with the response.
        return FileResponse(path, media_type=media,
                            background=BackgroundTask(path.unlink, missing_ok=True))

    @app.post("/video")
    def generate_video(req: VideoRequest):
        """Return the generated video directly, using the same renderer as video jobs."""
        return guarded("video generation",
                       lambda: jobs.render_sync(checked_spec(req)), video_file,
                       envelope=error)

    # Everything past here is the job API, which answers JSON -- refusals included.
    @app.post("/videos", status_code=202)
    def create_video(req: VideoRequest):
        # A full queue and a model that cannot take the job are neither the caller's
        # fault nor a fault worth a traceback.
        return guarded(
            "video job", lambda: jobs.submit(checked_spec(req)),
            lambda snapshot: {key: snapshot[key]
                              for key in ("id", "status", "model", "created_at")},
            envelope=json_error, statuses={OverflowError: 429, RuntimeError: 503})

    @app.get("/videos/{video_id}")
    def get_video(video_id: str):
        result = jobs.status(video_id)
        if result is None:
            return json_error(404, "unknown video")
        return result

    @app.get("/videos/{video_id}/content")
    def get_video_content(video_id: str):
        status, path = jobs.content(video_id)
        if status == "not_found":
            return json_error(404, "unknown video")
        if status != "completed":
            return json_error(409, f"video is {status}")
        if not path.is_file():
            return json_error(404, "video content missing")
        return video_file(path)

    @app.post("/videos/{video_id}/cancel")
    def cancel_video(video_id: str):
        if not jobs.cancel(video_id):
            return json_error(409, "video not found or not cancellable")
        return {"status": "canceling"}

