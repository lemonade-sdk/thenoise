"""Asynchronous video job routes."""
from __future__ import annotations

import base64
import binascii
import io
import logging
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict

from ..utils.lora_roles import LoraSpecs
from .jobs import VideoJobs
from .spec import VideoSpec

log = logging.getLogger("uvicorn.error")


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

    @app.post("/video")
    def generate_video(req: VideoRequest):
        """Return the generated video directly, using the same renderer as video jobs."""
        try:
            path = jobs.render_sync(checked_spec(req))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception:
            log.exception("synchronous video generation failed")
            return JSONResponse({"error": "video generation failed"}, status_code=500)
        media = "video/mp4" if path.suffix == ".mp4" else "video/webm"
        return FileResponse(path, media_type=media,
                            background=BackgroundTask(path.unlink, missing_ok=True))

    @app.post("/videos", status_code=202)
    def create_video(req: VideoRequest):
        try:
            snapshot = jobs.submit(checked_spec(req))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except OverflowError as e:
            return JSONResponse({"error": str(e)}, status_code=429)
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        return {key: snapshot[key] for key in ("id", "status", "model", "created_at")}

    @app.get("/videos/{video_id}")
    def get_video(video_id: str):
        result = jobs.status(video_id)
        if result is None:
            return JSONResponse({"error": "unknown video"}, status_code=404)
        return result

    @app.get("/videos/{video_id}/content")
    def get_video_content(video_id: str):
        status, path = jobs.content(video_id)
        if status == "not_found":
            return JSONResponse({"error": "unknown video"}, status_code=404)
        if status != "completed":
            return JSONResponse({"error": f"video is {status}"}, status_code=409)
        if not path.is_file():
            return JSONResponse({"error": "video content missing"}, status_code=404)
        media = "video/mp4" if path.suffix == ".mp4" else "video/webm"
        return FileResponse(path, media_type=media)

    @app.post("/videos/{video_id}/cancel")
    def cancel_video(video_id: str):
        if not jobs.cancel(video_id):
            return JSONResponse({"error": "video not found or not cancellable"}, status_code=409)
        return {"status": "canceling"}

