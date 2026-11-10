"""Lemonade PR #3506-compatible video job routes for TheNoise standalone HTTP."""
from __future__ import annotations

import base64
import binascii
import io
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .jobs import VideoJobs
from .spec import VideoSpec


class VideoRequest(BaseModel):
    # Unknown fields must NOT be silently discarded (e.g. an unsupported LoRA).
    model_config = ConfigDict(extra="forbid")
    model: Optional[str] = None
    prompt: str
    negative_prompt: str = ""
    width: int = 1280
    height: int = 704
    video_frames: Optional[int] = Field(default=None, ge=1)
    duration: Optional[float] = Field(default=None, gt=0)
    fps: int = Field(default=24, ge=1, le=60)
    steps: int = Field(default=30, ge=1, le=100)
    cfg_scale: float = 5.0
    seed: Optional[int] = None
    output_format: Literal["mp4", "webm"] = "webm"
    sampling_method: Literal["unipc", "euler"] = "unipc"
    flow_shift: float = 5.0
    image: Optional[str] = None  # base64-encoded PNG/JPEG initial frame for I2V

    def to_spec(self) -> VideoSpec:
        data = self.model_dump(exclude={"model", "image"})
        if self.image is not None:
            # Bound encoded allocation *before* base64 decode.
            if len(self.image) > 20 * 1024 * 1024:
                raise ValueError("image exceeds 15 MiB")
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
                    if img.format not in ("PNG", "JPEG") or img.width * img.height > 25_000_000:
                        raise ValueError("image must be PNG/JPEG and at most 25 megapixels")
                    img.verify()
            except (OSError, Image.DecompressionBombError) as e:
                raise ValueError("image must be a valid PNG/JPEG") from e
        return VideoSpec(**data)


def install_video_routes(app: FastAPI, jobs: VideoJobs, model_name: str) -> None:
    @app.post("/v1/videos", status_code=202)
    def create_video(req: VideoRequest):
        if req.model and req.model != model_name:
            return JSONResponse({"error": f"model {req.model!r} is not loaded"}, status_code=400)
        try:
            spec = req.to_spec().resolve()
            snapshot = jobs.submit(spec)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except OverflowError as e:
            return JSONResponse({"error": str(e)}, status_code=429)
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        return {key: snapshot[key] for key in ("id", "status", "model", "created_at")}

    @app.get("/v1/videos/{video_id}")
    def get_video(video_id: str):
        result = jobs.status(video_id)
        if result is None:
            return JSONResponse({"error": "unknown video"}, status_code=404)
        return result

    @app.get("/v1/videos/{video_id}/content")
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

    @app.post("/v1/videos/{video_id}/cancel")
    def cancel_video(video_id: str):
        if not jobs.cancel(video_id):
            return JSONResponse({"error": "video not found or not cancellable"}, status_code=409)
        return {"status": "canceling"}

