"""Focused HTTP API. A single generic /text2image endpoint serves whichever model
the runtime currently holds (the runtime loads exactly one model at a time).

Synchronous request/response: each generate() call blocks until the image is ready.
A per-model inference lock serializes concurrent requests.

The request carries only the shared, model-agnostic parameters. Per-model defaults
(including the "advanced" sampler params) are owned by the model class and are NOT
exposed here.
"""
from __future__ import annotations

import io
import logging
import os
import base64
from typing import List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from PIL import Image

from .runtime import NotLoadedError

logger = logging.getLogger(__name__)

_UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


class Text2ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    upscale: bool = False
    upscale_factor: float = 1.0
    upscale_type: str = "refined"
    sampler: Optional[str] = None
    qwen_vae_enhance: bool = False
    film_grain: float = 0.0
    sharpening: float = 0.0
    lora_specs: Optional[List[str]] = None  # ["filename.safetensors:0.8", ...]
    out: Literal["png", "json"] = "png"


def create_app(runtime) -> FastAPI:
    app = FastAPI(title="thenoise", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def ui():
        with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.get("/health")
    def health():
        return {"status": "ok", "models": runtime.available()}

    @app.get("/lora")
    def loras():
        """List available LoRA names (short, no .safetensors suffix)."""
        try:
            model = runtime.model
        except NotLoadedError:
            return Response(status_code=503, content="no model is loaded")
        return {"loras": model.list_loras()}

    @app.post("/text2image")
    def text2image(req: Text2ImageRequest):
        try:
            model = runtime.model
        except NotLoadedError:
            return Response(status_code=503, content="no model is loaded")
        try:
            image = model.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                upscale=req.upscale,
                upscale_factor=req.upscale_factor,
                upscale_type=req.upscale_type,
                sampler=req.sampler,
                qwen_vae_enhance=req.qwen_vae_enhance,
                film_grain=req.film_grain,
                sharpening=req.sharpening,
                lora_specs=req.lora_specs,
            )
        except Exception as e:  # surface generation errors cleanly
            logger.exception("generation failed")
            return Response(status_code=500, content=f"generation failed: {e}")

        buf = io.BytesIO()
        image.save(buf, format="PNG", pnginfo=getattr(image, "_pnginfo", None))
        content = buf.getvalue()

        if req.out == "json":
            return {"b64_json": base64.b64encode(content).decode("ascii")}
        return Response(
            content=content,
            media_type="image/png",
        )

    return app
