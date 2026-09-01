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
import mimetypes
import os
import base64
from typing import List, Literal, Optional, Union

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from PIL import Image

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
    pixel_upscaler: Optional[str] = None  # name (no .safetensors) in upscaler_dir
    out: Literal["png", "json"] = "png"

    def to_request(self):
        """Convert this wire request into a ``GenerateRequest`` for the controller."""
        from .models.config import GenerateRequest

        return GenerateRequest(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            steps=self.steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
            upscale=self.upscale,
            upscale_factor=self.upscale_factor,
            upscale_type=self.upscale_type,
            sampler=self.sampler,
            qwen_vae_enhance=self.qwen_vae_enhance,
            film_grain=self.film_grain,
            sharpening=self.sharpening,
            lora_specs=self.lora_specs,
            pixel_upscaler=self.pixel_upscaler,
        )


class UpscaleRequest(BaseModel):
    image_b64: str  # base64-encoded input image bytes (PNG/JPEG)
    upscale_factor: float = 0.0  # desired final factor; 0.0 = detected native scale
    pixel_upscaler: str  # name (no .safetensors) in upscaler_dir
    out: Literal["png", "json"] = "png"


class EditRequest(Text2ImageRequest):
    """Instruction-based editing: image(s) + prompt -> edited image.

    ``image`` accepts one or more base64-encoded images (OpenAI-style).
    """

    image: Union[str, List[str]]

    def to_edit_request(self):
        """Convert into a ``GenerateRequest`` carrying decoded PIL images."""
        from .models.config import GenerateRequest

        b64_list = self.image if isinstance(self.image, list) else [self.image]
        images = [
            Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
            for b in b64_list
        ]
        req: GenerateRequest = self.to_request()
        # OpenAI-style: ``image`` is one or more images; store single or list.
        req.image = images[0] if len(images) == 1 else images
        return req


def create_app(runtime) -> FastAPI:
    app = FastAPI(title="thenoise", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def ui():
        with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.get("/static/{filename:path}", response_class=Response)
    def static_file(filename: str):
        """Serve any static UI file (CSS/JS/etc.) from the UI directory."""
        base = os.path.realpath(_UI_DIR)
        path = os.path.realpath(os.path.join(base, filename))
        if not path.startswith(base + os.sep):
            return Response(status_code=403, content="forbidden")
        try:
            with open(path, "rb") as f:
                content = f.read()
        except OSError:
            return Response(status_code=404, content="not found")
        media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return Response(content=content, media_type=media_type)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "models": runtime.available(),
            "capabilities": runtime.model_capabilities(),
        }

    @app.get("/lora")
    def loras():
        """List available LoRA names (short, no .safetensors suffix)."""
        pipeline = runtime.pipeline
        if pipeline is None:
            return Response(status_code=503, content="no model is loaded")
        return {"loras": pipeline.list_loras()}

    @app.get("/upscalers")
    def upscalers():
        """List available pixel upscaler names + their detected native scales.

        Pixel upscalers are a pixel-space / server concern and need no diffusion
        model, so this works even when no model is loaded. ``scales`` maps each
        name to its detected native scale (2 or 4); unknown/undetectable entries
        are 0.
        """
        names = runtime.pixel_upscalers.list()
        scales = {}
        for name in names:
            try:
                scales[name] = runtime.pixel_upscalers.scale(name)
            except Exception:
                logger.exception("failed to detect scale for upscaler %r", name)
                scales[name] = 0
        return {"upscalers": names, "scales": scales}

    @app.post("/text2image")
    def text2image(req: Text2ImageRequest):
        pipeline = runtime.pipeline
        if pipeline is None:
            return Response(status_code=503, content="no model is loaded")
        try:
            image = pipeline.generate(req.to_request())
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

    @app.post("/edit")
    def edit(req: EditRequest):
        """Reference-latent instruction editing (image + prompt -> edited image)."""
        pipeline = runtime.pipeline
        if pipeline is None:
            return Response(status_code=503, content="no model is loaded")
        try:
            image = pipeline.edit(req.to_edit_request())
        except ValueError as e:  # model doesn't support editing / bad request
            return Response(status_code=400, content=str(e))
        except Exception as e:  # surface edit errors cleanly
            logger.exception("edit failed")
            return Response(status_code=500, content=f"edit failed: {e}")

        buf = io.BytesIO()
        image.save(buf, format="PNG", pnginfo=getattr(image, "_pnginfo", None))
        content = buf.getvalue()

        if req.out == "json":
            return {"b64_json": base64.b64encode(content).decode("ascii")}
        return Response(
            content=content,
            media_type="image/png",
        )

    @app.post("/upscale")
    def upscale(req: UpscaleRequest):
        """Upscale an input image by ``upscale_factor``x with a named pixel upscaler.

        Pixel upscaling is a pixel-space / server concern and needs no diffusion
        model, so this works even when no model is loaded.
        """
        try:
            image = runtime.upscaler.upscale(
                Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB"),
                req.upscale_factor,
                req.pixel_upscaler,
            )
        except ValueError as e:
            return Response(status_code=400, content=str(e))
        except Exception as e:  # surface upscale errors cleanly
            logger.exception("upscale failed")
            return Response(status_code=500, content=f"upscale failed: {e}")

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
