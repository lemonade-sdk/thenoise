"""Focused HTTP API. A single generic /text2image endpoint serves whichever model
the runtime currently holds (the runtime loads exactly one model at a time).

Synchronous request/response: each generate() call blocks until the image is ready.
A per-model inference lock serializes concurrent requests.

The request carries only the shared, model-agnostic parameters. Per-model defaults
(including the "advanced" sampler params) are owned by the model class and are NOT
exposed here.

Gallery support
---------------
When ``gallery_dir`` is set (via ``--gallery``), every generated image is saved
to that directory and served via ``/gallery-images/<filename>``.
The ``/gallery`` endpoint returns a list of images for the persistent gallery,
which is scanned at startup so images survive restarts.
"""
from __future__ import annotations

import io
import json
import logging
import os
import random
import base64
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


def _parse_gen_data(pnginfo: Any) -> dict | None:
    """Extract generation_data JSON from a PIL ``PngInfo`` metadata object or ``info`` dict."""
    if pnginfo is None:
        return None
    try:
        # Handle PIL PngInfo object (from build_pnginfo / _pnginfo)
        if hasattr(pnginfo, "chunks"):
            for ctype, data, _comp in pnginfo.chunks:
                if ctype != b"tEXt":
                    continue
                null_idx = data.find(b"\x00")
                if null_idx < 0:
                    continue
                keyword = data[:null_idx]
                if keyword == b"generation_data":
                    raw = data[null_idx + 1:]
                    return json.loads(raw.decode("latin-1", errors="replace"))
        # Handle .info dict (from Image.open().info)
        raw = getattr(pnginfo, "get", lambda _: None)("generation_data")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return json.loads(raw.decode("latin-1", errors="replace"))
        if isinstance(raw, str):
            return json.loads(raw)
        return None
    except Exception:
        return None


class Gallery:
    """Thread-safe file-based gallery backed by metadata in saved PNGs.

    Images are saved with the prompt metadata embedded in the PNG (``tEXt``
    chunks) and kept in an in-memory cache for fast ``/gallery`` responses.
    """

    def __init__(self, gallery_dir: str) -> None:
        self.gallery_dir = gallery_dir
        self._cache: dict[str, dict] = {}

    @property
    def active(self) -> bool:
        return bool(self.gallery_dir)

    @staticmethod
    def _read_meta(gallery_dir: str, fname: str) -> dict | None:
        try:
            from PIL import Image
            img = Image.open(os.path.join(gallery_dir, fname))
            meta = _parse_gen_data(img.info)
            img.close()
            return meta
        except Exception:
            return None

    # -- initial load ---------------------------------------------------------

    def initialize(self) -> None:
        """Scan the gallery directory and extract metadata from existing images."""
        if not self.active:
            return
        try:
            filenames = sorted(
                f for f in os.listdir(self.gallery_dir)
                if f.lower().endswith(".png")
            )
        except FileNotFoundError:
            return
        for fname in filenames:
            meta = self._read_meta(self.gallery_dir, fname)
            if meta:
                self._cache[fname] = meta

    # -- save -----------------------------------------------------------------

    def save(self, image: Any) -> str | None:
        """Save *image* to the gallery and cache its metadata.

        Returns the filename (for ``currentUrl``) or ``None`` on failure.
        """
        if not self.active:
            return None
        pnginfo = getattr(image, "_pnginfo", None)
        meta = _parse_gen_data(pnginfo) if pnginfo else None
        if not self.gallery_dir:
            return None

        # Ensure directory exists or create it
        gallery_dir = os.path.abspath(self.gallery_dir)
        try:
            os.makedirs(gallery_dir, exist_ok=True)
        except OSError:
            logger.exception("gallery: cannot create directory %s", gallery_dir)
            return None
        for _ in range(5):
            if meta:
                fname = (
                    f"{datetime.now(timezone.utc).timestamp():010.3f}_"
                    f"{random.randint(0, 9999):04d}_"
                    f"{meta.get('seed', 'x')}.png"
                )
            else:
                fname = (
                    f"{datetime.now(timezone.utc).timestamp():010.3f}_"
                    f"{random.randint(0, 9999):04d}.png"
                )
            dest = os.path.join(gallery_dir, fname)
            if not os.path.exists(dest):
                break
        else:
            return None
        try:
            image.save(dest, format="PNG", pnginfo=pnginfo)
            if meta:
                self._cache[fname] = meta
            return fname
        except Exception:
            logger.exception("gallery: failed to save %s", fname)
            return None

    # -- list ------------------------------------------------------------------

    def list_images(self) -> list[dict]:
        """Return list of {filename, url, meta} dicts, newest first."""
        if not self.active:
            return []
        now = set(self._cache.keys())
        try:
            on_disk = frozenset(
                f for f in os.listdir(self.gallery_dir)
                if f.lower().endswith(".png")
            )
        except FileNotFoundError:
            on_disk = frozenset()
        stale = now - on_disk
        if stale:
            for s in stale:
                self._cache.pop(s, None)
        fresh = on_disk - now
        if fresh:
            for f in list(fresh)[:50]:
                m = self._read_meta(self.gallery_dir, f)
                if m:
                    self._cache[f] = m
        order = sorted(self._cache, reverse=True)
        return [
            {"filename": f, "url": f"/gallery-images/{f}", "meta": self._cache[f]}
            for f in order
        ]

    # -- image serving ---------------------------------------------------------

    def serve(self, filename: str) -> Response | JSONResponse:
        """Return the raw PNG for *filename*, or a 404."""
        if not self.active:
            return JSONResponse(status_code=404, content="gallery not configured")
        safe = os.path.basename(filename)
        gallery_abs = os.path.abspath(self.gallery_dir)
        path = os.path.join(gallery_abs, safe)
        abs_path = os.path.abspath(path)
        if not (abs_path.startswith(gallery_abs) and (
            abs_path == gallery_abs or abs_path.startswith(gallery_abs + os.sep)
        )):
            return JSONResponse(status_code=404, content="not found")
        if not os.path.isfile(path):
            return JSONResponse(status_code=404, content="not found")
        try:
            data = open(path, "rb").read()
            return Response(content=data, media_type="image/png")
        except OSError:
            return JSONResponse(status_code=404, content="not found")


# ---------------------------------------------------------------------------
# Request model & app factory

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


def create_app(runtime, gallery: Gallery | None = None) -> FastAPI:
    app = FastAPI(title="thenoise", version="0.1.0")

    if gallery is not None and gallery.active:
        gallery.initialize()
        logger.info("gallery initialized with %d images from %s", len(gallery._cache), gallery.gallery_dir)

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
        pipeline = runtime.pipeline
        if pipeline is None:
            return Response(status_code=503, content="no model is loaded")
        return {"loras": pipeline.list_loras()}

    @app.get("/upscalers")
    def upscalers():
        """List available pixel upscaler names (short, no .safetensors suffix).

        Pixel upscalers are a pixel-domain / server concern and need no diffusion
        model, so this works even when no model is loaded.
        """
        return {"upscalers": runtime.pixel_upscalers.list()}

    @app.get("/gallery")
    def gallery_list():
        """Return a list of saved gallery images with their metadata."""
        if gallery is None or not gallery.active:
            return []
        return gallery.list_images()

    @app.get("/gallery-images/{filename:path}")
    def gallery_image(filename: str):
        """Serve a specific gallery image."""
        if gallery is None:
            return JSONResponse(status_code=404, content="gallery not configured")
        return gallery.serve(filename)

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

        if gallery is not None:
            fname = gallery.save(image)
            if fname:
                logger.info("gallery: saved %s", fname)
            else:
                logger.warning("gallery: save returned None for image")

        if req.out == "json":
            return {"b64_json": base64.b64encode(content).decode("ascii")}
        return Response(
            content=content,
            media_type="image/png",
        )

    return app
