"""Focused HTTP API: one endpoint per image job, synchronous, lock-serialized.
Requests carry model-agnostic fields only; per-model defaults live on the model class.
"""
from __future__ import annotations

import io
import logging
import mimetypes
import os
import base64
from typing import List, Literal, Optional, Union

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, create_model, model_validator

from thenoise.controlmap.options import (
    OPTIONS_BY_KIND,
    CannyOptions,
    DepthOptions,
    LineartOptions,
    PoseOptions,
    ControlMapOptions,
    SoftedgeOptions,
)
from thenoise.postprocess.options import (
    POSTPROCESS_NAMES,
    PostprocessOptions,
    flat_options,
)
from thenoise.utils.image_tensor import load_image
from thenoise.utils.options import options_schema, wire_fields
from thenoise.utils.png import pnginfo_of

logger = logging.getLogger(__name__)

_UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


def decode_image(encoded: str) -> "PIL.Image.Image":
    """Base64 wire bytes -> PIL image, keeping the alpha when the input carries one.
    Not base64, or an image PIL cannot read, raises the ``ValueError`` routes map to 400.
    """
    return load_image(io.BytesIO(base64.b64decode(encoded)))


class _Text2ImageBody(BaseModel):
    """Everything a generation asks for, except the post-decode chain.

    :data:`Text2ImageRequest` is this body plus the knobs of ``PostprocessOptions``,
    which are appended to it as flat wire fields.
    """

    prompt: str
    negative_prompt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None

    steps: Optional[int] = None
    sigmas: Optional[List[float]] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    upscale: bool = False
    upscale_factor: float = 1.0
    upscale_type: str = "refined"
    sampler: Optional[str] = None
    lora_specs: Optional[List[str]] = None  # ["filename.safetensors:0.8", ...]
    pixel_upscaler: Optional[str] = None  # name (no .safetensors) in upscaler_dir
    kv_cache: Optional[bool] = None  # reference-latent KV cache (edit only; null = auto)
    ref_method: Optional[str] = None  # reference conditioning (edit only; null = auto)
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
            sigmas=self.sigmas,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
            upscale=self.upscale,
            upscale_factor=self.upscale_factor,
            upscale_type=self.upscale_type,
            sampler=self.sampler,
            **flat_options(self),
            lora_specs=self.lora_specs,
            pixel_upscaler=self.pixel_upscaler,
            kv_cache=self.kv_cache,
            ref_method=self.ref_method,
        )

Text2ImageRequest = create_model(
    "Text2ImageRequest", __base__=_Text2ImageBody, **wire_fields(PostprocessOptions)
)


class UpscaleRequest(BaseModel):
    image_b64: str  # base64-encoded input image bytes (PNG/JPEG)
    upscale_factor: float = 0.0  # desired final factor; 0.0 = detected native scale
    pixel_upscaler: str  # name (no .safetensors) in upscaler_dir
    out: Literal["png", "json"] = "png"


class ControlMapRequest(BaseModel):
    """Extract a control map, at the size of the image it was measured on."""

    image_b64: str
    # Off the option registry, so a new kind widens flags and UI with it.
    type: Literal[tuple(OPTIONS_BY_KIND)]
    model: Optional[str] = None  # implementation name; null = the kind's default
    canny: Optional[CannyOptions] = None
    depth: Optional[DepthOptions] = None
    pose: Optional[PoseOptions] = None
    lineart: Optional[LineartOptions] = None
    softedge: Optional[SoftedgeOptions] = None
    out: Literal["png", "json"] = "png"

    @model_validator(mode="after")
    def _only_the_chosen_kind(self) -> "ControlMapRequest":
        """Options for another kind are a mistake, not something to quietly ignore."""
        stray = [
            kind
            for kind in OPTIONS_BY_KIND
            if kind != self.type and getattr(self, kind) is not None
        ]
        if stray:
            raise ValueError(
                f"{', '.join(stray)} options need type "
                f"{' or '.join(repr(kind) for kind in stray)}, not {self.type!r}"
            )
        return self

    def to_options(self) -> ControlMapOptions:
        """The kind's options, or its defaults when the request named none."""
        return getattr(self, self.type) or OPTIONS_BY_KIND[self.type]()


class EditRequest(Text2ImageRequest):
    """Instruction-based editing: image(s) + prompt -> edited image.

    ``image`` accepts one or more base64-encoded images.
    """

    image: Union[str, List[str]]

    def to_edit_request(self):
        """Convert into a ``GenerateRequest`` carrying decoded PIL images."""
        from .models.config import GenerateRequest

        b64_list = self.image if isinstance(self.image, list) else [self.image]
        # Keep an alpha if the input has one; only the model knows whether it wants it.
        images = [decode_image(b) for b in b64_list]
        req: GenerateRequest = self.to_request()
        req.image = images[0] if len(images) == 1 else images
        return req


def _option_problem(error: dict) -> str:
    """One rejected option as a sentence that names it.

    A kind's options validate themselves and already say which field they mean; a
    pydantic constraint message does not, and "Input should be less than or equal to
    1" is no use to whoever sent the request.
    """
    message = error["msg"]
    name = str(error["loc"][-1]) if error["loc"] else ""
    if not name or name == "body" or message.startswith(name):
        return message
    return f"{name}: {message}"


def _is_options_problem(error: dict) -> bool:
    """Whether a validation error is about a request's options rather than the body."""
    where = error["loc"][1:] if error["loc"][:1] == ("body",) else list(error["loc"])
    return not where or where[0] in OPTIONS_BY_KIND or where[0] in POSTPROCESS_NAMES


def create_app(runtime) -> FastAPI:
    app = FastAPI(title="thenoise", version="0.1.0")

    def image_response(image, out: str):
        """PNG bytes, or the same bytes base64-wrapped, with the image's own metadata."""
        content = io.BytesIO()
        image.save(content, format="PNG", pnginfo=pnginfo_of(image))
        if out == "json":
            return {"b64_json": base64.b64encode(content.getvalue()).decode("ascii")}
        return Response(content=content.getvalue(), media_type="image/png")

    def image_job(failure: str, action, out: str):
        """Run one image-producing request, answering with the image or the failure.
        A ``ValueError`` is the request's fault (400, echoed); anything else is 500 + log.
        """
        try:
            image = action()
        except ValueError as e:
            return Response(status_code=400, content=str(e))
        except Exception as e:
            logger.exception("%s failed", failure)
            return Response(status_code=500, content=f"{failure} failed: {e}")
        return image_response(image, out)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        """Bad option values are the request's fault (400); unparseable bodies stay 422."""
        problems = [error for error in exc.errors() if _is_options_problem(error)]
        if not problems:
            return await request_validation_exception_handler(request, exc)
        return Response(status_code=400, content="; ".join(_option_problem(e) for e in problems))

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
            # 4 when the loaded model's VAE is RGBA, so the UI can show transparency.
            "pixel_channels": runtime.model_pixel_channels(),
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
        """Pixel-upscaler names and their detected native scales, with no model loaded.
        An undetectable entry is 0.
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

    @app.get("/postprocess")
    def postprocess():
        """The post-decode knobs, described for the web UI to build its controls from."""
        return {"options": options_schema(PostprocessOptions)}

    @app.get("/controlmaps")
    def controlmaps():
        """Control-map kinds, their implementations, and the fields each one takes."""
        return {
            "controlmaps": runtime.controlmaps.available(),
            "defaults": runtime.controlmaps.defaults(),
            # The person detectors a pose request crops its people with, by name.
            "detectors": runtime.controlmaps.detectors(),
            "options": runtime.controlmaps.option_schemas(),
        }

    @app.post("/text2image")
    def text2image(req: Text2ImageRequest):
        if runtime.pipeline is None:
            return Response(status_code=503, content="no model is loaded")
        return image_job(
            "generation",
            lambda: runtime.run(lambda pipeline: pipeline.generate(req.to_request())),
            req.out,
        )

    @app.post("/edit")
    def edit(req: EditRequest):
        """Reference-latent instruction editing (image + prompt -> edited image)."""
        if runtime.pipeline is None:
            return Response(status_code=503, content="no model is loaded")
        # A ValueError here is "this model does not edit" or "an edit needs an image".
        return image_job(
            "edit",
            lambda: runtime.run(lambda pipeline: pipeline.edit(req.to_edit_request())),
            req.out,
        )

    @app.post("/upscale")
    def upscale(req: UpscaleRequest):
        """Upscale an input image with a named pixel upscaler, with no model loaded."""
        return image_job(
            "upscale",
            lambda: runtime.upscaler.upscale(
                decode_image(req.image_b64), req.upscale_factor, req.pixel_upscaler
            ),
            req.out,
        )

    @app.post("/controlmap")
    def controlmap(req: ControlMapRequest):
        """Extract a control map from an input image, with no model loaded."""
        return image_job(
            "controlmap",
            lambda: runtime.controlmap.run(
                decode_image(req.image_b64),
                req.type,
                req.to_options(),
                req.model or "",
            ),
            req.out,
        )

    return app
