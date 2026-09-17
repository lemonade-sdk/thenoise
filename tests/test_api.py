"""HTTP surface tests (no torch, no weights).

Cheap cases call the route functions directly; anything where FastAPI/pydantic
request validation, status codes or error mapping matter goes through
``TestClient`` — those are exactly the branches a user hits and that a direct
call silently skips.
"""
from __future__ import annotations

import base64
import io
import os
import re

import pytest

from thenoise.api import (
    EditRequest,
    Text2ImageRequest,
    UpscaleRequest,
    _UI_DIR,
    create_app,
)
from thenoise.runtime import Runtime, Settings


class FakePipeline:
    """Records the converted request; optionally fails to exercise error paths."""

    def __init__(self, generate_error=None, edit_error=None):
        self.requests = []
        self.edit_requests = []
        self.generate_error = generate_error
        self.edit_error = edit_error

    @staticmethod
    def _image(size=(8, 8)):
        from PIL import Image

        return Image.new("RGB", size)

    def generate(self, request):
        if self.generate_error:
            raise self.generate_error
        self.requests.append(request)
        return self._image()

    def edit(self, request):
        if self.edit_error:
            raise self.edit_error
        self.edit_requests.append(request)
        return self._image()

    def list_loras(self):
        return ["style", "pose"]


def _runtime(tmp_path=None, pipeline=None):
    """A runtime with a fake pipeline + model (and optionally an upscaler dir)."""
    runtime = Runtime(Settings())
    runtime._pipeline = pipeline if pipeline is not None else FakePipeline()
    runtime._model = object()
    runtime._model_name = "fake"
    if tmp_path is not None:
        (tmp_path / "RealESRGAN_x4.safetensors").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x2.safetensors").write_text("x")
        runtime._pixel_upscalers.upscaler_dir = str(tmp_path)
    return runtime


def _empty_runtime():
    return Runtime(Settings())


def _endpoint(app, path):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"no route {path}")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    def make(runtime):
        return TestClient(create_app(runtime))

    return make


def _png_b64(size=(2, 2)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------- /health


def test_health_reports_edit_capability():
    """/health exposes model capabilities so the UI can gate the Edit tab."""
    runtime = _runtime()
    runtime._model = type("M", (), {"supports_edit": True})()
    res = _endpoint(create_app(runtime), "/health")()
    assert res["models"] == ["fake"]
    assert res["capabilities"] == {"supports_edit": True}


def test_health_capabilities_empty_without_model():
    res = _endpoint(create_app(_empty_runtime()), "/health")()
    assert res["models"] == []
    assert res["capabilities"] == {}


# ----------------------------------------------------------------------- /lora


def test_lora_lists_the_model_names(client):
    res = client(_runtime()).get("/lora")
    assert res.status_code == 200
    assert res.json() == {"loras": ["style", "pose"]}


def test_lora_requires_a_loaded_model(client):
    res = client(_empty_runtime()).get("/lora")
    assert res.status_code == 503


# ----------------------------------------------------------------------- `/`


def test_index_serves_the_ui(client):
    res = client(_empty_runtime()).get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "<html" in res.text


def test_ui_assets_are_shipped_and_served(client):
    """Every local href/src in index.html is served (no 404s once the UI loads).

    Structural on purpose: an exact-string grep breaks on any reformat, while the
    real contract is "the linked files ship with the package and the route serves
    them, and there is no inline CSS/JS or network font dependency".
    """
    with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read()

    refs = re.findall(r'(?:href|src)="([^"]+)"', html)
    local = [
        r for r in refs
        if not r.startswith(("http:", "https:", "//", "data:", "#", "mailto:"))
    ]
    assert local, f"index.html links nothing local: {refs}"

    http = client(_empty_runtime())
    for ref in local:
        # The page is served at "/", so a relative "static/x" resolves to /static/x.
        res = http.get("/" + ref)
        assert res.status_code == 200, f"{ref} is linked but not served"
        assert res.content, f"{ref} is served empty"

    assert "<style>" not in html
    assert "<script>" not in html
    assert "fonts.googleapis.com" not in html


# ------------------------------------------------------------------- /upscalers


def test_upscalers_lists_names_and_scales(tmp_path, monkeypatch):
    # Non-seeded path: the scale is read from each model's header. The fake files
    # are not real safetensors, so stub detection by filename.
    def fake_detect(path):
        if "x4" in path:
            return 4
        if "x2" in path:
            return 2
        raise ValueError("unknown scale")

    monkeypatch.setattr("thenoise.upscale.pixel.detect_pixel_upscaler_scale", fake_detect)
    res = _endpoint(create_app(_runtime(tmp_path)), "/upscalers")()
    assert res["upscalers"] == ["RealESRGAN_x4", "sub/x2"]
    assert res["scales"] == {"RealESRGAN_x4": 4, "sub/x2": 2}


def test_upscalers_scales_pass_through(tmp_path):
    # Pre-seeded scales are served verbatim (no file access needed).
    runtime = _runtime(tmp_path)
    runtime._pixel_upscalers._pixel_upscaler_scales = {"RealESRGAN_x4": 4, "sub/x2": 2}
    res = _endpoint(create_app(runtime), "/upscalers")()
    assert res["scales"] == {"RealESRGAN_x4": 4, "sub/x2": 2}


def test_upscalers_undetectable_scale_is_reported_as_zero(tmp_path, monkeypatch):
    """A broken file must not 500 the listing the whole UI tab depends on."""
    monkeypatch.setattr(
        "thenoise.upscale.pixel.detect_pixel_upscaler_scale",
        lambda path: (_ for _ in ()).throw(ValueError("bad header")),
    )
    res = _endpoint(create_app(_runtime(tmp_path)), "/upscalers")()
    assert res["upscalers"] == ["RealESRGAN_x4", "sub/x2"]
    assert res["scales"] == {"RealESRGAN_x4": 0, "sub/x2": 0}


def test_upscalers_available_without_model():
    # Pixel upscalers are a pixel-space/server concern and need no diffusion model.
    res = _endpoint(create_app(_empty_runtime()), "/upscalers")()
    assert res["upscalers"] == []


# ----------------------------------------------------------------- /text2image


def test_text2image_passes_the_pixel_upscaler_through(tmp_path):
    runtime = _runtime(tmp_path)
    req = Text2ImageRequest(
        prompt="a fox", width=512, height=512, pixel_upscaler="RealESRGAN_x4"
    )
    res = _endpoint(create_app(runtime), "/text2image")(req)
    assert res.status_code == 200
    assert runtime._pipeline.requests[-1].pixel_upscaler == "RealESRGAN_x4"


def test_text2image_returns_png_bytes_by_default(client):
    res = client(_runtime()).post("/text2image", json={"prompt": "a fox", "out": "png"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content.startswith(b"\x89PNG")


def test_text2image_json_returns_decodable_b64(client):
    res = client(_runtime()).post("/text2image", json={"prompt": "a fox", "out": "json"})
    assert res.status_code == 200
    payload = res.json()
    assert set(payload) == {"b64_json"}
    assert base64.b64decode(payload["b64_json"]).startswith(b"\x89PNG")


def test_text2image_without_model_is_503(client):
    res = client(_empty_runtime()).post("/text2image", json={"prompt": "a fox"})
    assert res.status_code == 503


def test_text2image_backend_failure_is_500(client):
    runtime = _runtime(pipeline=FakePipeline(generate_error=RuntimeError("CUDA out of memory")))
    res = client(runtime).post("/text2image", json={"prompt": "a fox"})
    assert res.status_code == 500
    assert "generation failed" in res.text


# ----------------------------------------------------------------------- /edit


def test_edit_decodes_a_single_image():
    """One b64 string -> a single ``PIL.Image``, not a one-element list."""
    from PIL import Image

    req = EditRequest(prompt="make it sunny", image=_png_b64())
    request = req.to_edit_request()

    assert isinstance(request.image, Image.Image)
    assert request.image.size == (2, 2)
    assert request.prompt == "make it sunny"


def test_edit_decodes_a_list_of_images():
    req = EditRequest(prompt="blend", image=[_png_b64((2, 2)), _png_b64((4, 4))])
    images = req.to_edit_request().image
    assert [img.size for img in images] == [(2, 2), (4, 4)]


def test_edit_returns_png(client):
    runtime = _runtime()
    res = client(runtime).post("/edit", json={"prompt": "sunny", "image": _png_b64()})
    assert res.status_code == 200
    assert res.content.startswith(b"\x89PNG")
    assert runtime._pipeline.edit_requests[-1].prompt == "sunny"


def test_edit_without_model_is_503(client):
    res = client(_empty_runtime()).post("/edit", json={"prompt": "x", "image": _png_b64()})
    assert res.status_code == 503


def test_edit_on_a_non_editing_model_is_400(client):
    """The controller's ValueError is a client error, not a server failure."""
    runtime = _runtime(
        pipeline=FakePipeline(edit_error=ValueError("model 'fake' does not support image editing"))
    )
    res = client(runtime).post("/edit", json={"prompt": "x", "image": _png_b64()})
    assert res.status_code == 400
    assert "does not support image editing" in res.text


@pytest.mark.parametrize(
    "image,expected",
    [
        # Malformed base64 (a ValueError) is a bad request...
        ("A", 400),
        # ...while decodable bytes that are not an image are a server-side failure.
        (base64.b64encode(b"not an image").decode(), 500),
    ],
    ids=["invalid-base64", "not-an-image"],
)
def test_edit_rejects_bad_image_payloads(client, image, expected):
    res = client(_runtime()).post("/edit", json={"prompt": "x", "image": image})
    assert res.status_code == expected


def test_edit_backend_failure_is_500(client):
    runtime = _runtime(pipeline=FakePipeline(edit_error=RuntimeError("boom")))
    res = client(runtime).post("/edit", json={"prompt": "x", "image": _png_b64()})
    assert res.status_code == 500
    assert "edit failed" in res.text


# --------------------------------------------------------------------- /upscale


def test_upscale_returns_400_without_upscaler_dir():
    """No upscaler_dir configured -> controller raises ValueError -> 400."""
    req = UpscaleRequest(image_b64=_png_b64(), pixel_upscaler="x4")
    assert _endpoint(create_app(_empty_runtime()), "/upscale")(req).status_code == 400


def test_upscale_returns_400_for_unknown_upscaler(tmp_path):
    req = UpscaleRequest(image_b64=_png_b64(), pixel_upscaler="nonexistent")
    assert _endpoint(create_app(_runtime(tmp_path)), "/upscale")(req).status_code == 400


def test_upscale_backend_failure_is_500(client, monkeypatch):
    """A non-ValueError failure must not be reported as a bad request."""
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.upscaler, "upscale",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("device hung")),
    )
    res = client(runtime).post(
        "/upscale", json={"image_b64": _png_b64(), "pixel_upscaler": "x4"}
    )
    assert res.status_code == 500
    assert "upscale failed" in res.text


@pytest.mark.parametrize(
    "out,assert_result",
    [
        ("png", lambda res: res.headers["content-type"] == "image/png" and bool(res.content)),
        ("json", lambda res: "b64_json" in res.json()),
    ],
    ids=["png", "json"],
)
def test_upscale_success_payloads(client, monkeypatch, out, assert_result):
    from PIL import Image

    runtime = _runtime()
    monkeypatch.setattr(
        runtime.upscaler, "upscale", lambda image, f, name: Image.new("RGB", (4, 4))
    )
    res = client(runtime).post(
        "/upscale", json={"image_b64": _png_b64(), "pixel_upscaler": "x4", "out": out}
    )
    assert res.status_code == 200
    assert assert_result(res)


def test_upscale_works_without_a_model(client, monkeypatch):
    from PIL import Image

    runtime = _empty_runtime()
    monkeypatch.setattr(
        runtime.upscaler, "upscale", lambda image, f, name: Image.new("RGB", (4, 4))
    )
    res = client(runtime).post(
        "/upscale", json={"image_b64": _png_b64(), "pixel_upscaler": "x4"}
    )
    assert res.status_code == 200


# ------------------------------------------------------------- validation (422)


@pytest.mark.parametrize(
    "path,payload",
    [
        # ``out`` is a Literal["png", "json"].
        ("/text2image", {"prompt": "x", "out": "webp"}),
        # ``prompt`` is required.
        ("/text2image", {"width": 64}),
        # ``pixel_upscaler`` is required on /upscale.
        ("/upscale", {"image_b64": "abc"}),
        # ``image`` is required on /edit.
        ("/edit", {"prompt": "x"}),
    ],
    ids=["bad-out", "missing-prompt", "missing-upscaler", "missing-image"],
)
def test_invalid_payload_is_rejected_by_pydantic(client, path, payload):
    assert client(_runtime()).post(path, json=payload).status_code == 422


# ----------------------------------------------------------------------- static


def test_static_blocks_path_traversal():
    res = _endpoint(create_app(_empty_runtime()), "/static/{filename:path}")("../../etc/passwd")
    assert res.status_code == 403


def test_static_missing_file_returns_404():
    res = _endpoint(create_app(_empty_runtime()), "/static/{filename:path}")("nope.txt")
    assert res.status_code == 404
