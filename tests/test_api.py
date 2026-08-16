"""API tests using a fake model (no torch, no weights, no TestClient)."""
from __future__ import annotations

from thenoise.api import create_app, Text2ImageRequest
from thenoise.runtime import Settings, Runtime


def _fake_runtime():
    class FakeModel:
        name = "fake"
        loras = ["style", "pose"]
        upscalers = ["RealESRGAN_x4", "sub/x2"]

        def list_loras(self):
            return list(self.loras)

        def list_pixel_upscalers(self):
            return list(self.upscalers)

        def generate(self, **kwargs):
            self.last_kwargs = kwargs
            from PIL import Image
            return Image.new("RGB", (8, 8))

    runtime = Runtime(Settings())
    runtime._model = FakeModel()
    runtime._model_name = "fake"
    return runtime


def _empty_runtime():
    return Runtime(Settings())


def _endpoint(app, path):
    for r in app.routes:
        if getattr(r, "path", None) == path:
            return r.endpoint
    raise AssertionError(f"no route {path}")


def test_upscalers_lists_names():
    app = create_app(_fake_runtime())
    res = _endpoint(app, "/upscalers")()
    assert res["upscalers"] == ["RealESRGAN_x4", "sub/x2"]


def test_upscalers_503_when_no_model():
    app = create_app(_empty_runtime())
    res = _endpoint(app, "/upscalers")()
    assert res.status_code == 503


def test_text2image_passes_pixel_upscaler():
    runtime = _fake_runtime()
    app = create_app(runtime)
    req = Text2ImageRequest(prompt="a fox", pixel_upscaler="RealESRGAN_x4")
    res = _endpoint(app, "/text2image")(req)
    assert res.status_code == 200
    assert runtime.model.last_kwargs["pixel_upscaler"] == "RealESRGAN_x4"


def test_request_field_defaults_none():
    req = Text2ImageRequest(prompt="x")
    assert req.pixel_upscaler is None
