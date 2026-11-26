"""HTTP surface tests (no torch, no weights).

Cheap cases call the route functions directly; anything where FastAPI/pydantic
validation, status codes or error mapping matter goes through ``TestClient`` —
those are the branches a user hits and that a direct call silently skips.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from types import SimpleNamespace

import pytest

from thenoise.api import (
    EditRequest,
    UpscaleRequest,
    _UI_DIR,
    create_app,
)
from thenoise.controlmap.options import OPTIONS_BY_KIND
from thenoise.postprocess import POSTPROCESS_NAMES, PostprocessOptions
from thenoise.runtime import Runtime, Settings
from thenoise.samplers import SAMPLERS
from thenoise.utils.attention import ATTENTION_METHODS


class FakePipeline:
    """Records the converted request; optionally fails to exercise error paths."""

    def __init__(self, generate_error=None, edit_error=None):
        self.requests = []
        self.edit_requests = []
        self.generate_error = generate_error
        self.edit_error = edit_error
        # `/lora` asks the loaded model which components a LoRA can be given to.
        self.model = SimpleNamespace(lora_components=lambda: {"dit": object()})

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
    runtime._model = type("Model", (), {"MEDIUM": "image"})()
    runtime._model_name = "fake"
    if tmp_path is not None:
        (tmp_path / "RealESRGAN_x4.safetensors").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x2.safetensors").write_text("x")
        runtime._pixel_upscalers.upscaler_dir = str(tmp_path)
    return runtime


def _empty_runtime():
    return Runtime(Settings())


def _cpu_runtime():
    """A model-free runtime that really extracts, on the CPU."""
    return Runtime(Settings(device="cpu"))


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


def _rgba_png_b64(size=(2, 2)) -> str:
    """A PNG whose only pixel is fully transparent."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", size, (255, 0, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------- /health


def test_health_reports_the_loaded_model():
    runtime = _runtime()
    runtime._model = type("M", (), {
        "MEDIUM": "image",
        "CAPABILITIES": {"image": {"t2i": True, "i2i": True}, "kv_cache": True,
                         "upscale": True, "video": None},
        "REF_METHODS": ("index", "index_timestep_zero", "vision"),
        "ATTENTION_METHODS": ATTENTION_METHODS,
        "pixel_channels": 4,
    })()
    res = _endpoint(create_app(runtime), "/health")()

    assert res["models"] == ["fake"]
    assert res["medium"] == "image"
    # The UI gates Edit, the KV cache and transparent output on these, and an image model
    # answers the video group null rather than leaving the key out. ``ref_methods`` says
    # what a reference would be used for, so a client can ask for one it can have.
    assert res["capabilities"] == {"image": {"t2i": True, "i2i": True},
                                   "kv_cache": True, "upscale": True, "video": None,
                                   "ref_methods": ["index", "index_timestep_zero",
                                                   "vision"],
                                   "attention_methods": list(ATTENTION_METHODS)}
    # Every solver this build has, whichever model is loaded.
    assert res["samplers"] == sorted(SAMPLERS)
    assert res["pixel_channels"] == 4


def test_health_with_nothing_loaded():
    res = _endpoint(create_app(_empty_runtime()), "/health")()
    assert res["models"] == []
    assert res["medium"] is None
    assert res["capabilities"] == {}
    assert res["samplers"] == sorted(SAMPLERS)  # a solver needs no weights
    assert res["pixel_channels"] == 3  # nothing loaded -> nothing can be transparent



# ----------------------------------------------------------------------- /lora


def test_lora_lists_the_model_names_and_the_roles_they_can_be_given_to(client):
    res = client(_runtime()).get("/lora")
    assert res.status_code == 200
    # One component, so every client keeps sending a bare list; a two-expert model
    # adds its second role here and the UI gets a picker without new code.
    assert res.json() == {"loras": ["style", "pose"], "roles": ["dit"]}


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
    """Every local href/src in index.html is served; nothing is inlined or fetched.

    Structural on purpose: an exact-string grep breaks on any reformat.
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


def test_ui_scripts_only_look_up_ids_the_page_has():
    """An id the markup stopped having takes the rest of that script's bindings with it."""
    with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
        ids = set(re.findall(r'\bid="([^"]+)"', f.read()))
    assert ids

    js_dir = os.path.join(_UI_DIR, "js")
    checked = 0
    for name in sorted(os.listdir(js_dir)):
        if not name.endswith(".js"):
            continue
        with open(os.path.join(js_dir, name), encoding="utf-8") as f:
            script = f.read()
        for ref in re.findall(r"\$\('([A-Za-z0-9_]+)'\)", script):
            assert ref in ids, f"{name} looks up #{ref}, which index.html does not have"
            checked += 1
        for suffix in re.findall(r"\$\((?:prefix|p) \+ '([A-Za-z0-9_]+)'\)", script):
            assert any(i.endswith(suffix) for i in ids), (
                f"{name} looks up a {suffix!r} control no pane's markup has"
            )
            checked += 1
        # Ids held by a descriptor are looked up elsewhere, so they match here.
        patterns = [
            r"root: '([A-Za-z0-9_]+)', label: '([A-Za-z0-9_]+)', bar: '([A-Za-z0-9_]+)'",
            r"timerEl: '([A-Za-z0-9_]+)',\s*\n\s*timerTextEl: '([A-Za-z0-9_]+)'",
        ]
        for pattern in patterns:
            for refs in re.findall(pattern, script):
                for ref in refs:
                    assert ref in ids, (
                        f"{name} names #{ref} in a descriptor, which index.html does not have"
                    )
                    checked += 1
    assert checked > 50, f"barely any lookup checked ({checked}): the patterns drifted"


@pytest.mark.parametrize("menu", ["sampler", "edit_sampler", "video_sampler",
                                  "mesh_sampler"])
def test_ui_asks_the_server_which_samplers_exist(menu):
    """A sampler menu holds no solver name of its own beyond `auto`: the list is the
    server's to hand out.
    """
    with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
        block = re.search(rf'<select id="{menu}">(.*?)</select>', f.read(), re.S)[1]
    assert block.count("<option") == 1, block
    assert '<option value="">auto</option>' in block


def test_ui_offers_every_container_the_mesh_writer_has():
    """The shape pane's Output menu is the writer's own list, not a copy of it."""
    from thenoise.mesh.spec import OUTPUT_FORMATS

    with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
        block = re.search(r'<select id="mesh_output_format">(.*?)</select>', f.read(), re.S)[1]
    offered = [m.group(1) for m in re.finditer(r'<option value="([^"]*)"', block)]
    assert offered == ["", *OUTPUT_FORMATS], block


def test_ui_offers_the_density_presets_the_spec_keeps():
    """The pane's density menu is ``DENSITY_PRESETS``, values and all."""
    from thenoise.mesh.spec import DENSITY_PRESETS

    with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
        block = re.search(r'<select id="mesh_octree_resolution">(.*?)</select>',
                          f.read(), re.S)[1]
    offered = [one.group(1) for one in re.finditer(r'<option value="([^"]*)"', block)]
    assert offered == ["", *[str(one) for one in DENSITY_PRESETS]], block


def test_ui_keeps_the_shape_tuning_behind_one_disclosure():
    """The release's own configuration is folded away rather than removed."""
    tuning = {"mesh_latents", "mesh_steps", "mesh_guidance_scale", "mesh_sampler",
              "mesh_bounds", "mesh_mc_level"}
    shown = {"mesh_octree_resolution", "mesh_seed", "mesh_output_format",
             "mesh_lora_specs", "mesh_file"}

    with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
        page = f.read()
    mesh = page[page.index('id="view-mesh"'):page.index("/view-mesh")]
    group = re.search(r"<details[^>]*thenoise-mesh-tuning-open[^>]*>(.*?)</details>",
                      mesh, re.S)
    assert group, "the shape pane has no tuning disclosure"
    inside = set(re.findall(r'id="(mesh_[a-z_]+)"', group[1]))
    assert inside == tuning, inside
    assert shown <= set(re.findall(r'id="(mesh_[a-z_]+)"', mesh)) - inside


def test_ui_every_named_view_exists():
    """A view that a tab or the medium logic names but nobody serves is a blank screen."""
    with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(_UI_DIR, "js", "app.js"), encoding="utf-8") as f:
        app = f.read()

    views = set(re.findall(r'id="view-([^"]+)"', html))
    assert views
    named = set(re.findall(r'data-tab="([^"]+)"', html))
    medium = re.findall(r"MODEL_VIEWS = \[([^\]]*)\]", app)
    assert medium, "app.js no longer lists the views a model decides"
    named |= {v.strip().strip("'\"") for v in medium[0].split(",")}
    assert named <= views, f"nobody serves {sorted(named - views)}"


# ------------------------------------------------------------------- /upscalers


def test_upscalers_lists_names_and_scales(tmp_path, monkeypatch):
    # The fake files are not real safetensors, so header detection is stubbed.
    monkeypatch.setattr(
        "thenoise.upscale.pixel.detect_pixel_upscaler_scale", lambda path: 4
    )
    res = _endpoint(create_app(_runtime(tmp_path)), "/upscalers")()
    assert res["upscalers"] == ["RealESRGAN_x4", "sub/x2"]
    assert res["scales"] == {"RealESRGAN_x4": 4, "sub/x2": 4}


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


# ------------------------------------------------------------------- /postprocess


def test_postprocess_describes_every_knob_of_the_chain():
    """The UI's post-decode controls, its request body and the info rows all come from
    this payload, which is the declaration itself."""
    res = _endpoint(create_app(_empty_runtime()), "/postprocess")()
    described = {entry["name"]: entry for entry in res["options"]}

    assert set(described) == set(POSTPROCESS_NAMES)
    grain = described["film_grain"]
    assert grain["label"] == "Film grain"
    # What the server accepts (0-10) is wider than what a slider is worth offering (0-1).
    assert (grain["min"], grain["max"], grain["ui_max"], grain["step"]) == (0.0, 10.0, 1.0, 0.1)
    grid = described["qwen_vae_enhance"]
    assert (grid["type"], grid["default"], grid["hint"]) == ("boolean", False, "enable when needed")


# ----------------------------------------------------------------- /text2image


def test_text2image_passes_the_request_through(client, tmp_path):
    runtime = _runtime(tmp_path)
    res = client(runtime).post(
        "/text2image",
        json={
            "prompt": "a fox", "width": 512, "height": 512,
            "pixel_upscaler": "RealESRGAN_x4", "sigmas": [1.0, 0.5, 0.2],
        },
    )

    assert res.status_code == 200
    request = runtime._pipeline.requests[-1]
    assert (request.width, request.height) == (512, 512)
    assert request.pixel_upscaler == "RealESRGAN_x4"
    assert request.sigmas == [1.0, 0.5, 0.2]


def test_text2image_passes_every_post_decode_knob_through(client):
    """The knobs stay flat on the wire and reach the chain under the same names."""
    runtime = _runtime()
    res = client(runtime).post(
        "/text2image",
        json={"prompt": "a fox", "qwen_vae_enhance": True, "sharpening": 0.2, "film_grain": 0.5},
    )

    assert res.status_code == 200
    assert runtime._pipeline.requests[-1].postprocess_options() == PostprocessOptions(
        qwen_vae_enhance=True, sharpening=0.2, film_grain=0.5
    )


def test_text2image_returns_400_for_a_knob_outside_its_bounds(client):
    """An out-of-range knob is the caller's mistake, not a 422 or a crash downstream."""
    res = client(_runtime()).post("/text2image", json={"prompt": "x", "sharpening": 42})

    assert res.status_code == 400
    assert "sharpening" in res.text


def test_edit_passes_sigmas_through():
    runtime = _runtime()
    req = EditRequest(prompt="sunnier", image=_png_b64(), sigmas=[1.0, 0.0])
    res = _endpoint(create_app(runtime), "/edit")(req)

    assert res.status_code == 200
    assert runtime._pipeline.edit_requests[-1].sigmas == [1.0, 0.0]


def test_text2image_passes_empty_sigmas_through(client, tmp_path):
    """``[]`` is a value, not "unset": pydantic must not drop it on the way in."""
    runtime = _runtime(tmp_path)
    res = client(runtime).post("/text2image", json={"prompt": "x", "sigmas": []})

    assert res.status_code == 200
    assert runtime._pipeline.requests[-1].sigmas == []


def test_text2image_returns_400_for_an_invalid_field_value(client):
    """A rejected sigma grid (or any invalid value) is the caller's error, not ours."""
    runtime = _runtime(
        pipeline=FakePipeline(generate_error=ValueError("sigmas must be strictly decreasing"))
    )
    res = client(runtime).post("/text2image", json={"prompt": "x", "sigmas": [0.2, 0.9]})

    assert res.status_code == 400
    assert "strictly decreasing" in res.text


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


def test_edit_input_keeps_its_alpha():
    """The wire layer must not decide the channel count before the model does."""
    from PIL import Image

    req = EditRequest(prompt="cut it out", image=_rgba_png_b64())
    image = req.to_edit_request().image

    assert image.mode == "RGBA"
    assert image.getpixel((0, 0))[3] == 0


def test_edit_input_of_an_opaque_image_stays_rgb():
    """No alpha in the request -> no alpha handed to the pipeline."""
    assert EditRequest(prompt="x", image=_png_b64()).to_edit_request().image.mode == "RGB"


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


# ---------------------------------------------------------------- /controlmaps


def test_controlmaps_lists_kinds_and_defaults():
    """Extraction needs no diffusion model, so the UI can always fill its dropdowns."""
    res = _endpoint(create_app(_empty_runtime()), "/controlmaps")()
    # The option schemas are the fields each kind takes; asserted on their own below.
    assert set(res["options"]) == set(OPTIONS_BY_KIND)
    del res["options"]
    assert res == {
        "controlmaps": {
            "canny": ["canny"],
            "lineart": ["lineart"],
            "softedge": ["softedge"],
        },
        "defaults": {
            "canny": "canny",
            "lineart": "lineart",
            "softedge": "softedge",
        },
        "detectors": [],
    }


def test_controlmaps_describe_the_options_of_every_kind():
    """The form, the CLI flags and the request schema come from this payload."""
    res = _endpoint(create_app(_empty_runtime()), "/controlmaps")()
    assert res["options"]["canny"] == [
        {"name": "low", "label": "Low edge strength", "type": "number",
         "default": 100.0, "min": 0, "step": 1},
        {"name": "high", "label": "High edge strength", "type": "number",
         "default": 200.0, "min": 0, "step": 1},
        {"name": "thickness", "label": "Line width", "type": "integer",
         "default": 1, "min": 1, "step": 1},
    ]
    # A field whose choices only the server knows names the list to fill it from.
    assert res["options"]["pose"][0]["name"] == "detector"
    assert res["options"]["pose"][0]["pool"] == "detectors"
    person = res["options"]["pose"][1]
    assert (person["min"], person["max"], person["default"]) == (0, 1, 0.5)
    assert [c["default"] for c in res["options"]["depth"]] == [False, "gray"]


def test_controlmaps_list_discovered_depth_checkpoints(tmp_path):
    """A controlmap dir on disk widens what the dropdowns offer."""
    directory = tmp_path / "depth_anything_v2_small"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type": "depth_anything"}')
    runtime = Runtime(Settings(controlmap_dir=str(tmp_path)))
    res = _endpoint(create_app(runtime), "/controlmaps")()
    del res["options"]
    assert res == {
        "controlmaps": {
            "canny": ["canny"],
            "lineart": ["lineart"],
            "softedge": ["softedge"],
            "depth": ["depth_anything_v2_small"],
        },
        "defaults": {
            "canny": "canny",
            "lineart": "lineart",
            "softedge": "softedge",
            "depth": "depth_anything_v2_small",
        },
        "detectors": [],
    }


def test_controlmaps_list_the_pose_checkpoints(tmp_path):
    """The UI's person-finder dropdown and its Pose entry both come from here: the
    detector is listed by name, and pose appears only once both checkpoints are there."""
    for name, model_type in (("vitpose_base_simple", "vitpose"), ("rtdetr_r18vd", "rt_detr")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    runtime = Runtime(Settings(controlmap_dir=str(tmp_path)))
    res = _endpoint(create_app(runtime), "/controlmaps")()
    assert res["controlmaps"]["pose"] == ["vitpose_base_simple"]
    assert res["detectors"] == ["rtdetr_r18vd"]


# ------------------------------------------------------------------- /controlmap


def _edged_png_b64(size=(16, 8)) -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size)
    ImageDraw.Draw(image).rectangle([2, 2, size[0] - 3, size[1] - 3], fill="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_controlmap_extracts_a_map_with_no_model_loaded(client):
    res = client(_cpu_runtime()).post(
        "/controlmap", json={"image_b64": _edged_png_b64(), "type": "canny"}
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"

    from PIL import Image

    image = Image.open(io.BytesIO(res.content))
    assert (image.mode, image.size) == ("L", (16, 8))
    assert set(image.get_flattened_data()) == {0, 255}  # a real map, not a blank one
    assert json.loads(image.info["controlmap_data"])["model"] == "canny"


def test_controlmap_answers_at_the_input_size(client):
    """A control map is paired with the image it was extracted from."""
    from PIL import Image

    res = client(_cpu_runtime()).post(
        "/controlmap", json={"image_b64": _edged_png_b64((30, 11)), "type": "canny"}
    )
    assert Image.open(io.BytesIO(res.content)).size == (30, 11)


def test_controlmap_applies_the_requested_options(client):
    res = client(_cpu_runtime()).post(
        "/controlmap",
        json={
            "image_b64": _edged_png_b64(), "type": "canny",
            "canny": {"low": 30, "high": 60, "thickness": 3},
        },
    )
    assert res.status_code == 200

    from PIL import Image

    image = Image.open(io.BytesIO(res.content))
    assert image.size == (16, 8)
    assert json.loads(image.info["controlmap_data"]) == {
        "type": "canny",
        "model": "canny",
        "options": {"low": 30.0, "high": 60.0, "thickness": 3},
    }


def test_controlmap_json_returns_decodable_b64(client):
    res = client(_cpu_runtime()).post(
        "/controlmap",
        json={"image_b64": _edged_png_b64(), "type": "canny", "out": "json"},
    )
    assert set(res.json()) == {"b64_json"}
    assert base64.b64decode(res.json()["b64_json"]).startswith(b"\x89PNG")


def test_controlmap_rejects_an_unknown_implementation(client):
    res = client(_cpu_runtime()).post(
        "/controlmap", json={"image_b64": _edged_png_b64(), "type": "canny", "model": "canny-xl"}
    )
    assert res.status_code == 400
    assert "available: canny" in res.text


def test_controlmap_returns_400_for_invalid_options(client):
    """Thresholds are validated by the request's own options dataclass."""
    res = client(_cpu_runtime()).post(
        "/controlmap",
        json={"image_b64": _edged_png_b64(), "type": "canny", "canny": {"low": 200, "high": 100}},
    )
    assert res.status_code == 400
    assert "exceeds high threshold" in res.text


def test_controlmap_returns_400_for_undecodable_image(client):
    res = client(_cpu_runtime()).post(
        "/controlmap", json={"image_b64": "not base64 @@", "type": "canny"}
    )
    assert res.status_code == 400


def test_controlmap_missing_depth_weights_names_the_download(client, tmp_path):
    """A discovered kind whose checkpoint is incomplete answers 400 with the remedy."""
    directory = tmp_path / "depth_anything_v2_small"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type": "depth_anything"}')
    runtime = Runtime(Settings(controlmap_dir=str(tmp_path)))
    res = client(runtime).post(
        "/controlmap", json={"image_b64": _edged_png_b64(), "type": "depth"}
    )
    assert res.status_code == 400
    assert "scripts/download.py --model controlmaps" in res.text


def test_controlmap_a_pose_request_without_checkpoints_names_the_pose_download(client, tmp_path):
    """The remedy has to be the one that fetches *this* kind, pose being two checkpoints
    of its own rather than the depth model the default variant downloads."""
    runtime = Runtime(Settings(controlmap_dir=str(tmp_path)))
    res = client(runtime).post(
        "/controlmap", json={"image_b64": _edged_png_b64(), "type": "pose"}
    )
    assert res.status_code == 400
    assert "--model controlmaps --variant pose" in res.text


def test_a_pose_request_becomes_the_pose_options():
    from thenoise.api import ControlMapRequest
    from thenoise.controlmap.pose import PoseOptions

    request = ControlMapRequest(
        image_b64="x", type="pose", pose={"radius": 2, "detector": "rtdetr_r50vd"}
    )
    assert request.to_options() == PoseOptions(radius=2, detector="rtdetr_r50vd")
    # A request that names nothing gets the kind's own defaults, detector included.
    assert ControlMapRequest(image_b64="x", type="pose").to_options() == PoseOptions()


def test_lineart_and_softedge_requests_become_their_options():
    from thenoise.api import ControlMapRequest
    from thenoise.controlmap.lineart import LineartOptions
    from thenoise.controlmap.softedge import SoftedgeOptions

    request = ControlMapRequest(
        image_b64="x", type="lineart", lineart={"threshold": 0.8, "thickness": 3}
    )
    assert request.to_options() == LineartOptions(threshold=0.8, thickness=3)
    assert ControlMapRequest(image_b64="x", type="lineart").to_options() == LineartOptions()
    assert ControlMapRequest(
        image_b64="x", type="softedge", softedge={"threshold": 0.1, "blur": 3.0}
    ).to_options() == SoftedgeOptions(threshold=0.1, blur=3.0)
    assert ControlMapRequest(image_b64="x", type="softedge").to_options() == SoftedgeOptions()



def test_controlmap_backend_failure_is_500(client, monkeypatch):
    runtime = _cpu_runtime()

    def boom(kind, name=""):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(runtime.controlmaps, "get", boom)
    res = client(runtime).post(
        "/controlmap", json={"image_b64": _edged_png_b64(), "type": "canny"}
    )
    assert res.status_code == 500
    assert "controlmap failed" in res.text


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
        # ``sigmas`` is a list of numbers, not a ComfyUI string.
        ("/text2image", {"prompt": "x", "sigmas": "1.0, 0.5"}),
        ("/text2image", {"prompt": "x", "sigmas": ["abc"]}),
        # ``image_b64`` is required on /controlmap.
        ("/controlmap", {"type": "canny"}),
        # ``type`` only accepts the implemented control-map kinds.
        ("/controlmap", {"image_b64": "abc", "type": "normals"}),
    ],
    ids=["bad-out", "missing-prompt", "missing-upscaler", "missing-image",
         "sigmas-string", "sigmas-not-numbers", "missing-map-image", "unknown-map-type"],
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
