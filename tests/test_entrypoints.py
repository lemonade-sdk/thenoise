"""Entrypoint wiring tests: arg -> ``Settings``/``ModelPaths``/``Runtime``.

CLI *parsing* is covered in ``test_cli.py``. Here we call the dispatch functions
(``_serve``, ``run_generate``, ``run_edit``, ``run_upscale``) with a faked
``Runtime`` and assert the constructed ``Settings``/``ModelPaths`` and the chosen
controller method — without touching torch devices or loading any weights.
"""
from __future__ import annotations

import pytest
from PIL import Image

from thenoise.__main__ import _serve, main
from thenoise.cli import build_parser
from thenoise.generate import run_edit, run_generate
from thenoise.runtime import ModelPaths, Settings
from thenoise.upscale_cli import run_upscale
from thenoise.utils.paths import ensure_png_extension


class _FakeImage:
    def __init__(self, factor=None):
        self._pnginfo = {"parameters": "fake"}
        self._upscale_factor = factor
        self.saved_path = None
        self.saved_pnginfo = None

    def save(self, path, pnginfo=None):
        self.saved_path = path
        self.saved_pnginfo = pnginfo


class _FakePipeline:
    def __init__(self):
        self.generate_request = None
        self.edit_request = None
        self.images = []

    def generate(self, request):
        self.generate_request = request
        img = _FakeImage()
        self.images.append(img)
        return img

    def edit(self, request):
        self.edit_request = request
        img = _FakeImage()
        self.images.append(img)
        return img


class _FakeUpscaler:
    def __init__(self):
        self.calls = []

    def upscale(self, image, factor, name):
        self.calls.append((image, factor, name))
        return _FakeImage(factor=factor)


class _FakeRuntime:
    def __init__(self, settings):
        self.settings = settings
        self.loaded = None
        self.pipeline = _FakePipeline()
        self.upscaler = _FakeUpscaler()
        self.model_name = "stub"

    def load(self, paths):
        self.loaded = paths

    def available(self):
        return True


@pytest.fixture
def fake_runtime(monkeypatch):
    """Install a fake ``Runtime``; the deferred ``from .runtime import ...`` in the
    dispatch functions picks it up, so no real model/pixel-manager is built."""
    runtimes = []

    def make(settings):
        rt = _FakeRuntime(settings)
        runtimes.append(rt)
        return rt

    monkeypatch.setattr("thenoise.runtime.Runtime", make)
    return runtimes


def _parse(argv):
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------- serve


def test_serve_builds_settings_and_loads_model(monkeypatch, fake_runtime):
    monkeypatch.setattr("thenoise.api.create_app", lambda runtime: object())
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: None)
    args = _parse(
        [
            "serve",
            "--dit", "d.safetensors",
            "--vae", "v.safetensors",
            "--text-encoder", "t.safetensors",
            "--lora-dir", "loras",
            "--host", "0.0.0.0",
            "--port", "9000",
            "--device", "cuda",
            "--offload-device", "cpu",
            "--upscaler-dir", "ups",
        ]
    )
    _serve(args)
    rt = fake_runtime[0]
    assert rt.settings == Settings(
        device="cuda",
        offload_device="cpu",
        host="0.0.0.0",
        port=9000,
        upscaler_dir="ups",
    )
    assert rt.loaded == ModelPaths(
        dit_path="d.safetensors",
        vae_path="v.safetensors",
        text_encoder_path="t.safetensors",
        lora_dir="loras",
    )


def test_serve_without_model_does_not_load(monkeypatch, fake_runtime):
    monkeypatch.setattr("thenoise.api.create_app", lambda runtime: object())
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: None)
    _serve(_parse(["serve"]))
    rt = fake_runtime[0]
    assert rt.loaded is None
    assert rt.settings == Settings(device="cuda", upscaler_dir="")


def test_serve_rejects_partial_model_paths(fake_runtime):
    with pytest.raises(SystemExit):
        _serve(_parse(["serve", "--dit", "d.safetensors"]))


# ---------------------------------------------------------------- generate


def test_generate_wires_settings_paths_and_request(monkeypatch, fake_runtime):
    monkeypatch.setattr("thenoise.generate.random.randint", lambda a, b: 12345)
    args = _parse(
        [
            "generate",
            "--dit", "d",
            "--vae", "v",
            "--text-encoder", "t",
            "--prompt", "hello",
            "--negative-prompt", "bad",
            "--width", "512",
            "--height", "768",
            "--steps", "30",
            "--guidance-scale", "7.5",
            "--seed", "99",
            "--out", "result",
            "--lora", "style:0.8",
            "--pixel-upscaler", "models/realesrgan.safetensors",
            "--upscale",
            "--upscale-factor", "4",
            "--sampler", "er_sde",
            "--qwen-vae-enhance",
            "--film-grain", "0.5",
            "--sharpening", "0.25",
            "--device", "cuda",
        ]
    )
    run_generate(args)
    rt = fake_runtime[0]
    # One-shot --pixel-upscaler is split into upscaler_dir + name.
    assert rt.settings.device == "cuda"
    assert rt.settings.upscaler_dir == "models"
    assert rt.loaded == ModelPaths(dit_path="d", vae_path="v", text_encoder_path="t")
    req = rt.pipeline.generate_request
    assert req.prompt == "hello"
    assert req.negative_prompt == "bad"
    assert req.width == 512
    assert req.height == 768
    assert req.steps == 30
    assert req.guidance_scale == 7.5
    assert req.seed == 99
    assert req.upscale is True
    assert req.upscale_factor == 4.0
    assert req.sampler == "er_sde"
    assert req.qwen_vae_enhance is True
    assert req.film_grain == 0.5
    assert req.sharpening == 0.25
    assert req.lora_specs == ["style:0.8"]
    assert req.pixel_upscaler == "realesrgan"
    # The bare output name gets the PNG extension so PIL can infer a format.
    img = rt.pipeline.images[0]
    assert img.saved_path == ensure_png_extension("result")
    assert img.saved_pnginfo == {"parameters": "fake"}


def test_generate_resolves_seed_when_not_given(monkeypatch, fake_runtime):
    monkeypatch.setattr("thenoise.generate.random.randint", lambda a, b: 12345)
    run_generate(
        _parse(["generate", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x"])
    )
    assert fake_runtime[0].pipeline.generate_request.seed == 12345


@pytest.mark.parametrize("name,value", [("width", 5000), ("height", -1)])
def test_generate_rejects_out_of_range_dimensions(capsys, name, value):
    argv = ["generate", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x"]
    argv += [f"--{name}", str(value)]
    with pytest.raises(SystemExit):
        run_generate(_parse(argv))
    assert f"error: {name} must be between 0 and 4096" in capsys.readouterr().err


# ---------------------------------------------------------------- edit


def test_edit_wires_images_into_the_request(fake_runtime, tmp_path):
    img_path = tmp_path / "in.png"
    Image.new("RGB", (64, 64)).save(img_path)
    run_edit(
        _parse(
            [
                "edit",
                "--dit", "d",
                "--vae", "v",
                "--text-encoder", "t",
                "--prompt", "make it sunny",
                "--image", str(img_path),
                "--seed", "7",
            ]
        )
    )
    rt = fake_runtime[0]
    req = rt.pipeline.edit_request
    assert req.prompt == "make it sunny"
    assert isinstance(req.image, list)
    assert len(req.image) == 1
    assert req.image[0].size == (64, 64)
    assert rt.loaded == ModelPaths(dit_path="d", vae_path="v", text_encoder_path="t")


# ---------------------------------------------------------------- upscale


def test_upscale_is_model_free_and_splits_the_path(fake_runtime, tmp_path):
    img_path = tmp_path / "in.png"
    Image.new("RGB", (64, 64)).save(img_path)
    run_upscale(
        _parse(
            [
                "upscale",
                "--pixel-upscaler", "models/realesrgan.safetensors",
                "--input", str(img_path),
                "--upscale-factor", "2",
                "--out", "up.png",
            ]
        )
    )
    rt = fake_runtime[0]
    assert rt.settings.device == "cuda"
    assert rt.settings.upscaler_dir == "models"
    assert rt.loaded is None  # model-free: no load() call
    image, factor, name = rt.upscaler.calls[0]
    assert factor == 2
    assert name == "realesrgan"


# ---------------------------------------------------------------- dispatch


def test_main_dispatches_to_the_correct_handler(monkeypatch):
    calls = []
    monkeypatch.setattr("thenoise.generate.run_generate", lambda args: calls.append("generate"))
    monkeypatch.setattr("thenoise.generate.run_edit", lambda args: calls.append("edit"))
    monkeypatch.setattr("thenoise.upscale_cli.run_upscale", lambda args: calls.append("upscale"))
    monkeypatch.setattr("thenoise.__main__._serve", lambda args: calls.append("serve"))

    main(["generate", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x"])
    main(["edit", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x", "--image", "i.png"])
    main(["upscale", "--pixel-upscaler", "u.safetensors", "--input", "i.png"])
    main(["serve"])
    assert calls == ["generate", "edit", "upscale", "serve"]
