"""Entrypoint wiring tests: arg -> ``Settings``/``ModelPaths``/``Runtime``/controller.
``test_cli.py`` covers parsing; here a faked ``Runtime`` shows what each handler calls.
"""
from __future__ import annotations

import logging

import pytest
from PIL import Image

from thenoise.__main__ import _QuietReads, _serve, main
from thenoise.cli import build_parser
from thenoise.generate import run_edit, run_generate
from thenoise.controlmap.manager import ControlMapManager
from thenoise.controlmap_cli import run_controlmap
from thenoise.jobs import Uncancellable
from thenoise.runtime import ModelPaths, Settings
from thenoise.upscale_cli import run_upscale
from thenoise.utils.paths import ensure_png_extension
from thenoise.video_cli import run_video


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


class _FakeControlMap:
    def __init__(self):
        self.calls = []
        self.images = []

    def run(self, image, kind, options, name=""):
        self.calls.append((image, kind, options, name))
        image = _FakeImage()
        self.images.append(image)
        return image


class _FakeVideo:
    """A video controller with fixed model defaults, keeping the render it was handed."""

    DEFAULTS = {"negative_prompt": "", "width": 480, "height": 272, "frames": 33,
                "fps": 16, "steps": 40, "guidance_scale": 4.0, "flow_shift": 12.0,
                "sampler": "euler", "output_format": "mp4"}

    def __init__(self):
        self.spec = None
        self.rendered = None

    def resolve(self, spec):
        self.spec = spec
        return spec.resolve(self.DEFAULTS)

    def render(self, spec, token, progress, output):
        progress("sampling", 0, 2)
        progress("sampling", 2, 2)
        progress("decoding", 2, 2)
        progress("encoding", 2, 2)
        output.write_bytes(b"mp4")
        self.rendered = {"spec": spec, "token": token, "output": output}


class _FakeRuntime:
    #: What a loaded model answers for ``Runtime.video``; the clip tests flip it.
    video_on_load = False

    def __init__(self, settings):
        self.settings = settings
        self.loaded = None
        self.video = False
        self.pipeline = _FakePipeline()
        self.upscaler = _FakeUpscaler()
        # Weight-free and I/O-free, so options/availability stay the real ones.
        self.controlmaps = ControlMapManager(device="cpu")
        self.controlmap = _FakeControlMap()
        self.model_name = "stub"

    def load(self, paths):
        self.loaded = paths
        if self.video_on_load:
            self.video = True
            self.pipeline = _FakeVideo()

    def run(self, action):
        """As the real one, minus the rebuild the tests here are not about."""
        return action(self.pipeline)

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


def test_successful_job_reads_are_the_only_lines_dropped():
    def access_line(method, path, status):
        record = logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1,
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:52000", method, path, "1.1", status), None)
        return not _QuietReads().filter(record)

    assert access_line("GET", "/jobs/3f9a", 200)
    assert access_line("GET", "/jobs", 200)
    assert access_line("GET", "/jobs/3f9a/content", 200)
    assert not access_line("GET", "/jobs/3f9a", 404)
    assert not access_line("POST", "/jobs/video", 202)
    assert not access_line("GET", "/health", 200)


def test_serve_builds_settings_and_loads_model(monkeypatch, fake_runtime):
    monkeypatch.setattr(
        "thenoise.api.create_app",
        lambda runtime, *, jobs=None: object(),
    )
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: None)
    args = _parse(
        [
            "serve",
            "--dit", "d.safetensors",
            "--dit-low", "low.safetensors",
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
        dit_low_path="low.safetensors",
    )


def test_serve_without_model_does_not_load(monkeypatch, fake_runtime):
    monkeypatch.setattr(
        "thenoise.api.create_app",
        lambda runtime, *, jobs=None: object(),
    )
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


@pytest.mark.parametrize("name,value", [("width", 5000), ("height", -1), ("width", 0)])
def test_generate_rejects_out_of_range_dimensions(capsys, name, value):
    """A dimension is a size, not a flag: 0 is rejected, omission means "auto".
    The pipeline resolves an omitted field from the model, so 0 would mean 0 pixels.
    """
    argv = ["generate", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x"]
    argv += [f"--{name}", str(value)]
    with pytest.raises(SystemExit):
        run_generate(_parse(argv))
    assert f"error: {name} must be between 1 and 4096" in capsys.readouterr().err


def test_generate_refuses_an_unknown_sampler_before_anything_loads(capsys, fake_runtime):
    """A solver name needs no model to check, so it is refused before one is built."""
    argv = ["generate", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x"]
    with pytest.raises(SystemExit):
        run_generate(_parse([*argv, "--sampler", "dpmpp"]))
    err = capsys.readouterr().err
    assert "error: unknown sampler: 'dpmpp' (choose" in err
    assert fake_runtime == []  # no Runtime was even built

    run_generate(_parse([*argv, "--sampler", "er_sde"]))
    assert len(fake_runtime) == 1


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


# ------------------------------------------------------------------- video


VIDEO_PATHS = ["--dit", "high", "--vae", "v", "--text-encoder", "t"]


@pytest.fixture
def video_runtime(fake_runtime, monkeypatch):
    """A fake runtime whose loaded model makes clips."""
    monkeypatch.setattr(_FakeRuntime, "video_on_load", True)
    return fake_runtime


def test_video_refuses_an_unknown_sampler_before_anything_loads(video_runtime, tmp_path):
    with pytest.raises(SystemExit, match="unknown sampler: 'dpmpp'"):
        run_video(_parse(["video", *VIDEO_PATHS, "--prompt", "fox",
                          "--sampler", "dpmpp", "--out", str(tmp_path / "fox")]))
    assert video_runtime == []  # no Runtime was even built


def test_video_wires_paths_frames_and_lora_roles(video_runtime, tmp_path):
    first, last = tmp_path / "first.png", tmp_path / "last.png"
    first.write_bytes(b"first bytes")
    last.write_bytes(b"last bytes")
    run_video(_parse([
        "video", *VIDEO_PATHS, "--dit-low", "low", "--lora-dir", "loras",
        "--device", "hip",
        "--prompt", "a fox turns into a cat", "--negative-prompt", "blur",
        "--width", "704", "--height", "480", "--frames", "9", "--fps", "24",
        "--steps", "8", "--guidance-scale", "3.5", "--flow-shift", "5",
        "--sampler", "unipc", "--seed", "7",
        "--image", str(first), "--end-image", str(last),
        "--lora", "high_lora:0.8", "--lora-low", "low_lora",
        "--out", str(tmp_path / "fox"),
    ]))
    rt = video_runtime[0]
    assert rt.settings.device == "hip"
    assert rt.loaded == ModelPaths(dit_path="high", vae_path="v",
                                   text_encoder_path="t", lora_dir="loras",
                                   dit_low_path="low")

    spec = rt.pipeline.spec
    assert (spec.image, spec.end_image) == (b"first bytes", b"last bytes")
    assert spec.lora_specs == {"dit": ["high_lora:0.8"], "dit_low": ["low_lora"]}
    assert (spec.frames, spec.fps, spec.steps, spec.sampler, spec.seed) == (
        9, 24, 8, "unipc", 7)
    assert (spec.width, spec.height, spec.flow_shift) == (704, 480, 5.0)

    rendered = rt.pipeline.rendered
    assert rendered["spec"].mode == "flf2v"
    assert rendered["spec"].frame_count == 9
    assert isinstance(rendered["token"], Uncancellable)
    # A bare `--out` takes the container the run settled on.
    assert rendered["output"] == tmp_path / "fox.mp4"
    assert rendered["output"].read_bytes() == b"mp4"


def test_video_takes_its_container_from_the_name_it_was_given(video_runtime, tmp_path):
    run_video(_parse(["video", *VIDEO_PATHS, "--prompt", "x",
                      "--out", str(tmp_path / "fox.webm")]))
    pipeline = video_runtime[0].pipeline
    assert pipeline.spec.output_format == "webm"
    assert pipeline.rendered["output"].suffix == ".webm"


def test_video_refuses_a_name_promising_another_container(video_runtime, tmp_path):
    with pytest.raises(SystemExit, match=r"ends in \.mp4 but this clip is webm"):
        run_video(_parse(["video", *VIDEO_PATHS, "--prompt", "x",
                          "--output-format", "webm",
                          "--out", str(tmp_path / "fox.mp4")]))


def test_video_refuses_a_model_that_renders_stills(fake_runtime):
    with pytest.raises(SystemExit, match="renders stills"):
        run_video(_parse(["video", *VIDEO_PATHS, "--prompt", "x"]))


def test_video_refuses_a_last_frame_with_no_first_one(video_runtime):
    """Checked off the flags, so finding it out costs no model load."""
    with pytest.raises(SystemExit, match=r"--end-image needs --image"):
        run_video(_parse(["video", *VIDEO_PATHS, "--prompt", "x",
                          "--end-image", "last.png"]))
    assert video_runtime == []


def test_video_reports_what_the_loaded_variant_refuses(video_runtime, monkeypatch):
    """A request the variant cannot serve is one line, not a trace."""

    def refuse(self, spec):
        raise ValueError("this variant has no channel for a last one")

    monkeypatch.setattr(_FakeVideo, "resolve", refuse)
    with pytest.raises(SystemExit, match="no channel for a last one"):
        run_video(_parse(["video", *VIDEO_PATHS, "--prompt", "x"]))


def test_video_names_the_flag_of_a_frame_it_cannot_read(video_runtime):
    with pytest.raises(SystemExit, match=r"--image.*nowhere\.png"):
        run_video(_parse(["video", *VIDEO_PATHS, "--prompt", "x",
                          "--image", "nowhere.png"]))


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


# ---------------------------------------------------------------- controlmap


def test_controlmap_is_model_free_and_maps_the_flags(fake_runtime, tmp_path):
    img_path = tmp_path / "in.png"
    Image.new("RGB", (64, 64)).save(img_path)
    run_controlmap(
        _parse(
            [
                "controlmap",
                "--input", str(img_path),
                "--type", "canny",
                "--canny-low", "30",
                "--canny-high", "60",
                "--canny-thickness", "2",
                "--out", "map",
                "--device", "cpu",
            ]
        )
    )
    rt = fake_runtime[0]
    assert rt.settings.device == "cpu"
    assert rt.loaded is None  # model-free: no load() call

    image, kind, options, name = rt.controlmap.calls[0]
    assert (image.size, kind, name) == ((64, 64), "canny", "")
    # The kind's own flags lose their namespace on the way in.
    assert (options.low, options.high, options.thickness) == (30.0, 60.0, 2)
    assert rt.controlmap.images[0].saved_path == ensure_png_extension("map")


def test_each_type_gets_its_own_file(fake_runtime, tmp_path):
    """With more than one map requested every output is named after its type."""
    img_path = tmp_path / "in.png"
    Image.new("RGB", (64, 64)).save(img_path)
    run_controlmap(
        _parse(["controlmap", "--input", str(img_path), "--type", "canny",
                "--type", "depth", "--out", "map"])
    )
    assert [i.saved_path for i in fake_runtime[0].controlmap.images] == [
        "map_canny.png",
        "map_depth.png",
    ]


def test_controlmap_list_reports_the_available_types(fake_runtime, caplog):
    with caplog.at_level(logging.INFO):
        run_controlmap(_parse(["controlmap", "--list"]))
    assert fake_runtime[0].loaded is None
    assert "canny: canny" in caplog.text


def test_controlmap_list_says_what_is_missing(fake_runtime, caplog):
    """A kind the engine has but whose weights have not been downloaded is worth naming:
    an empty list alone leaves the user guessing whether the feature exists at all."""
    with caplog.at_level(logging.INFO):
        run_controlmap(_parse(["controlmap", "--list"]))
    assert "pose: unavailable, has no checkpoint" in caplog.text
    assert "scripts/download.py --model controlmaps --variant pose" in caplog.text


def test_controlmap_without_input_or_type_explains_itself(fake_runtime):
    with pytest.raises(SystemExit, match="--input and --type are required"):
        run_controlmap(_parse(["controlmap"]))


def test_controlmap_reports_an_unknown_type_as_an_exit(fake_runtime, tmp_path):
    img_path = tmp_path / "in.png"
    Image.new("RGB", (64, 64)).save(img_path)
    with pytest.raises(SystemExit, match="no control map for 'normals'"):
        run_controlmap(_parse(["controlmap", "--input", str(img_path), "--type", "normals"]))


# ---------------------------------------------------------------- dispatch


def test_main_dispatches_to_the_correct_handler(monkeypatch):
    calls = []
    monkeypatch.setattr("thenoise.generate.run_generate", lambda args: calls.append("generate"))
    monkeypatch.setattr("thenoise.generate.run_edit", lambda args: calls.append("edit"))
    monkeypatch.setattr("thenoise.video_cli.run_video", lambda args: calls.append("video"))
    monkeypatch.setattr("thenoise.upscale_cli.run_upscale", lambda args: calls.append("upscale"))
    monkeypatch.setattr(
        "thenoise.controlmap_cli.run_controlmap", lambda args: calls.append("controlmap")
    )
    monkeypatch.setattr("thenoise.__main__._serve", lambda args: calls.append("serve"))

    main(["generate", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x"])
    main(["edit", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x", "--image", "i.png"])
    main(["video", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x"])
    main(["upscale", "--pixel-upscaler", "u.safetensors", "--input", "i.png"])
    main(["controlmap", "--input", "i.png", "--type", "canny"])
    main(["serve"])
    assert calls == ["generate", "edit", "video", "upscale", "controlmap", "serve"]
