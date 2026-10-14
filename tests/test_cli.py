"""CLI parsing tests (no torch / ROCm required, no real weights).

Only argparse itself is exercised here: the arg -> ``Settings``/``ModelPaths``/
``Runtime`` wiring is covered in ``test_entrypoints.py``, and the LoRA spec/path
helpers live in ``test_lora.py``.
"""
from __future__ import annotations

import pytest

from thenoise.cli import build_parser
from thenoise.controlmap.options import OPTIONS_BY_KIND

# The checkpoints every generation subcommand needs (``upscale`` is model-free).
PATHS = ["--dit", "d.safetensors", "--vae", "v.safetensors", "--text-encoder", "t.safetensors"]


@pytest.fixture
def parse():
    return lambda argv: build_parser().parse_args(argv)


# ------------------------------------------------------------------ serve


def test_cli_serve_parses_model_paths_and_defaults(parse):
    args = parse([
        "serve", *PATHS,
        "--lora-dir", "/path/to/loras",
        "--upscaler-dir", "/path/to/upscalers",
        "--controlmap-dir", "/path/to/controlmaps",
        "--host", "0.0.0.0", "--port", "9000", "--device", "hip",
    ])
    assert args.command == "serve"
    assert args.dit == "d.safetensors"
    assert args.lora_dir == "/path/to/loras"
    assert args.upscaler_dir == "/path/to/upscalers"
    assert args.controlmap_dir == "/path/to/controlmaps"
    assert (args.host, args.port, args.device) == ("0.0.0.0", 9000, "hip")

    # A bare ``serve`` still parses: the server runs model-free (upscale only).
    bare = parse(["serve"])
    assert (bare.host, bare.port, bare.device) == ("127.0.0.1", 8000, "cuda")
    assert bare.dit is None and bare.vae is None and bare.text_encoder is None
    assert bare.upscaler_dir == "" and bare.offload_device == ""
    assert bare.controlmap_dir == ""
    # A one-shot server has no pixel-upscaler flag (it takes a directory instead).
    assert not hasattr(bare, "pixel_upscaler")


def test_cli_rejects_removed_flags(parse):
    """``--model`` and ``--dtype`` are gone: everything is auto-detected / bf16."""
    with pytest.raises(SystemExit):
        parse(["serve", "--model", "krea2"])


# --------------------------------------------------------------- generate/edit


def test_cli_generate_parses(parse):
    args = parse([
        "generate", *PATHS,
        "--prompt", "a fox", "--steps", "30", "--seed", "7", "--out", "x.png",
        "--lora", "style.safetensors:0.8",
        "--lora", "pose.safetensors:1.0",
        "--pixel-upscaler", "/models/RealESRGAN_x4.safetensors",
        "--upscale-type", "no-refiner",
    ])
    assert args.command == "generate"
    assert args.prompt == "a fox"
    assert (args.steps, args.seed, args.device) == (30, 7, "cuda")
    assert args.out == "x.png"
    assert args.lora == ["style.safetensors:0.8", "pose.safetensors:1.0"]
    assert args.pixel_upscaler == "/models/RealESRGAN_x4.safetensors"
    assert args.upscale_type == "no-refiner"
    # ``generate`` takes a one-shot path, not a server directory.
    assert not hasattr(args, "upscaler_dir")


def test_cli_generate_rejects_removed_types_and_flags(parse):
    """The 'fast' upscale type and ``--esrgan`` are removed."""
    base = ["generate", *PATHS, "--prompt", "x"]
    with pytest.raises(SystemExit):
        parse(base + ["--upscale-type", "fast"])
    with pytest.raises(SystemExit):
        parse(base + ["--esrgan", "/models/x.safetensors"])


def test_cli_edit_parses_images_size_and_kv_cache(parse):
    args = parse([
        "edit", *PATHS,
        "--prompt", "make it sunny",
        "--image", "a.png", "--image", "b.png",
        "--width", "1024", "--height", "512",
        "--out", "e.png", "--seed", "9",
        "--kv-cache", "--ref-method", "index_timestep_zero",
    ])
    assert args.command == "edit"
    assert args.image == ["a.png", "b.png"]
    assert (args.width, args.height) == (1024, 512)
    assert args.prompt == "make it sunny"
    assert (args.out, args.seed) == ("e.png", 9)
    assert args.kv_cache is True
    assert args.ref_method == "index_timestep_zero"


def test_cli_edit_defaults(parse):
    """No size on the CLI -> the pipeline derives it from the image. ``kv_cache`` is
    a tri-state: both ``--kv-cache`` and ``--no-kv-cache`` override a ``None`` default.
    """
    args = parse(["edit", *PATHS, "--prompt", "x", "--image", "in.png"])
    assert args.out == "out_edit.png"
    assert args.width is None and args.height is None
    assert args.kv_cache is None and args.ref_method is None

    assert parse(["edit", *PATHS, "--prompt", "x", "--image", "i.png",
                  "--no-kv-cache"]).kv_cache is False


def test_cli_edit_requires_an_image(parse):
    with pytest.raises(SystemExit):
        parse(["edit", *PATHS, "--prompt", "x"])


# -------------------------------------------------------------------- upscale


def test_cli_upscale_parses_and_is_model_free(parse):
    args = parse([
        "upscale",
        "--pixel-upscaler", "/models/RealESRGAN_x4.safetensors",
        "--input", "in.png", "--upscale-factor", "4", "--out", "out.png",
        "--device", "hip",
    ])
    assert args.command == "upscale"
    assert args.pixel_upscaler == "/models/RealESRGAN_x4.safetensors"
    assert (args.input, args.upscale_factor, args.out, args.device) == (
        "in.png", 4, "out.png", "hip",
    )

    defaults = parse(["upscale", "--pixel-upscaler", "/models/x.safetensors", "--input", "in.png"])
    assert defaults.upscale_factor == 0.0  # 0.0 sentinel -> detected model scale
    assert defaults.out == "out_upscaled.png"
    assert defaults.device == "cuda"
    # model-free: no checkpoint or prompt flags on the upscale subcommand
    assert not hasattr(defaults, "dit") and not hasattr(defaults, "prompt")


def test_cli_requires_the_subcommand_arguments(parse):
    with pytest.raises(SystemExit):
        parse([])                      # no subcommand at all
    with pytest.raises(SystemExit):
        parse(["upscale"])             # --pixel-upscaler/--input are required


# ---------------------------------------------------------------- controlmap


def test_cli_controlmap_parses_its_flags(parse):
    args = parse([
        "controlmap",
        "--input", "in.png", "--type", "canny", "--type", "depth",
        "--model", "depth-anything-v2-base",
        "--canny-low", "30", "--canny-high", "60", "--canny-thickness", "2",
        "--depth-invert", "--depth-colormap", "turbo",
        "--lineart-threshold", "0.7", "--lineart-thickness", "3",
        "--softedge-threshold", "0.1", "--softedge-blur", "2.0",
        "--pose-detector", "rtdetr_r50vd", "--pose-threshold", "0.7",
        "--pose-kp-threshold", "0.4", "--pose-line-width", "6",
        "--pose-radius", "3", "--pose-background", "green",
        "--controlmap-dir", "models/controlmaps",
        "--out", "map.png", "--device", "hip",
    ])
    assert args.command == "controlmap"
    assert (args.input, args.types) == ("in.png", ["canny", "depth"])
    assert args.model == "depth-anything-v2-base"
    assert (args.canny_low, args.canny_high, args.canny_thickness) == (30.0, 60.0, 2)
    assert (args.depth_invert, args.depth_colormap) == (True, "turbo")
    assert (args.lineart_threshold, args.lineart_thickness) == (0.7, 3)
    assert (args.softedge_threshold, args.softedge_blur) == (0.1, 2.0)
    assert (args.pose_detector, args.pose_threshold, args.pose_kp_threshold) == (
        "rtdetr_r50vd", 0.7, 0.4,
    )
    assert (args.pose_line_width, args.pose_radius, args.pose_background) == (
        6, 3, "green",
    )
    assert args.controlmap_dir == "models/controlmaps"
    assert (args.out, args.device, args.list) == ("map.png", "hip", False)


def test_cli_controlmap_defaults(parse):
    """A bare ``controlmap`` parses: ``--list`` needs no input and no type."""
    args = parse(["controlmap"])
    assert (args.input, args.types, args.model) == (None, None, "")
    assert args.out == "out_control.png" and args.list is False
    # No flag passed, no option in the namespace: the dataclass owns the default.
    assert not [name for name in vars(args) if name.split("_")[0] in OPTIONS_BY_KIND]


def test_cli_controlmap_knows_only_the_canvases_we_can_paint(parse):
    with pytest.raises(SystemExit):
        parse(["controlmap", "--input", "i.png", "--type", "pose",
               "--pose-background", "plaid"])


@pytest.mark.parametrize("flag", ["--width", "--height", "--fit"])
def test_a_control_map_has_no_sizing_flags(parse, flag):
    """The map comes out its input's size, so asking for one is not offered."""
    with pytest.raises(SystemExit):
        parse(["controlmap", "--input", "i.png", "--type", "canny", flag, "512"])


# ----------------------------------------------------------------------- paths


@pytest.mark.parametrize(
    "out,expected",
    [
        ("out", "out.png"),          # PIL cannot infer a format from a bare name
        ("dir/out", "dir/out.png"),
        ("out.png", "out.png"),
        ("out.jpg", "out.jpg"),      # an explicit format is respected
        ("out.tar.gz", "out.tar.gz"),
    ],
)
def test_out_defaults_to_png_when_no_extension(out, expected):
    from thenoise.utils.paths import ensure_png_extension

    assert ensure_png_extension(out) == expected
