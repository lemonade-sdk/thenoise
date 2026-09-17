"""CLI parsing tests (no torch / ROCm required, no real weights).

Only argparse itself is exercised here: the arg -> ``Settings``/``ModelPaths``/
``Runtime`` wiring is covered in ``test_entrypoints.py``, and the LoRA spec/path
helpers live in ``test_lora.py``.
"""
from __future__ import annotations

import pytest

from thenoise.cli import build_parser


@pytest.fixture
def parse():
    return lambda argv: build_parser().parse_args(argv)


# ------------------------------------------------------------------ serve


def test_cli_serve_parses_model_paths(parse):
    args = parse([
        "serve",
        "--dit", "dit.safetensors",
        "--vae", "vae.safetensors",
        "--text-encoder", "te.safetensors",
        "--lora-dir", "/path/to/loras",
        "--upscaler-dir", "/path/to/upscalers",
        "--host", "0.0.0.0", "--port", "9000", "--device", "hip",
    ])
    assert args.command == "serve"
    assert args.dit == "dit.safetensors"
    assert args.lora_dir == "/path/to/loras"
    assert args.upscaler_dir == "/path/to/upscalers"
    assert (args.host, args.port, args.device) == ("0.0.0.0", 9000, "hip")


def test_cli_serve_defaults(parse):
    args = parse([
        "serve",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
    ])
    assert (args.host, args.port, args.device) == ("127.0.0.1", 8000, "cuda")
    assert args.upscaler_dir == ""
    assert args.offload_device == ""
    # A one-shot server has no pixel-upscaler flag (it takes a directory instead).
    assert not hasattr(args, "pixel_upscaler")


def test_cli_serve_model_paths_are_optional(parse):
    """A bare ``serve`` must parse: the server runs model-free (upscale only)."""
    args = parse(["serve"])
    assert args.command == "serve"
    assert args.dit is None and args.vae is None and args.text_encoder is None


def test_cli_rejects_removed_flags(parse):
    """``--model`` and ``--dtype`` are gone: everything is auto-detected / bf16."""
    with pytest.raises(SystemExit):
        parse(["serve", "--model", "krea2"])


# --------------------------------------------------------------- generate/edit


def test_cli_generate_parses(parse):
    args = parse([
        "generate",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
        "--prompt", "a fox", "--steps", "30", "--seed", "7", "--out", "x.png",
    ])
    assert args.command == "generate"
    assert args.prompt == "a fox"
    assert args.steps == 30
    assert args.seed == 7
    assert args.out == "x.png"
    assert args.device == "cuda"


def test_cli_generate_parses_repeated_loras(parse):
    args = parse([
        "generate",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
        "--prompt", "a fox",
        "--lora", "style.safetensors:0.8",
        "--lora", "pose.safetensors:1.0",
    ])
    assert args.lora == ["style.safetensors:0.8", "pose.safetensors:1.0"]


def test_cli_generate_parses_one_shot_pixel_upscaler_and_type(parse):
    args = parse([
        "generate",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
        "--prompt", "a fox",
        "--pixel-upscaler", "/models/RealESRGAN_x4.safetensors",
        "--upscale-type", "no-refiner",
    ])
    assert args.pixel_upscaler == "/models/RealESRGAN_x4.safetensors"
    assert args.upscale_type == "no-refiner"
    # ``generate`` takes a one-shot path, not a server directory.
    assert not hasattr(args, "upscaler_dir")


def test_cli_generate_rejects_removed_types_and_flags(parse):
    """The 'fast' upscale type and '--esrgan' are removed."""
    base = ["generate", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x"]
    with pytest.raises(SystemExit):
        parse(base + ["--upscale-type", "fast"])
    with pytest.raises(SystemExit):
        parse(base + ["--esrgan", "/models/x.safetensors"])


def test_cli_edit_parses_required_image_and_size(parse):
    args = parse([
        "edit",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
        "--prompt", "make it sunny",
        "--image", "in.png",
        "--width", "1024",
        "--height", "512",
        "--out", "e.png",
        "--seed", "9",
    ])
    assert args.command == "edit"
    assert args.image == ["in.png"]
    assert (args.width, args.height) == (1024, 512)
    assert args.prompt == "make it sunny"
    assert (args.out, args.seed) == ("e.png", 9)


def test_cli_edit_defaults_out_and_leaves_size_unset(parse):
    """No size on the CLI -> the pipeline derives it from the image."""
    args = parse([
        "edit",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
        "--prompt", "x",
        "--image", "in.png",
    ])
    assert args.out == "out_edit.png"
    assert args.width is None and args.height is None


def test_cli_edit_requires_an_image(parse):
    with pytest.raises(SystemExit):
        parse([
            "edit",
            "--dit", "d.safetensors",
            "--vae", "v.safetensors",
            "--text-encoder", "te.safetensors",
            "--prompt", "x",
        ])


def test_cli_edit_accepts_repeated_images(parse):
    args = parse([
        "edit",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
        "--prompt", "x",
        "--image", "a.png",
        "--image", "b.png",
    ])
    assert args.image == ["a.png", "b.png"]


# -------------------------------------------------------------------- upscale


def test_cli_upscale_parses(parse):
    args = parse([
        "upscale",
        "--pixel-upscaler", "/models/RealESRGAN_x4.safetensors",
        "--input", "in.png",
        "--upscale-factor", "4",
        "--out", "out.png",
        "--device", "hip",
    ])
    assert args.command == "upscale"
    assert args.pixel_upscaler == "/models/RealESRGAN_x4.safetensors"
    assert (args.input, args.upscale_factor, args.out, args.device) == (
        "in.png", 4, "out.png", "hip",
    )


def test_cli_upscale_defaults_and_is_model_free(parse):
    args = parse([
        "upscale",
        "--pixel-upscaler", "/models/x.safetensors",
        "--input", "in.png",
    ])
    assert args.upscale_factor == 0.0  # 0.0 sentinel -> detected model scale
    assert args.out == "out_upscaled.png"
    assert args.device == "cuda"
    # model-free: no checkpoint or prompt flags on the upscale subcommand
    assert not hasattr(args, "dit")
    assert not hasattr(args, "prompt")


def test_cli_upscale_requires_pixel_upscaler_and_input(parse):
    with pytest.raises(SystemExit):
        parse(["upscale"])


def test_cli_requires_a_subcommand(parse):
    with pytest.raises(SystemExit):
        parse([])


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
