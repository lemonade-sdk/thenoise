"""CLI parsing tests (no torch / ROCm required, no real weights).

Only argparse itself is exercised here: the arg -> ``Settings``/``ModelPaths``/
``Runtime`` wiring is covered in ``test_entrypoints.py``, and the LoRA spec/path
helpers live in ``test_lora.py``.
"""
from __future__ import annotations

import pytest

from thenoise.cli import build_parser

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
        "--host", "0.0.0.0", "--port", "9000", "--device", "hip",
    ])
    assert args.command == "serve"
    assert args.dit == "d.safetensors"
    assert args.lora_dir == "/path/to/loras"
    assert args.upscaler_dir == "/path/to/upscalers"
    assert (args.host, args.port, args.device) == ("0.0.0.0", 9000, "hip")

    # A bare ``serve`` still parses: the server runs model-free (upscale only).
    bare = parse(["serve"])
    assert (bare.host, bare.port, bare.device) == ("127.0.0.1", 8000, "cuda")
    assert bare.dit is None and bare.vae is None and bare.text_encoder is None
    assert bare.upscaler_dir == "" and bare.offload_device == ""
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
