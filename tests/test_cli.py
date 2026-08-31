"""CLI parsing tests (no torch / ROCm required, no real weights)."""
from __future__ import annotations

import pytest

from thenoise.cli import build_parser


def test_parse_lora_spec_no_suffix():
    """_parse_lora_spec auto-appends .safetensors when no extension."""
    # We test the logic directly without instantiating a model
    from thenoise.models.base import DiffusionModel
    # Create a minimal concrete subclass for testing
    class _TestModel(DiffusionModel):
        name = "test"
        @staticmethod
        def detect(f): return False
        def encode_prompt(self, prompt, negative_prompt="", *, guidance_scale): pass
        def init_latents(self, height, width, seed): pass
        def schedule(self, steps, height, width): pass
        def denoise_step(self, latents, t, cond, guidance_scale, i): pass
        def _upscale_format(self): return "wan21"

    # Can't fully instantiate (needs VAE), but we can test the pure parsing
    # by calling the method directly with a mock object
    import types
    model = object.__new__(_TestModel)
    model.lora_dir = "/tmp/loras"

    # No extension → auto-append
    filename, weight = model._parse_lora_spec("style")
    assert filename == "style.safetensors"
    assert weight == 1.0

    # No extension with weight
    filename, weight = model._parse_lora_spec("style:0.8")
    assert filename == "style.safetensors"
    assert weight == 0.8

    # Subdirectory allowed
    filename, weight = model._parse_lora_spec("sub/style:0.7")
    assert filename == "sub/style.safetensors"
    assert weight == 0.7


def test_resolve_lora_path_blocks_traversal():
    """_resolve_lora_path rejects .. escape attempts."""
    import tempfile, os
    from thenoise.models.base import DiffusionModel

    class _TestModel(DiffusionModel):
        name = "test"
        @staticmethod
        def detect(f): return False
        def encode_prompt(self, prompt, negative_prompt="", *, guidance_scale): pass
        def init_latents(self, height, width, seed): pass
        def schedule(self, steps, height, width): pass
        def denoise_step(self, latents, t, cond, guidance_scale, i): pass
        def _upscale_format(self): return "wan21"

    model = object.__new__(_TestModel)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.lora_dir = tmpdir

        # Normal file → OK
        path = model._resolve_lora_path("style.safetensors")
        assert path == os.path.join(tmpdir, "style.safetensors")

        # Subdirectory → OK
        path = model._resolve_lora_path("sub/style.safetensors")
        assert "sub/style.safetensors" in path

        # .. escape → ValueError
        with pytest.raises(ValueError, match="escapes base directory"):
            model._resolve_lora_path("../etc/passwd")

        with pytest.raises(ValueError, match="escapes base directory"):
            model._resolve_lora_path("sub/../../etc/passwd")


def test_lora_spec_hash():
    """_make_lora_spec_hash produces consistent hashes."""
    from thenoise.models.base import DiffusionModel

    class _TestModel(DiffusionModel):
        name = "test"
        @staticmethod
        def detect(f): return False
        def encode_prompt(self, prompt, negative_prompt="", *, guidance_scale): pass
        def init_latents(self, height, width, seed): pass
        def schedule(self, steps, height, width): pass
        def denoise_step(self, latents, t, cond, guidance_scale, i): pass
        def _upscale_format(self): return "wan21"

    model = object.__new__(_TestModel)
    model.lora_dir = "/tmp/loras"

    assert model._make_lora_spec_hash(None) == "__none__"
    assert model._make_lora_spec_hash([]) == "__none__"

    h1 = model._make_lora_spec_hash(["a:0.5", "b:1.0"])
    h2 = model._make_lora_spec_hash(["b:1.0", "a:0.5"])  # different order
    assert h1 == h2  # sorted, so same hash


def test_list_loras_returns_short_names_recursive():
    """list_loras strips .safetensors and scans subdirectories."""
    import tempfile, os
    from thenoise.models.base import DiffusionModel

    class _TestModel(DiffusionModel):
        name = "test"
        @staticmethod
        def detect(f): return False
        def encode_prompt(self, prompt, negative_prompt="", *, guidance_scale): pass
        def init_latents(self, height, width, seed): pass
        def schedule(self, steps, height, width): pass
        def denoise_step(self, latents, t, cond, guidance_scale, i): pass
        def _upscale_format(self): return "wan21"

    model = object.__new__(_TestModel)

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "sub"))
        for rel in ["12345_something.safetensors",
                    "67890_other.safetensors",
                    "sub/style.safetensors",
                    "not_a_lora.txt"]:
            with open(os.path.join(tmpdir, rel), "w") as f:
                f.write("x")

        model.lora_dir = tmpdir
        assert model.list_loras() == [
            "12345_something",
            "67890_other",
            "sub/style",
        ]

        model.lora_dir = ""
        assert model.list_loras() == []


def test_cli_serve_parses_model_paths():
    args = build_parser().parse_args([
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
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.device == "hip"


def test_cli_serve_has_no_pixel_upscaler():
    args = build_parser().parse_args([
        "serve",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
    ])
    assert args.upscaler_dir == ""
    assert not hasattr(args, "pixel_upscaler")


def test_cli_generate_parses_pixel_upscaler_and_type():
    args = build_parser().parse_args([
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
    assert not hasattr(args, "upscaler_dir")


def test_cli_generate_rejects_fast_and_old_flags():
    # 'fast' type and '--esrgan' are removed.
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "generate",
            "--dit", "d.safetensors",
            "--vae", "v.safetensors",
            "--text-encoder", "te.safetensors",
            "--prompt", "a fox", "--upscale-type", "fast",
        ])
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "generate",
            "--dit", "d.safetensors",
            "--vae", "v.safetensors",
            "--text-encoder", "te.safetensors",
            "--prompt", "a fox", "--esrgan", "/models/x.safetensors",
        ])


def test_cli_serve_defaults():
    args = build_parser().parse_args([
        "serve",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
    ])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.device == "cuda"
    assert args.upscaler_dir == ""


def test_cli_serve_requires_paths():
    # A bare `serve` parses cleanly (model-free); missing model paths are
    # rejected by ``resolve_model_paths`` instead.
    args = build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.dit is None
    assert args.vae is None
    assert args.text_encoder is None
    from thenoise.cli import resolve_model_paths

    with pytest.raises(SystemExit):
        resolve_model_paths(args)


def test_cli_serve_partial_paths_parse():
    """Partial model paths still parse at the CLI level (validated at runtime)."""
    args = build_parser().parse_args(["serve", "--dit", "d.safetensors"])
    assert args.dit == "d.safetensors"
    assert args.vae is None


def test_cli_resolve_checkpoint_path():
    from thenoise.cli import resolve_model_paths

    args = build_parser().parse_args([
        "generate", "--checkpoint", "mix.safetensors",
        "--prompt", "a fox",
    ])
    paths = resolve_model_paths(args)
    assert paths["checkpoint_path"] == "mix.safetensors"
    assert "dit_path" not in paths  # split trio unused


def test_cli_checkpoint_conflicts_with_trio():
    from thenoise.cli import resolve_model_paths

    args = build_parser().parse_args([
        "generate", "--checkpoint", "mix.safetensors",
        "--dit", "d.safetensors", "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors", "--prompt", "a fox",
    ])
    with pytest.raises(SystemExit):
        resolve_model_paths(args)


def test_cli_generate_parses():
    args = build_parser().parse_args([
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


def test_cli_generate_parses_lora():
    args = build_parser().parse_args([
        "generate",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
        "--prompt", "a fox",
        "--lora", "style.safetensors:0.8",
        "--lora", "pose.safetensors:1.0",
    ])
    assert args.lora == ["style.safetensors:0.8", "pose.safetensors:1.0"]


def test_cli_edit_parses_required_image_and_width_height():
    args = build_parser().parse_args([
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
    assert args.width == 1024
    assert args.height == 512
    assert args.prompt == "make it sunny"
    assert args.out == "e.png"
    assert args.seed == 9


def test_cli_edit_defaults_out_and_width_height():
    args = build_parser().parse_args([
        "edit",
        "--dit", "d.safetensors",
        "--vae", "v.safetensors",
        "--text-encoder", "te.safetensors",
        "--prompt", "x",
        "--image", "in.png",
    ])
    assert args.out == "out_edit.png"
    assert args.width is None
    assert args.height is None


def test_cli_edit_requires_image():
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "edit",
            "--dit", "d.safetensors",
            "--vae", "v.safetensors",
            "--text-encoder", "te.safetensors",
            "--prompt", "x",
        ])


def test_out_defaults_to_png_when_no_extension():
    """A bare --out with no extension gets .png appended before save."""
    from thenoise.utils.paths import ensure_png_extension
    assert ensure_png_extension("out") == "out.png"
    assert ensure_png_extension("dir/out") == "dir/out.png"
    assert ensure_png_extension("out.png") == "out.png"
    assert ensure_png_extension("out.jpg") == "out.jpg"
    assert ensure_png_extension("out.tar.gz") == "out.tar.gz"


def test_cli_rejects_unknown_flags():
    # --model and --dtype are gone: everything is auto-detected / fixed bf16.
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "serve", "--model", "krea2",
            "--dit", "d.safetensors",
            "--vae", "v.safetensors",
            "--text-encoder", "te.safetensors",
        ])


def test_cli_upscale_parses():
    args = build_parser().parse_args([
        "upscale",
        "--pixel-upscaler", "/models/RealESRGAN_x4.safetensors",
        "--input", "in.png",
        "--upscale-factor", "4",
        "--out", "out.png",
        "--device", "hip",
    ])
    assert args.command == "upscale"
    assert args.pixel_upscaler == "/models/RealESRGAN_x4.safetensors"
    assert args.input == "in.png"
    assert args.upscale_factor == 4
    assert args.out == "out.png"
    assert args.device == "hip"


def test_cli_upscale_defaults_and_is_model_free():
    args = build_parser().parse_args([
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


def test_cli_upscale_requires_pixel_upscaler_and_input():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["upscale"])
