"""Command-line interface for thenoise.

Four subcommands:
  * ``serve``    run the FastAPI HTTP server with a single loaded model
  * ``generate`` run one generation and save the PNG
  * ``edit``     edit an image from an instruction (image + prompt -> edited image)
  * ``upscale``  pixel-upscale an image (no diffusion model needed)

Model checkpoints are supplied to ``serve``/``generate``/``edit`` (``--dit`` /
``--vae`` / ``--text-encoder``), and the model type is detected automatically from
the ``--dit`` checkpoint. ``upscale`` needs no diffusion model. All options are
passed on the command line (there is no config file).
"""
from __future__ import annotations

import argparse
from typing import Optional


def _add_model_paths(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dit", metavar="PATH",
                   help="DiT checkpoint (.safetensors); omit when using --checkpoint")
    p.add_argument("--vae", metavar="PATH",
                   help="VAE checkpoint (.safetensors); omit when using --checkpoint")
    p.add_argument("--text-encoder", metavar="PATH",
                   help="text encoder checkpoint (.safetensors); omit when using --checkpoint")
    p.add_argument("--checkpoint", metavar="PATH",
                   help="single combined SDXL/Illustrious checkpoint (.safetensors); "
                        "loads the DiT, VAE and text encoders from it directly, "
                        "no splitting needed")
    p.add_argument("--lora-dir", default="", metavar="PATH",
                   help="directory containing LoRA .safetensors files "
                        "(subdirectories allowed)")
    p.add_argument("--sd-zsnr", action="store_true",
                   help="force the zero-terminal-SNR (zsnr) schedule for an "
                        "SDXL/Illustrious checkpoint; auto-enabled when the "
                        "checkpoint carries the ztsnr marker, this flag is for "
                        "models whose marker was stripped")
    p.add_argument("--no-sd-zsnr", action="store_true",
                   help="disable the zsnr schedule even when the checkpoint "
                        "carries the ztsnr marker (overrides --sd-zsnr)")


def _resolve_sd_zsnr(args) -> Optional[bool]:
    """Tri-state zsnr override: None = auto-detect, True = force on, False = off.

    ``--no-sd-zsnr`` wins over ``--sd-zsnr`` so a marker-bearing checkpoint can
    be forced onto the plain linear schedule for debugging (e.g. isolating a
    zsnr vs v-prediction bug in a NoobAI checkpoint).
    """
    if args.no_sd_zsnr:
        return False
    if args.sd_zsnr:
        return True
    return None


def resolve_model_paths(args) -> dict:
    """Validate and assemble the model-path fields from parsed CLI args.

    Accepts either ``--checkpoint`` (a single combined SDXL file) OR the split
    ``--dit`` + ``--vae`` + ``--text-encoder`` trio; exactly one form must be
    given. Returns the keyword args for ``ModelPaths``.
    """
    if args.checkpoint:
        if args.dit or args.vae or args.text_encoder:
            raise SystemExit(
                "--checkpoint cannot be combined with --dit/--vae/--text-encoder"
            )
        return {"checkpoint_path": args.checkpoint, "lora_dir": args.lora_dir,
                "sd_zsnr": _resolve_sd_zsnr(args)}
    missing = [n for n, v in (("--dit", args.dit), ("--vae", args.vae),
                              ("--text-encoder", args.text_encoder)) if not v]
    if missing:
        raise SystemExit(
            "missing model checkpoint: provide --checkpoint, or "
            f"--dit/--vae/--text-encoder (missing: {', '.join(missing)})"
        )
    return {
        "dit_path": args.dit,
        "vae_path": args.vae,
        "text_encoder_path": args.text_encoder,
        "lora_dir": args.lora_dir,
        "sd_zsnr": _resolve_sd_zsnr(args),
    }


def _add_upscaler_args(p: argparse.ArgumentParser) -> None:
    """Add the pixel-upscaler flags for a subcommand.

    ``serve`` exposes ``--upscaler-dir`` (a directory, selected per-request via
    the ``pixel_upscaler`` API field). ``generate`` instead takes a one-shot
    ``--pixel-upscaler`` full path, which is split internally into
    ``upscaler_dir`` + ``pixel_upscaler`` before being passed down the chain.
    """
    p.add_argument("--upscaler-dir", default="", metavar="PATH",
                   help="directory containing pixel upscaler .safetensors files "
                        "(e.g. Real-ESRGAN); selected per-request via the "
                        "'pixel_upscaler' API field")


def _add_generation_args(p: argparse.ArgumentParser, out_default: str = "out.png") -> None:
    """Add the generation options shared by ``generate`` and ``edit``.
    """
    p.add_argument("--pixel-upscaler", default="", metavar="PATH",
                   help="full path to the pixel upscaler model (.safetensors) "
                        "to use for this one-shot generation (e.g. a Real-ESRGAN "
                        "model)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance-scale", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--out", default=out_default)
    p.add_argument("--lora", action="append", default=[],
                   metavar="FILE[:WEIGHT]",
                   help="LoRA to apply (format: 'style:0.8' or 'sub/style', "
                        ".safetensors auto-appended, repeatable)")
    p.add_argument("--upscale", action="store_true",
                   help="upscale the latent 2x in latent space (SesquiLSR) and "
                        "run a low-strength refine denoise before decoding")
    p.add_argument("--upscale-factor", type=float, default=1.0,
                   help="upscale factor, > 0.0 (default: 1.0 = no upscale); "
                        "max depends on the pixel upscaler scale: 'no-refiner' "
                        "is limited to the model scale, 'refined' to latent 2x * "
                        "model scale")
    p.add_argument("--upscale-type", choices=["refined", "no-refiner"],
                   default="refined",
                   help="'refined' (default): latent 2x + refiner, plus pixel "
                        "upscaler above factor 2; 'no-refiner': pixel upscaler "
                        "only (no latent 2x)")
    p.add_argument("--sampler", choices=["euler", "er_sde"], default=None,
                   help="denoising solver (default: auto)")
    p.add_argument("--qwen-vae-enhance", action="store_true",
                   help="apply the Nyquist Notch post filter to decoded pixels "
                        "(removes 2px grid artifacts)")
    p.add_argument("--film-grain", type=float, default=0.0,
                   help="film-grain strength 0.0–10.0. Reasonable values are <= 1.0 (default: 0.0)")
    p.add_argument("--sharpening", type=float, default=0.0,
                   help="RCAS sharpening strength 0.0–1.0 (contrast-adaptive, default: 0.0 / off)")
    p.add_argument("--device", default="cuda",
                   help="inference device; ROCm aliases cuda -> hip (default: cuda)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thenoise")
    sub = parser.add_subparsers(dest="command", required=True)

    # serve: model paths are optional — without them the server runs model-free
    # and only the pixel-upscale tab is usable.
    serve = sub.add_parser("serve", help="run the FastAPI HTTP server")
    _add_model_paths(serve)
    _add_upscaler_args(serve)
    serve.add_argument("--host", default="127.0.0.1",
                       help="bind host (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000,
                       help="bind port (default: 8000)")
    serve.add_argument("--device", default="cuda",
                       help="inference device; ROCm aliases cuda -> hip (default: cuda)")

    # generate
    gen = sub.add_parser("generate", help="run one generation and save a PNG")
    _add_model_paths(gen)
    _add_generation_args(gen)

    # edit
    edit = sub.add_parser("edit", help="edit an image from an instruction "
                                        "(image + prompt -> edited image)")
    _add_model_paths(edit)
    _add_generation_args(edit, out_default="out_edit.png")
    edit.add_argument("--image", action="append", required=True, metavar="PATH",
                      help="input image(s) to edit; repeatable for multiple "
                           "reference images (first sets the output size)")


    # upscale
    up = sub.add_parser("upscale", help="pixel-upscale an image (no model needed)")
    up.add_argument("--pixel-upscaler", required=True, metavar="PATH",
                    help="full path to the pixel upscaler model (.safetensors), "
                         "e.g. a Real-ESRGAN model")
    up.add_argument("--input", required=True, metavar="PATH",
                    help="input image to upscale")
    up.add_argument("--upscale-factor", type=float, default=0.0,
                    help="upscale factor (default: 0.0 = use the model's detected "
                         "scale); must be in [1, the model's detected scale]")
    up.add_argument("--out", default="out_upscaled.png",
                    help="output image path (default: out_upscaled.png)")
    up.add_argument("--device", default="cuda",
                    help="inference device; ROCm aliases cuda -> hip (default: cuda)")

    return parser
