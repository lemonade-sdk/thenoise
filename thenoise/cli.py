"""CLI subcommands: serve, generate, edit, upscale, preprocess — the first three take
model paths, the last two are model-free. No config file: everything is a flag.
"""
from __future__ import annotations

import argparse

from thenoise.preprocess.options import OPTIONS_BY_KIND, add_option_args


def _add_model_paths(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--dit", required=required, metavar="PATH",
                   help="DiT checkpoint (.safetensors)")
    p.add_argument("--vae", required=required, metavar="PATH",
                   help="VAE checkpoint (.safetensors)")
    p.add_argument("--text-encoder", required=required, metavar="PATH",
                   help="text encoder checkpoint (.safetensors)")
    p.add_argument("--lora-dir", default="", metavar="PATH",
                   help="directory containing LoRA .safetensors files "
                        "(subdirectories allowed)")


def _add_upscaler_args(p: argparse.ArgumentParser) -> None:
    """Pixel-upscaler flags for a subcommand: ``serve`` takes ``--upscaler-dir``, the
    upscaler itself being selected per request through the ``pixel_upscaler`` API field."""
    p.add_argument("--upscaler-dir", default="", metavar="PATH",
                   help="directory containing pixel upscaler .safetensors files "
                        "(e.g. Real-ESRGAN); selected per-request via the "
                        "'pixel_upscaler' API field")


def _add_preprocessor_args(p: argparse.ArgumentParser) -> None:
    """Control-map flags for a subcommand: ``--preprocessor-dir`` is server/one-shot
    configuration, the preprocessor being selected per request through ``--model``."""
    p.add_argument("--preprocessor-dir", default="", metavar="PATH",
                   help="directory containing control-map preprocessor checkpoints "
                        "(e.g. Depth-Anything-V2); each subdirectory's config.json "
                        "declares the kind it implements")


def _add_preprocessor_option_args(p: argparse.ArgumentParser) -> None:
    """One ``--<kind>-<option>`` flag per option, defaulted from the dataclass."""
    for kind, options in OPTIONS_BY_KIND.items():
        add_option_args(p, kind, options)


def _add_generation_args(p: argparse.ArgumentParser, out_default: str = "out.png") -> None:
    """Add the generation options shared by ``generate`` and ``edit``."""
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
                   help="upscale the latent 2x in latent space and run a "
                        "low-strength refine denoise "
                        "before decoding")
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
    p.add_argument("--kv-cache", dest="kv_cache", action="store_true",
                   help="reference-latent KV cache (edit only): freeze the reference "
                        "tokens' K/V across denoise steps for faster editing "
                        "(implies --ref-method index_timestep_zero unless given)")
    p.add_argument("--no-kv-cache", dest="kv_cache", action="store_false",
                   help="disable the reference-latent KV cache (auto: on for "
                        "'index_timestep_zero' reference conditioning)")
    p.set_defaults(kv_cache=None)
    p.add_argument("--ref-method", choices=["index", "index_timestep_zero"],
                   default=None,
                   help="reference conditioning for editing (default: auto, i.e. "
                        "detected from the checkpoint marker, else 'index'); "
                        "'index_timestep_zero' conditions reference tokens at "
                        "timestep zero, which is what makes the KV cache valid")
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
    _add_model_paths(serve, required=False)
    _add_upscaler_args(serve)
    _add_preprocessor_args(serve)
    serve.add_argument("--host", default="127.0.0.1",
                       help="bind host (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000,
                       help="bind port (default: 8000)")
    serve.add_argument("--device", default="cuda",
                       help="inference device; ROCm aliases cuda -> hip (default: cuda)")
    serve.add_argument("--offload-device", default="",
                       help="device for idle weights (default: auto-detect from safetensors size vs VRAM)")

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

    # preprocess: extract a control map; needs no diffusion model either
    pre = sub.add_parser("preprocess",
                         help="extract a control map (canny / depth / pose) from an "
                              "image (no model needed)")
    pre.add_argument("--input", metavar="PATH", help="image to extract from")
    pre.add_argument("--type", dest="types", action="append", metavar="KIND",
                     help="control map to extract, repeatable "
                          "(see --list for what is available)")
    pre.add_argument("--model", default="",
                     help="implementation to use for the chosen type "
                          "(default: the type's own)")
    _add_preprocessor_args(pre)
    pre.add_argument("--list", action="store_true",
                     help="list the available control-map types and exit")
    _add_preprocessor_option_args(pre)
    pre.add_argument("--out", default="out_control.png",
                     help="output image path; the map is the size of the input, and "
                          "extra --type values add a _<type> suffix "
                          "(default: out_control.png)")
    pre.add_argument("--device", default="cuda",
                     help="inference device; ROCm aliases cuda -> hip (default: cuda)")

    return parser
