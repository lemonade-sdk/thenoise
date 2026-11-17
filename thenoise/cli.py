"""CLI subcommands: serve, generate, edit, video, upscale, controlmap — the first four
take model paths, the last two are model-free. No config file: everything is a flag.
"""
from __future__ import annotations

import argparse

from thenoise.controlmap.options import OPTIONS_BY_KIND, add_option_args
from thenoise.postprocess.options import PostprocessOptions
from thenoise.utils.options import add_flags


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


def _add_dit_low(p: argparse.ArgumentParser) -> None:
    """The second expert of a two-expert model, where ``--dit`` is the first."""
    p.add_argument("--dit-low", default="", metavar="PATH",
                   help="low-noise DiT checkpoint of a two-expert model "
                        "(--dit selects the high-noise expert)")


def _add_upscaler_args(p: argparse.ArgumentParser) -> None:
    """Pixel-upscaler flags for a subcommand: ``serve`` takes ``--upscaler-dir``, the
    upscaler itself being selected per request through the ``pixel_upscaler`` API field."""
    p.add_argument("--upscaler-dir", default="", metavar="PATH",
                   help="directory containing pixel upscaler .safetensors files "
                        "(e.g. Real-ESRGAN); selected per-request via the "
                        "'pixel_upscaler' API field")


def _add_controlmap_args(p: argparse.ArgumentParser) -> None:
    """Control-map flags for a subcommand: ``--controlmap-dir`` is server/one-shot
    configuration, the control map being selected per request through ``--model``."""
    p.add_argument("--controlmap-dir", default="", metavar="PATH",
                   help="directory containing control-map checkpoints "
                        "(e.g. Depth-Anything-V2); each subdirectory's config.json "
                        "declares the kind it implements")


def _add_controlmap_option_args(p: argparse.ArgumentParser) -> None:
    """One ``--<kind>-<option>`` flag per option, defaulted from the dataclass."""
    for kind, options in OPTIONS_BY_KIND.items():
        add_option_args(p, kind, options)


def _add_postprocess_args(p: argparse.ArgumentParser) -> None:
    """One flag per post-decode knob, defaulted and bounded from the declaration."""
    add_flags(p, PostprocessOptions, bounds=True)


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
    p.add_argument("--sampler", metavar="NAME",
                   help="denoising solver (default: auto)")
    p.add_argument("--kv-cache", dest="kv_cache", action="store_true",
                   help="reuse step-invariant K/V during denoising (T2I opt-in; "
                        "automatic for compatible edits)")
    p.add_argument("--no-kv-cache", dest="kv_cache", action="store_false",
                   help="disable KV caching (overrides automatic editing defaults)")
    p.set_defaults(kv_cache=None)
    p.add_argument("--ref-method",
                   choices=["index", "index_timestep_zero", "vision"],
                   default=None,
                   help="how a reference is used (default: auto, i.e. detected from "
                        "the checkpoint marker, else 'index'); "
                        "'index_timestep_zero' conditions reference tokens at "
                        "timestep zero, which is what makes the KV cache valid, and "
                        "'vision' has the text encoder read the picture rather than "
                        "the VAE. Which of them a model takes is the model's own")
    _add_postprocess_args(p)
    p.add_argument("--device", default="cuda",
                   help="inference device; ROCm aliases cuda -> hip (default: cuda)")


def _add_video_args(p: argparse.ArgumentParser) -> None:
    """The clip knobs. Nothing defaults but the output path and the device: what a
    release ships is the model's to answer for."""
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--frames", type=int,
                   help="frames to render, rounded onto the model's temporal grid")
    p.add_argument("--duration", type=float, metavar="SECONDS",
                   help="clip length, an alternative to --frames")
    p.add_argument("--fps", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance-scale", type=float)
    p.add_argument("--flow-shift", type=float,
                   help="schedule shift; a distilled LoRA wants the one it ships with")
    p.add_argument("--sampler", metavar="NAME",
                   help="denoising solver (default: the model's)")
    p.add_argument("--seed", type=int)
    p.add_argument("--output-format", choices=["mp4", "webm"],
                   help="container; --out's extension is taken when this is left out")
    p.add_argument("--image", metavar="PATH",
                   help="image the clip starts on")
    p.add_argument("--end-image", metavar="PATH",
                   help="image the clip ends on, for a model that can be told "
                        "where a clip ends; needs --image")
    p.add_argument("--lora", action="append", default=[], metavar="FILE[:WEIGHT]",
                   help="LoRA for the primary transformer (repeatable)")
    p.add_argument("--lora-low", dest="lora_low", action="append", default=[],
                   metavar="FILE[:WEIGHT]",
                   help="LoRA for the low-noise expert of a --dit-low model; a LoRA "
                        "for the whole clip goes to both")
    p.add_argument("--out", default="out.mp4",
                   help="output clip path (default: out.mp4)")
    p.add_argument("--device", default="cuda",
                   help="inference device; ROCm aliases cuda -> hip (default: cuda)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thenoise")
    sub = parser.add_subparsers(dest="command", required=True)

    # serve: model paths are optional — without them the server runs model-free
    # and only the pixel-upscale tab is usable.
    serve = sub.add_parser("serve", help="run the FastAPI HTTP server")
    _add_model_paths(serve, required=False)
    _add_dit_low(serve)
    _add_upscaler_args(serve)
    _add_controlmap_args(serve)
    serve.add_argument("--cache-dir", default="~/.cache/thenoise", metavar="DIR",
                       help="directory for generated files; a queued job's image or "
                            "clip lands in <cache-dir>/jobs")
    serve.add_argument("--host", default="127.0.0.1",
                       help="bind host (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000,
                       help="bind port (default: 8000)")
    serve.add_argument("--device", default="cuda",
                       help="inference device; ROCm aliases cuda -> hip (default: cuda)")
    serve.add_argument(
        "--offload-device", default="cpu",
        help="device for idle weights (default: %(default)s; pass the compute device to"
             " keep every weight set resident)")

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

    # video
    video = sub.add_parser("video", help="run one generation and save a clip")
    _add_model_paths(video)
    _add_dit_low(video)
    _add_video_args(video)

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

    # controlmap: extract a control map; needs no diffusion model either
    pre = sub.add_parser("controlmap",
                         help="extract a control map (canny / depth / pose) from an "
                              "image (no model needed)")
    pre.add_argument("--input", metavar="PATH", help="image to extract from")
    pre.add_argument("--type", dest="types", action="append", metavar="KIND",
                     help="control map to extract, repeatable "
                          "(see --list for what is available)")
    pre.add_argument("--model", default="",
                     help="implementation to use for the chosen type "
                          "(default: the type's own)")
    _add_controlmap_args(pre)
    pre.add_argument("--list", action="store_true",
                     help="list the available control-map types and exit")
    _add_controlmap_option_args(pre)
    pre.add_argument("--out", default="out_control.png",
                     help="output image path; the map is the size of the input, and "
                          "extra --type values add a _<type> suffix "
                          "(default: out_control.png)")
    pre.add_argument("--device", default="cuda",
                     help="inference device; ROCm aliases cuda -> hip (default: cuda)")

    return parser
