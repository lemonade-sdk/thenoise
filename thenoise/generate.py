"""CLI generation: load one model, run one generation, save a single PNG.

Thin wrappers over the same adapter ``generate()``/``edit()`` methods the HTTP API
uses, so there is no logic drift between the two surfaces. The seed is resolved
here (when not given) so it can be reported for reproducibility.

``run_generate`` is text-to-image only; ``run_edit`` handles instruction-based
editing through ``pipeline.edit`` with the input image(s).
"""
from __future__ import annotations

import logging
import os
import random
import sys

from .utils.paths import ensure_png_extension

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_MAX_DIM = 4096


def _check_dim(name: str, value) -> None:
    if value is not None and (value < 0 or value > _MAX_DIM):
        print(f"error: {name} must be between 0 and {_MAX_DIM} (got {value}).", file=sys.stderr)
        sys.exit(1)

def _build_runtime(args):
    """Load the model and split a one-shot --pixel-upscaler into dir+name."""
    from .cli import resolve_model_paths
    from .runtime import Settings, ModelPaths, Runtime

    settings = Settings(device=args.device)

    # ``--pixel-upscaler`` is a one-shot convenience: a full path to the model.
    # Split it into ``upscaler_dir`` (server config) + ``pixel_upscaler`` (name,
    # sans suffix) so the runtime/controller only ever see the same directory +
    # name form as the ``serve``/API path. Must run before ``Runtime`` is built:
    # the pixel-upscaler manager reads ``settings.upscaler_dir`` at construction.
    upscaler_dir = ""
    pixel_upscaler = None
    if args.pixel_upscaler:
        # ``or "."`` keeps a bare filename (no directory) usable: it then
        # resolves against the current working directory.
        upscaler_dir = os.path.dirname(args.pixel_upscaler) or "."
        pixel_upscaler = os.path.basename(args.pixel_upscaler)
        if pixel_upscaler.endswith(".safetensors"):
            pixel_upscaler = pixel_upscaler[: -len(".safetensors")]
        settings.upscaler_dir = upscaler_dir

    runtime = Runtime(settings)
    runtime.load(ModelPaths(**resolve_model_paths(args)))
    return runtime, pixel_upscaler


def run_generate(args) -> None:
    _check_dim("width", args.width)
    _check_dim("height", args.height)

    from .models.config import GenerateRequest
    runtime, pixel_upscaler = _build_runtime(args)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    request = GenerateRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=seed,
        upscale=args.upscale,
        upscale_factor=args.upscale_factor,
        upscale_type=args.upscale_type,
        sampler=args.sampler,
        qwen_vae_enhance=args.qwen_vae_enhance,
        film_grain=args.film_grain,
        sharpening=args.sharpening,
        lora_specs=args.lora or None,
        pixel_upscaler=pixel_upscaler,
    )

    image = runtime.pipeline.generate(request)

    # If the user omitted the output extension, PIL cannot infer a format.
    # Default to PNG so a bare --out like ``out`` (or ``dir/out``) still works.
    out_path = ensure_png_extension(args.out)

    image.save(out_path, pnginfo=getattr(image, "_pnginfo", None))
    logger.info("saved %s (model=%s, seed=%s)", out_path, runtime.model_name, seed)


def run_edit(args) -> None:
    """One-shot instruction-based edit: load the model, edit one image, save a PNG.
    """
    _check_dim("width", args.width)
    _check_dim("height", args.height)

    from .models.config import GenerateRequest
    runtime, pixel_upscaler = _build_runtime(args)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    request = GenerateRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=seed,
        upscale=args.upscale,
        upscale_factor=args.upscale_factor,
        upscale_type=args.upscale_type,
        sampler=args.sampler,
        qwen_vae_enhance=args.qwen_vae_enhance,
        film_grain=args.film_grain,
        sharpening=args.sharpening,
        lora_specs=args.lora or None,
        pixel_upscaler=pixel_upscaler,
    )

    from PIL import Image

    # ``--image`` is repeatable; first sets aspect/size, rest are refs.
    request.image = [Image.open(p).convert("RGB") for p in args.image]
    image = runtime.pipeline.edit(request)

    out_path = ensure_png_extension(args.out)
    image.save(out_path, pnginfo=getattr(image, "_pnginfo", None))
    logger.info("saved %s (model=%s, seed=%s)", out_path, runtime.model_name, seed)
