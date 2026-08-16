"""CLI generation: load one model, run one generation, save a single PNG.

Thin wrapper over the same adapter ``generate()`` methods the HTTP API uses, so
there is no logic drift between the two surfaces. The seed is resolved here (when
not given) so it can be reported for reproducibility.
"""
from __future__ import annotations

import logging
import os
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_generate(args) -> None:
    from .runtime import Settings, ModelPaths, Runtime
    settings = Settings(device=args.device)

    runtime = Runtime(settings)

    # ``--pixel-upscaler`` is a one-shot convenience: a full path to the model.
    # Split it into ``upscaler_dir`` + ``pixel_upscaler`` (name, sans suffix)
    # before passing it down the chain, so the runtime/model only ever see the
    # same directory + name form as the ``serve``/API path.
    upscaler_dir = ""
    pixel_upscaler = None
    if args.pixel_upscaler:
        upscaler_dir = os.path.dirname(args.pixel_upscaler)
        pixel_upscaler = os.path.basename(args.pixel_upscaler)
        if pixel_upscaler.endswith(".safetensors"):
            pixel_upscaler = pixel_upscaler[: -len(".safetensors")]

    runtime.load(
        ModelPaths(
            dit_path=args.dit,
            vae_path=args.vae,
            text_encoder_path=args.text_encoder,
            lora_dir=args.lora_dir,
            upscaler_dir=upscaler_dir,
        ),
    )

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    image = runtime.model.generate(
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

    image.save(args.out, pnginfo=getattr(image, "_pnginfo", None))
    logger.info("saved %s (model=%s, seed=%s)", args.out, runtime.model_name, seed)
