"""CLI pixel upscaling: load a pixel upscaler, upscale one image, save a PNG.

Model-free: needs no diffusion model (unlike ``generate``). Thin wrapper over the
same ``PixelUpscaleController`` the HTTP API uses, so there is no logic drift
between the two surfaces.
"""
from __future__ import annotations

import logging
import os

from .utils.paths import ensure_png_extension

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_upscale(args) -> None:
    from .runtime import Settings, Runtime
    from .utils.image_tensor import load_image

    # ``--pixel-upscaler`` is a one-shot convenience: a full path to the model.
    # Split it into ``upscaler_dir`` (server config) + name (sans suffix), the
    # same form the ``serve``/API path uses. ``or "."`` keeps a bare filename
    # (no directory) usable: it then resolves against the current directory.
    upscaler_dir = os.path.dirname(args.pixel_upscaler) or "."
    name = os.path.basename(args.pixel_upscaler)
    if name.endswith(".safetensors"):
        name = name[: -len(".safetensors")]

    settings = Settings(device=args.device, upscaler_dir=upscaler_dir)
    runtime = Runtime(settings)  # no load() — pixel upscaling is model-free

    # Opened without flattening: an alpha is preserved through the upscaler.
    image = load_image(args.input)
    out = runtime.upscaler.upscale(image, args.upscale_factor, name)

    out_path = ensure_png_extension(args.out)
    out.save(out_path, pnginfo=getattr(out, "_pnginfo", None))
    factor = getattr(out, "_upscale_factor", args.upscale_factor)
    logger.info("saved %s (upscaler=%s, factor=%s)", out_path, name, factor)
