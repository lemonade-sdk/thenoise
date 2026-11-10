"""H3-Image: a packed single-stream DiT, its sigma-grid schedule and its token layout.

The conditioner it is trained against — a Qwen3-VL-32B cut to 50 layers — lives in
:mod:`.encoder`; the codec it denoises in :mod:`thenoise.vae.h3_image`.
"""
from .layout import Layout
from .models import H3ImageDiT, H3ImageParams, detect_params, load_h3_image_dit

__all__ = [
    "H3ImageDiT",
    "H3ImageParams",
    "Layout",
    "detect_params",
    "load_h3_image_dit",
]
