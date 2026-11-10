"""H3-Image: a packed single-stream DiT, its sigma-grid schedule and its token layout."""
from .layout import Layout
from .models import H3ImageDiT, H3ImageParams, detect_params, load_h3_image_dit

__all__ = [
    "H3ImageDiT",
    "H3ImageParams",
    "Layout",
    "detect_params",
    "load_h3_image_dit",
]
