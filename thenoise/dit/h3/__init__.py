"""H3: one packed single-stream DiT trunk, two models.

:mod:`.trunk` holds what they share, :mod:`.image` the still-image model and its loader;
the audiovisual model joins them as :mod:`.av`. The sigma grid and the token layout are
shared too — see :mod:`.sampling` and :mod:`.layout`.
"""
from .image import H3_IMAGE_SIGNATURE, H3ImageDiT, load_h3_image_dit
from .layout import Layout
from .trunk import H3DiT, H3Params

__all__ = [
    "H3_IMAGE_SIGNATURE",
    "H3DiT",
    "H3ImageDiT",
    "H3Params",
    "Layout",
    "load_h3_image_dit",
]
