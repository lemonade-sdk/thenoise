"""H3: one packed trunk, two models.

:mod:`.trunk` holds what they share and :mod:`.checkpoint` what a file states about
itself; :mod:`.image` is the still-image model and :mod:`.av` the audiovisual one. The
sigma grid and the token layout are shared too — see :mod:`.sampling` and :mod:`.layout`.
"""
from .av import H3AVDiT
from .checkpoint import H3_AV_SIGNATURE, H3_IMAGE_SIGNATURE, detect_params, load_h3_image_dit
from .image import H3ImageDiT
from .layout import Layout
from .trunk import H3DiT, H3Params

__all__ = [
    "H3_AV_SIGNATURE",
    "H3_IMAGE_SIGNATURE",
    "H3AVDiT",
    "H3DiT",
    "H3ImageDiT",
    "H3Params",
    "Layout",
    "detect_params",
    "load_h3_image_dit",
]
