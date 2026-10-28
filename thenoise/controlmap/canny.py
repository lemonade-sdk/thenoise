"""Canny edge maps in pure torch: blur, gradients, non-maximum suppression, hysteresis.
A third-party CV package cannot be installed here without overwriting the ROCm torch.
"""
from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from PIL import Image

from thenoise.utils.image_tensor import pil_to_pixels
from .base import ControlMap
from .options import CannyOptions

_LUMA = (0.2126, 0.7152, 0.0722)  # Rec. 709
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_BLUR = (5, 1.2)  # kernel size, sigma
# Gradient magnitude in 8-bit grey levels, so conventional 100/200 thresholds land on edges.
_GRAY_SCALE = 255.0
# The four quantised gradient orientations, as the neighbour offsets they compare with.
_ORIENTATIONS = ((0, 1), (1, 1), (1, 0), (1, -1))
# Weak pixels are absorbed one chain-pixel per pass; this only bounds pathological runs.
_MAX_GROW = 2048


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """``[3, H, W]`` pixels in ``[-1, 1]`` -> ``[1, 1, H, W]`` luma in ``[0, 1]``."""
    weights = rgb.new_tensor(_LUMA).view(3, 1, 1)
    gray = (rgb * weights).sum(0, keepdim=True) + 1.0
    return (gray / 2.0).unsqueeze(0)


def edge_mask(gray: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """``[1, 1, H, W]`` luma -> ``[1, H, W]`` edge mask in ``{0, 1}``."""
    blurred = _blur(gray)
    gx, gy = _sobel(blurred)
    magnitude = _GRAY_SCALE * torch.sqrt(gx * gx + gy * gy)
    return _hysteresis(_suppress(magnitude, gx, gy), low, high)[0].to(gray.dtype)


def thicken(mask: torch.Tensor, thickness: int) -> torch.Tensor:
    """Dilate a ``[1, H, W]`` mask by ``thickness - 1`` pixels."""
    for _ in range(thickness - 1):
        mask = F.max_pool2d(mask.unsqueeze(0), 3, stride=1, padding=1)[0]
    return mask


def _gauss1d(size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _blur(gray: torch.Tensor) -> torch.Tensor:
    """The control map's default blur: separable 5x5 Gaussian, sigma 1.2."""
    size, sigma = _BLUR
    return gaussian_blur(gray, size, sigma)


def gaussian_blur(gray: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    """Separable Gaussian, each pass padding only the axis it walks."""
    edge = size // 2
    kernel = _gauss1d(size, sigma).to(gray)
    along_rows = F.conv2d(
        F.pad(gray, (edge, edge, 0, 0), mode="replicate"), kernel.view(1, 1, 1, size)
    )
    along_columns = F.conv2d(
        F.pad(along_rows, (0, 0, edge, edge), mode="replicate"),
        kernel.view(1, 1, size, 1),
    )
    return along_columns


def _sobel(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = _SOBEL_X.to(gray)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, x.view(1, 1, 3, 3))
    gy = F.conv2d(padded, x.t().contiguous().view(1, 1, 3, 3))
    return gx, gy


def _shift(field: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    height, width = field.shape[-2:]
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    return padded[:, :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]


def _suppress(magnitude: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """Thin the edges: keep only local maxima along the gradient orientation."""
    orientation = torch.atan2(gy, gx).rad2deg().remainder(180.0)
    index = ((orientation + 22.5) // 45).long() % 4
    keep = torch.zeros_like(magnitude, dtype=torch.bool)
    for orientation_index, (dy, dx) in enumerate(_ORIENTATIONS):
        here = index == orientation_index
        keep |= (
            here
            & (magnitude >= _shift(magnitude, dy, dx))
            & (magnitude >= _shift(magnitude, -dy, -dx))
        )
    return magnitude * keep


def _hysteresis(suppressed: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Keep the weak pixels 8-connected to a strong one, by growing from strong seeds."""
    strong = suppressed >= high
    weak = (suppressed >= low) & ~strong
    edges = strong
    for _ in range(_MAX_GROW):
        connected = F.max_pool2d(edges.float(), 3, stride=1, padding=1) > 0
        grown = strong | (weak & connected)
        if bool(torch.equal(grown, edges)):
            return grown
        edges = grown
    return edges


class CannyControlMap(ControlMap):
    """Black canvas, white edges — the ControlNet ``canny`` conditioning input."""

    name: ClassVar[str] = "canny"
    kind: ClassVar[str] = "canny"
    DTYPE: ClassVar[torch.dtype] = torch.float32

    def detect(self, image: Image.Image, options: CannyOptions) -> torch.Tensor:
        """The edge mask at the input size as a ``[1, H, W]`` float tensor in ``{0, 1}``."""
        rgb = pil_to_pixels(image, 3).to(self.device, self.DTYPE)
        mask = edge_mask(luminance(rgb), options.low, options.high)
        return thicken(mask, options.thickness)


__all__ = [
    "CannyOptions",
    "CannyControlMap",
    "edge_mask",
    "gaussian_blur",
    "luminance",
    "thicken",
]
