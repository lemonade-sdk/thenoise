"""Colour: saturation, vibrance, and a white-balance pair.

Everything happens in **OKLab** — a perceptual colour space whose lightness axis is
close to what the eye reads as brightness, and whose ``a``/``b`` plane is chroma
(colour intensity) with hue as the angle in it. Working there rather than in RGB means
a saturation change moves colours *outward from grey* evenly: it does not brighten the
reds and swallow the blues the way ``lerp(luma, rgb, s)`` does, because in RGB a step
of a channel means a different thing at every hue.

One filter takes all four knobs, so the whole correction is a single
linearise → convert → scale → convert back → encode pass over the pixels, not four.

* ``saturation`` scales chroma outright: -1 greyscales, 1 doubles.
* ``vibrance`` scales it by how *unsaturated* each pixel already is, so a boost lifts
  the washed-out parts of an image and leaves already-vivid ones (and skin) alone. It
  is the same arithmetic once the chroma has been measured, hence nearly free.
* ``temperature`` and ``tint`` are a von Kries white balance: a diagonal gain on the
  L, M and S cone responses. Multiplying rather than adding is what makes it behave
  like white balance — the cast scales with luminance, so blacks stay black — and the
  gains are normalised to leave a neutral white exactly as bright as it was, so
  sliding a slider changes the colour and not the exposure.

White balance runs before chroma: balancing changes how much chroma there is to scale,
and not the other way round.

Scaling chroma leaves OKLab lightness *exactly* where it was — the lightness axis and
the chroma plane are independent linear combinations of the same cone responses — so a
saturation change never brightens or darkens an image except where the result no longer
fits in sRGB. There the value is clamped, so an extreme push shifts the hue of that
pixel rather than letting it wrap.
"""
from __future__ import annotations

import torch

# OKLab (Björn Ottosson): linear sRGB -> cone responses -> cube-rooted responses ->
# the perceptual axes, and both inverses. The rows of the first matrix sum to 1, so a
# neutral white is (1, 1, 1) in cone space too.
_RGB_TO_LMS = [
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
]
_LMS_GAMMA_TO_OKLAB = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
]
_OKLAB_TO_LMS_GAMMA = [
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
]
_LMS_GAMMA_TO_RGB = [
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010],
]

#: Rec. 709 luminance weights: what a linear RGB triple is worth as brightness.
_LUMA = (0.2126, 0.7152, 0.0722)

# How far a full deflection of each white-balance knob swings the cone gains. Warmth
# lifts the long (red) cone and drops the short (blue) one; tint cuts the middle (green)
# cone response for magenta. Slightly asymmetric because the eye resolves blue far less
# finely than red or green.
_WARM_L, _WARM_S = 0.10, 0.12
_TINT_M = 0.07

#: The chroma a strongly saturated sRGB colour reaches: its primaries sit at 0.26 (red),
#: 0.30 (green) and 0.31 (blue). Vibrance measures "how much room is left" against it,
#: which is what gives the knob its own scale and leaves an already pure colour alone.
#: A hue-aware limit would be stricter, at several times the code.
_CHROMA_FULL = 0.26


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """The sRGB transfer function's inverse, on ``[0, 1]``."""
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(u: torch.Tensor) -> torch.Tensor:
    """The sRGB transfer function, on linear ``[0, 1]``; out of range is clamped."""
    u = u.clamp(0.0, 1.0)
    return torch.where(u <= 0.0031308, u * 12.92, 1.055 * u.pow(1.0 / 2.4) - 0.055)


def _matrix(values, like: torch.Tensor) -> torch.Tensor:
    """A 3x3 constant on ``like``'s device and dtype."""
    return torch.tensor(values, dtype=like.dtype, device=like.device)


def _cube(x: torch.Tensor) -> torch.Tensor:
    """Element-wise cube, by multiplication: negative axes must stay negative."""
    return x * x * x


def _cone_gains(temperature: float, tint: float, like: torch.Tensor) -> torch.Tensor:
    """The von Kries diagonal, normalised so a neutral white keeps its luminance."""
    gains = torch.tensor(
        [
            1.0 + _WARM_L * temperature,
            1.0 - _TINT_M * tint,
            1.0 - _WARM_S * temperature,
        ],
        dtype=like.dtype, device=like.device,
    )
    # A white is (1, 1, 1) in cone space as well, so this is where the balance leaves
    # it. Normalising happens on the cube-rooted gains, because that is the domain the
    # RGB transform is linear in: dividing them by the white's own luminance divides
    # the white's luminance by the same factor, which puts the exposure back.
    rooted = gains.pow(1.0 / 3.0)
    luminance = _matrix(_LUMA, like) @ (_matrix(_LMS_GAMMA_TO_RGB, like) @ rooted)
    return (rooted / luminance.clamp(min=1e-6)).pow(3.0)


def adjust_color(
    pixels: torch.Tensor,
    *,
    saturation: float = 0.0,
    vibrance: float = 0.0,
    temperature: float = 0.0,
    tint: float = 0.0,
) -> torch.Tensor:
    """Correct the colour of ``[C, H, W]`` pixels in ``[-1, 1]``; every knob is neutral at 0."""
    c, h, w = pixels.shape
    rgb = pixels[: min(3, c)]
    if not any((saturation, vibrance, temperature, tint)):
        return pixels
    if rgb.shape[0] == 0:
        return pixels

    flat = _srgb_to_linear(((rgb + 1.0) * 0.5).clamp(0.0, 1.0)).reshape(3, -1)

    lms = _matrix(_RGB_TO_LMS, pixels) @ flat
    if temperature or tint:
        lms = lms * _cone_gains(temperature, tint, pixels)[:, None]
    axes = _matrix(_LMS_GAMMA_TO_OKLAB, pixels) @ lms.clamp(min=0.0).pow(1.0 / 3.0)

    if saturation or vibrance:
        chroma = axes[1:].norm(dim=0)
        # At full deflection vibrance leaves a pixel that is already at full chroma
        # where it was, and scales a washed-out one as hard as saturation would.
        room = 1.0 - (chroma / _CHROMA_FULL).clamp(0.0, 1.0)
        axes[1:] = axes[1:] * ((1.0 + saturation) * (1.0 + vibrance * room))

    lms_gamma = _matrix(_OKLAB_TO_LMS_GAMMA, pixels) @ axes
    back = _matrix(_LMS_GAMMA_TO_RGB, pixels) @ _cube(lms_gamma)
    corrected = _linear_to_srgb(back.reshape(3, h, w)) * 2.0 - 1.0

    if c > 3:
        return torch.cat([corrected, pixels[3:]], dim=0)
    return corrected


__all__ = ["adjust_color"]
