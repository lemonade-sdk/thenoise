"""Post-process filters: ``rcas``, ``nyquist_notch``, ``film_grain`` and ``adjust_color``.

Pure, deterministic, CPU, tiny inputs. The assertions are invariants (identity at
strength 0, non-RGB channels passed through, DC preservation, seed determinism,
shape/dtype stability) rather than pixel values.
"""
from __future__ import annotations

import pytest
import torch

from thenoise.postprocess.color import adjust_color
from thenoise.postprocess.film_grain import film_grain
from thenoise.postprocess.nyquist import nyquist_notch
from thenoise.postprocess.rcas import rcas

ALL_FILTERS = {
    "rcas": lambda x: rcas(x, strength=0.8),
    "nyquist_notch": nyquist_notch,
    "film_grain": lambda x: film_grain(x, strength=0.05, seed=1234),
    "adjust_color": lambda x: adjust_color(
        x, saturation=0.6, vibrance=0.4, temperature=0.3, tint=-0.2
    ),
}


def test_zero_strength_is_the_identity():
    pixels = torch.rand(3, 9, 9)
    assert rcas(pixels, strength=0.0) is pixels
    assert film_grain(pixels, strength=0.0) is pixels
    # Every colour knob is neutral at 0, so a request that named none is untouched.
    assert adjust_color(pixels) is pixels
    assert adjust_color(pixels, saturation=0.0, vibrance=0.0, temperature=0.0, tint=0.0) is pixels


@pytest.mark.parametrize("filter_name", sorted(ALL_FILTERS))
def test_channels_beyond_rgb_pass_through_untouched(filter_name):
    """An alpha (or any 4th+) channel must never be filtered."""
    pixels = torch.rand(5, 8, 6) * 2 - 1
    out = ALL_FILTERS[filter_name](pixels)
    assert out.shape == pixels.shape
    assert torch.equal(out[3:], pixels[3:])


@pytest.mark.parametrize("filter_name", sorted(ALL_FILTERS))
def test_shape_and_dtype_survive_odd_dimensions(filter_name):
    pixels = torch.rand(3, 7, 5) * 2 - 1
    out = ALL_FILTERS[filter_name](pixels)
    assert out.shape == pixels.shape
    assert out.dtype == pixels.dtype


# ------------------------------------------------------------------------- colour


def _chroma(pixels: torch.Tensor) -> torch.Tensor:
    """How far each pixel sits from its own grey, in RGB: 0 for a neutral pixel.

    Deliberately crude. The filters work in OKLab; measuring the result in RGB keeps
    the assertions about *behaviour* independent of the space they were written in.
    """
    return (pixels - pixels.mean(dim=0, keepdim=True)).norm(dim=0)


def _luminance(pixels: torch.Tensor) -> torch.Tensor:
    """Rec. 709 luminance of ``[-1, 1]`` sRGB pixels, in linear light."""
    x = ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)
    linear = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))
    return (linear * torch.tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1)).sum(0)


def test_saturation_greyscales_at_its_negative_extreme():
    pixels = torch.rand(3, 12, 12) * 2 - 1
    grey = adjust_color(pixels, saturation=-1.0)

    assert torch.allclose(grey[0], grey[1], atol=3e-3)
    assert torch.allclose(grey[1], grey[2], atol=3e-3)


def test_saturation_moves_every_colour_away_from_grey():
    pixels = torch.rand(3, 12, 12) * 2 - 1
    out = adjust_color(pixels, saturation=0.5)

    before, after = _chroma(pixels), _chroma(out)
    moving = before > 1e-3
    assert ((after[moving] - before[moving]) > 0).float().mean() > 0.97


def test_saturation_leaves_neutrals_neutral():
    """Grey has no chroma, so scaling chroma cannot move it — no colour cast, ever."""
    grey = torch.linspace(-0.8, 0.8, 16).repeat(3, 8, 1)

    for knob in ("saturation", "vibrance"):
        out = adjust_color(grey, **{knob: 1.0})
        assert torch.allclose(out[0], out[1], atol=2e-4)
        assert torch.allclose(out[1], out[2], atol=2e-4)


def test_vibrance_lifts_washed_out_pixels_far_more_than_vivid_ones():
    """The point of vibrance over saturation: lift the washed-out, spare the vivid."""
    washed = torch.tensor([[[0.1]], [[0.0]], [[-0.1]]])     # a hint of colour
    vivid = torch.tensor([[[0.95]], [[-0.95]], [[-0.95]]])  # a near-pure red

    def lift(pixels, **knobs):
        before = _chroma(pixels)[0, 0].item()
        after = _chroma(adjust_color(pixels, **knobs))[0, 0].item()
        return (after - before) / before

    assert lift(washed, vibrance=1.0) > 0.4   # nearly the whole boost, as saturation gives
    assert lift(vivid, vibrance=1.0) < 0.05   # a colour at the gamut edge is left alone

    # Saturation has no such mercy. The gain there is far from the doubling the knob
    # asks for, because a pixel already at the gamut edge has nowhere to go and clamps.
    assert lift(vivid, saturation=1.0) > lift(vivid, vibrance=1.0)


@pytest.mark.parametrize("temperature,warmer", [(1.0, True), (-1.0, False)])
def test_temperature_moves_red_and_blue_in_opposite_directions(temperature, warmer):
    grey = torch.zeros(3, 4, 4)
    out = adjust_color(grey, temperature=temperature)

    red_up = out[0, 0, 0].item() > 0.0
    blue_down = out[2, 0, 0].item() < 0.0
    assert (red_up, blue_down) == (warmer, warmer)


@pytest.mark.parametrize("tint,magenta", [(1.0, True), (-1.0, False)])
def test_tint_moves_along_the_green_magenta_axis(tint, magenta):
    """Magenta is red *and* blue up with green down; the reverse is green."""
    grey = torch.zeros(3, 4, 4)
    out = adjust_color(grey, tint=tint)

    red, green, blue = (out[:, 0, 0] > 0.0).tolist()
    assert (red, blue, green) == (magenta, magenta, not magenta)


@pytest.mark.parametrize("value", [-0.6, 0.0, 0.7])
@pytest.mark.parametrize("knob", ["temperature", "tint"])
def test_white_balance_is_a_balance_not_an_exposure_change(knob, value):
    """The cone gains are normalised, so a mid-grey keeps the luminance it had."""
    grey = torch.full((3, 8, 8), 0.1)

    assert abs(_luminance(adjust_color(grey, **{knob: value})).mean().item()
               - _luminance(grey).mean().item()) < 2e-3


def test_white_balance_leaves_black_black():
    black = torch.full((3, 4, 4), -1.0)
    out = adjust_color(black, temperature=1.0, tint=-1.0, saturation=1.0, vibrance=1.0)

    assert torch.allclose(out, black)


def test_a_push_outside_the_gamut_is_clamped_and_stays_sane():
    """An extreme chroma scale cannot produce NaN, infinities or out-of-range pixels."""
    pixels = torch.rand(3, 16, 16) * 2 - 1
    out = adjust_color(pixels, saturation=1.0, vibrance=1.0, temperature=1.0, tint=1.0)

    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_nyquist_notch_preserves_dc():
    """A flat field must come back flat (no shading/halo)."""
    flat = torch.full((3, 9, 9), 0.5)
    assert torch.allclose(nyquist_notch(flat), flat, atol=1e-5)


def test_rcas_boosts_a_local_feature_without_spreading_it():
    """A bright pixel on a mid-grey field gets brighter; the field stays put."""
    pixels = torch.full((3, 9, 9), 0.5)
    pixels[:, 4, 4] = 0.9

    out = rcas(pixels, strength=1.0)
    assert out[0, 4, 4].item() > 0.9
    assert out[0, 3, 4].item() < 0.5

    # Nothing outside the 3x3 cross neighbourhood changes: sharpening is local.
    outside = out.clone()
    outside[:, 3:6, 3:6] = pixels[:, 3:6, 3:6]
    assert torch.equal(outside, pixels)


@pytest.mark.parametrize("base", [-0.4, 0.0, 0.4])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_rcas_sharpens_edges_anywhere_in_the_pixel_range(base, sign):
    """A feature must be sharpened no matter where it sits in [-1, 1]: the headroom
    on either side of it changes with ``base`` and ``sign``.
    """
    pixels = torch.full((3, 9, 9), base)
    pixels[:, 4, 4] = base + sign * 0.4

    out = rcas(pixels, strength=1.0)

    assert (out[0, 4, 4] - pixels[0, 4, 4]) * sign > 1e-3
    assert (out[0, 3, 4] - base) * sign < -1e-3


def test_rcas_sharpens_dark_and_bright_features_equally():
    """The response must be symmetric in brightness."""
    bright = torch.full((3, 9, 9), -0.3)
    bright[:, 4, 4] = 0.2
    dark = -bright.clone()

    out_bright = rcas(bright, strength=0.7)
    out_dark = rcas(dark, strength=0.7)
    assert torch.allclose(out_dark, -out_bright, atol=1e-6)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_rcas_at_full_strength_uses_the_available_headroom(sign):
    """The lobe is solved to push as far as it can without clipping, so at full
    strength a lone mid-tone feature lands on the edge of the range."""
    pixels = torch.full((3, 9, 9), 0.4 * sign)
    pixels[:, 4, 4] = 0.0

    out = rcas(pixels, strength=1.0)
    assert out[0, 4, 4].abs().item() >= 0.8


def test_rcas_lobes_are_limited_by_the_highest_contrast_channel():
    """A channel must not be sharpened using another channel's contrast, which
    overshoots it straight into clipping."""
    pixels = torch.full((3, 9, 9), 0.2)
    pixels[:, 4, 4] = torch.tensor([0.3, 0.5, 0.7])

    out = rcas(pixels, strength=0.5)

    assert out[2, 4, 4].item() < 0.95
    assert out[0, 4, 4].item() > 0.3


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_rcas_flat_fields_are_untouched(value):
    flat = torch.full((3, 9, 9), value)
    out = rcas(flat, strength=1.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, flat, atol=1e-5)


def test_rcas_leaves_a_saturated_step_edge_alone():
    """A black/white step has no headroom: the lobe must go to 0."""
    step = torch.tensor([-1.0, -1.0, 1.0, 1.0]).repeat(9, 1).unsqueeze(0)
    pixels = step.expand(3, 9, 4).contiguous()

    out = rcas(pixels, strength=1.0)
    assert torch.equal(out, pixels)


def test_nyquist_notch_flattens_a_two_pixel_checkerboard():
    """The filter exists to remove exactly this 2px grid artifact."""
    grid = torch.arange(16).view(-1, 1) + torch.arange(16).view(1, -1)
    checker = torch.where(grid % 2 == 0, 0.5, 0.3).float().expand(3, 16, 16).contiguous()

    out = nyquist_notch(checker)
    assert out.std().item() < checker.std().item() / 10
    # ...without shifting the average brightness.
    assert abs(out.mean().item() - checker.mean().item()) < 1e-4


def test_film_grain_is_seed_deterministic():
    pixels = torch.rand(3, 16, 16)
    first = film_grain(pixels, strength=0.1, seed=7)
    assert torch.equal(first, film_grain(pixels, strength=0.1, seed=7))
    assert not torch.equal(first, film_grain(pixels, strength=0.1, seed=8))
    # With no seed at all, every call is a different draw.
    assert not torch.equal(
        film_grain(pixels, strength=0.1), film_grain(pixels, strength=0.1)
    )


def test_film_grain_adds_to_luminance_only():
    """A luminance shift == the same delta on every channel: channel *differences*
    (i.e. the colour) must be unchanged."""
    pixels = torch.rand(3, 16, 16)
    out = film_grain(pixels, strength=0.05, seed=3)

    assert torch.allclose(out[0] - out[1], pixels[0] - pixels[1], atol=1e-6)
    assert torch.allclose(out[1] - out[2], pixels[1] - pixels[2], atol=1e-6)

    # The grain is spatially correlated (blurred), so it is not white noise and
    # stays small relative to the signal range.
    delta = out - pixels
    assert delta.abs().max().item() < 0.5
    assert delta.abs().mean().item() < delta.abs().max().item()
    assert out.isfinite().all()
