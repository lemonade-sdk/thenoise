"""Post-process filters: ``rcas``, ``nyquist_notch`` and ``film_grain``.

Pure, deterministic, CPU, tiny inputs. The assertions are invariants (identity at
strength 0, non-RGB channels passed through, DC preservation, seed determinism,
shape/dtype stability) rather than pixel values.
"""
from __future__ import annotations

import pytest
import torch

from thenoise.postprocess.film_grain import film_grain
from thenoise.postprocess.nyquist import nyquist_notch
from thenoise.postprocess.rcas import rcas

ALL_FILTERS = {
    "rcas": lambda x: rcas(x, strength=0.8),
    "nyquist_notch": nyquist_notch,
    "film_grain": lambda x: film_grain(x, strength=0.05, seed=1234),
}


def test_zero_strength_is_the_identity():
    pixels = torch.rand(3, 9, 9)
    assert rcas(pixels, strength=0.0) is pixels
    assert film_grain(pixels, strength=0.0) is pixels


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
