"""Post-process filters: ``rcas``, ``nyquist_notch`` and ``film_grain``.

Pure, deterministic, CPU, tiny inputs. The assertions are the documented
invariants (identity at strength 0, non-RGB channels passed through, DC
preservation, seed determinism, shape/dtype stability) rather than pixel
values, so a legitimate kernel tweak does not break them but a real regression
does.
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


@pytest.mark.parametrize("strength", [0.0, -0.0])
def test_rcas_zero_strength_is_the_identity(strength):
    pixels = torch.rand(3, 9, 9)
    assert rcas(pixels, strength=strength) is pixels


def test_film_grain_zero_strength_is_the_identity():
    pixels = torch.rand(3, 9, 9)
    assert film_grain(pixels, strength=0.0) is pixels


@pytest.mark.parametrize("filter_name", sorted(ALL_FILTERS))
def test_channels_beyond_rgb_pass_through_untouched(filter_name):
    """An alpha (or any 4th+) channel must never be filtered."""
    pixels = torch.rand(5, 8, 6)
    out = ALL_FILTERS[filter_name](pixels)
    assert out.shape == pixels.shape
    assert torch.equal(out[3:], pixels[3:])


@pytest.mark.parametrize("filter_name", sorted(ALL_FILTERS))
def test_shape_and_dtype_survive_odd_dimensions(filter_name):
    pixels = torch.rand(3, 7, 5)
    out = ALL_FILTERS[filter_name](pixels)
    assert out.shape == pixels.shape
    assert out.dtype == pixels.dtype


@pytest.mark.parametrize("filter_name", ["rcas", "nyquist_notch"])
def test_constant_field_stays_constant(filter_name):
    """DC preservation: a flat field must come back flat (no shading/halo)."""
    flat = torch.full((3, 9, 9), 0.5)
    out = ALL_FILTERS[filter_name](flat)
    assert torch.allclose(out, flat, atol=1e-5)


def test_rcas_boosts_a_local_feature_without_spreading_it():
    """A bright pixel on a mid-grey field gets brighter; the field stays put."""
    pixels = torch.full((3, 9, 9), 0.5)
    pixels[:, 4, 4] = 0.9

    out = rcas(pixels, strength=1.0)
    assert out[0, 4, 4].item() > 0.9  # the peak is pushed up
    assert out[0, 3, 4].item() < 0.5  # ...and its ring is pushed down

    # Nothing outside the 3x3 cross neighbourhood changes: sharpening is local.
    outside = out.clone()
    outside[:, 3:6, 3:6] = pixels[:, 3:6, 3:6]
    assert torch.equal(outside, pixels)


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


def test_film_grain_without_a_seed_is_random():
    pixels = torch.rand(3, 16, 16)
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
