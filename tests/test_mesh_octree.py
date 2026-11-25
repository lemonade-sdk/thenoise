import pytest
import torch

from thenoise.mesh.octree import (CORNERS, MIN_LEVEL_RESOLUTION, octree_levels,
                           sample_sdf, volume_bytes)


def sphere(radius=0.5):
    def sdf(points):
        return points.norm(dim=-1) - radius
    return sdf


def brute_force(sdf, resolution, bounds=1.0, level=0.0):
    axis = torch.linspace(-bounds, bounds, resolution + 1)
    corners = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    values = sdf(corners.reshape(-1, 3)).reshape(resolution + 1, resolution + 1,
                                                 resolution + 1)
    shifts = torch.stack([values[i:i + resolution, j:j + resolution, k:k + resolution]
                          for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    return values, (shifts.amin(0) <= level) & (shifts.amax(0) >= level)


def test_levels_end_at_the_resolution_and_divide_the_next():
    assert octree_levels(384) == (96, 192, 384)
    assert octree_levels(128) == (64, 128)
    for resolution in range(1, 600):
        levels = octree_levels(resolution, start=4)
        assert levels[-1] == resolution
        assert list(levels) == sorted(set(levels))
        for coarse, fine in zip(levels, levels[1:]):
            assert fine % coarse == 0, (resolution, levels)


def test_a_resolution_below_the_start_is_sampled_once():
    assert octree_levels(8) == (8,) == octree_levels(8, start=MIN_LEVEL_RESOLUTION)
    assert octree_levels(32, start=4) == (4, 8, 16, 32)


def test_the_octree_finds_exactly_the_cells_a_dense_sweep_finds():
    resolution, bounds = 32, 1.0
    volume = sample_sdf(sphere(), resolution, device="cpu", bounds=bounds, start=4)
    values, cells = brute_force(sphere(), resolution, bounds)
    assert volume.cells.equal(cells)
    assert torch.allclose(volume.values[volume.sampled], values[volume.sampled], atol=1e-5)


def test_sampling_asks_for_a_fraction_of_the_dense_grid():
    resolution = 192
    dense = (resolution + 1) ** 3
    volume = sample_sdf(sphere(0.7), resolution, device="cpu")
    assert volume.queries < dense / 3
    assert volume.cells.sum() < dense
    assert volume.cells.any()


def test_the_chunk_size_changes_nothing():
    whole = sample_sdf(sphere(), 24, device="cpu", start=6, chunk=1_000_000)
    parts = sample_sdf(sphere(), 24, device="cpu", start=6, chunk=17)
    assert whole.cells.equal(parts.cells)
    assert torch.equal(whole.values, parts.values)
    assert parts.queries == whole.queries


def test_every_extracted_corner_carries_a_queried_value():
    volume = sample_sdf(sphere(0.4), 48, device="cpu", start=6)
    corners = volume.extracted_cells()[:, None, :] + CORNERS
    assert volume.sampled[corners[..., 0], corners[..., 1], corners[..., 2]].all()
    positions = volume.corner_positions(volume.extracted_cells())
    assert positions.shape == corners.shape
    lattice = (positions + volume.bounds) / (2 * volume.bounds) * volume.resolution
    assert torch.equal(lattice, lattice.round())


def test_a_surface_outside_the_box_yields_no_cells():
    volume = sample_sdf(lambda points: points.norm(dim=-1) - 4.0, 64, device="cpu",
                        bounds=1.0, start=8)
    assert not volume.cells.any()
    assert volume.queries > 0


def test_a_saturating_field_still_yields_a_surface():
    """The decoder answers in logits, not distances: steep, and no unit gradient anywhere."""
    volume = sample_sdf(lambda points: torch.tanh(50.0 * (points.norm(dim=-1) - 0.4)),
                        96, device="cpu", start=12)
    assert volume.cells.any()
    radius = volume.corner_positions(volume.extracted_cells()).norm(dim=-1)
    assert 0.35 < radius.mean() < 0.45


def test_a_field_narrower_than_the_grid_is_stored_anyway():
    """A bf16 codec answers in bf16, and an index store refuses a narrower source."""
    for dtype in (torch.bfloat16, torch.float16):
        volume = sample_sdf(lambda points: (points.norm(dim=-1) - 0.4).to(dtype),
                            24, device="cpu", start=6)
        assert volume.values.dtype is torch.float32
        assert volume.cells.any()


def test_a_field_that_never_reaches_the_level_says_so():
    volume = sample_sdf(lambda points: torch.full((points.shape[0],), -3.0), 64,
                        device="cpu", start=8)
    assert not volume.cells.any()
    assert volume.span() == "-3.0000..-3.0000"


def test_a_band_chases_the_level_and_not_zero():
    """A cut the field never reaches must end the ladder, not refine every cell: the band
    is a distance to ``level``, so a level far outside the field's span has nothing near it
    and the request comes back as "no corner changed sign" instead of a dense sweep."""

    def soft(points):
        return torch.tanh(2.0 * (points.norm(dim=-1) - 0.4))

    def taken(level):
        return sample_sdf(soft, 192, device="cpu", bounds=1.01, level=level, band=0.95)

    unreachable = taken(3.0)
    assert not unreachable.cells.any()
    assert unreachable.queries == 97 ** 3, "the ladder refined past the level it cannot meet"
    assert taken(0.0).queries > unreachable.queries, "the band never chased the surface"


def test_the_level_selects_which_surface_is_extracted():
    def two_shells(points):
        radius = points.norm(dim=-1)
        return torch.minimum((radius - 0.3).abs() - 0.05, (radius - 0.8).abs() - 0.05)

    inner = sample_sdf(two_shells, 64, device="cpu", level=-0.01, start=8)
    outer = sample_sdf(two_shells, 64, device="cpu", level=-0.01, start=8, bounds=0.55)
    assert inner.cells.any() and outer.cells.any()
    assert inner.queries > outer.queries


def test_a_band_refines_the_cells_that_only_come_near_the_level():
    """A field that saturates can hide its surface in a cell whose corners are all on one
    side of it, so refining on sign changes alone stops before the surface. The band chases
    what only approaches the level, without falling back to a dense grid."""

    def ledge(points):
        near = (points.norm(dim=-1) - 0.6).abs() < 0.02
        return 1.3 - 0.8 * near

    plain = sample_sdf(ledge, 64, device="cpu", level=0.0, start=16)
    banded = sample_sdf(ledge, 64, device="cpu", level=0.0, start=16, band=0.9)
    assert not banded.cells.any()
    assert plain.queries < banded.queries < 65 ** 3 // 3


@pytest.mark.parametrize("kwargs", [
    {"resolution": 0},
    {"resolution": 64, "bounds": 0.0},
    {"resolution": 64, "chunk": 0},
    {"resolution": 64, "start": 0},
])
def test_nonsense_grids_are_refused(kwargs):
    with pytest.raises(ValueError):
        sample_sdf(sphere(), device="cpu", **{**{"resolution": 64}, **kwargs})


def test_a_cut_costs_the_volume_of_its_grid_and_says_so():
    """Each step is eight times the memory of the one below, which is what the request's
    own limit says; the number is what a refusal is quoted from."""
    assert volume_bytes(192) > 7 * volume_bytes(96)
    assert volume_bytes(96) > 7 * volume_bytes(48)


def test_a_cut_the_device_cannot_pay_for_is_refused_before_anything_is_allocated(monkeypatch):
    monkeypatch.setattr("thenoise.mesh.octree.spare_bytes", lambda device: 1 << 20)
    with pytest.raises(ValueError,
                       match=r"a cut at 192\^3 needs .* lower the voxel resolution"):
        sample_sdf(lambda points: points.norm(dim=-1) - 0.4, 192, device="cpu")


def test_a_level_that_outgrows_the_device_is_named_as_that_level(monkeypatch):
    """The grids are certain and the band is not: a level the field will not fit in is
    refused by its own name, and not blamed on a resolution that would have fitted."""
    rooms = iter([1 << 40, 1 << 40, 1 << 20])
    monkeypatch.setattr("thenoise.mesh.octree.spare_bytes", lambda device: next(rooms))
    with pytest.raises(ValueError, match=r"level 192 of a 192\^3 cut"):
        sample_sdf(lambda points: points.norm(dim=-1) - 0.4, 192, device="cpu")


def test_a_device_that_will_not_say_its_room_is_left_alone(monkeypatch):
    """cpu reports nothing, and the tests cut on cpu."""
    monkeypatch.setattr("thenoise.mesh.octree.spare_bytes", lambda device: None)
    volume = sample_sdf(lambda points: points.norm(dim=-1) - 0.4, 16, device="cpu",
                        start=8)
    assert volume.resolution == 16
