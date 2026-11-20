import pytest
import torch

from thenoise.mesh.octree import CORNERS, MIN_LEVEL_RESOLUTION, octree_levels, sample_sdf


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
    assert octree_levels(384) == (48, 96, 192, 384)
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


def test_the_level_selects_which_surface_is_extracted():
    def two_shells(points):
        radius = points.norm(dim=-1)
        return torch.minimum((radius - 0.3).abs() - 0.05, (radius - 0.8).abs() - 0.05)

    inner = sample_sdf(two_shells, 64, device="cpu", level=-0.01, start=8)
    outer = sample_sdf(two_shells, 64, device="cpu", level=-0.01, start=8, bounds=0.55)
    assert inner.cells.any() and outer.cells.any()
    assert inner.queries > outer.queries


@pytest.mark.parametrize("kwargs", [
    {"resolution": 0},
    {"resolution": 64, "bounds": 0.0},
    {"resolution": 64, "chunk": 0},
    {"resolution": 64, "start": 0},
])
def test_nonsense_grids_are_refused(kwargs):
    with pytest.raises(ValueError):
        sample_sdf(sphere(), device="cpu", **{**{"resolution": 64}, **kwargs})
