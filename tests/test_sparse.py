import pytest
import torch
import torch.nn.functional as F

from thenoise.sparse import (HashMap, SparseChannel2Spatial, SparseSpatial2Channel,
                             SparseTensor, SubmanifoldConv3d, child_offsets, flat_keys,
                             kernel_offsets, maps as sparse_maps, pixel_shuffle_3d,
                             pixel_unshuffle_3d, sparse_mean_pool3d,
                             sparse_upsample3d_nearest)

GRID = (1, 7, 6, 5)
BOX = (1, 4, 4, 4)


def lattice(channels: int = 3, shape=GRID, keep: float = 0.5, seed: int = 0) -> SparseTensor:
    """A random lattice over a small grid: never empty, and by default not the whole grid."""
    batch, width, height, depth = shape
    cells = torch.cartesian_prod(torch.arange(batch), torch.arange(width),
                                 torch.arange(height), torch.arange(depth))
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(cells.shape[0], generator=generator) < keep
    mask[0] = True
    return SparseTensor(torch.randn(int(mask.sum()), channels, generator=generator),
                        cells[mask], shape)


def full_lattice(channels: int, shape=GRID) -> SparseTensor:
    batch, width, height, depth = shape
    cells = torch.cartesian_prod(torch.arange(batch), torch.arange(width),
                                 torch.arange(height), torch.arange(depth))
    feats = torch.arange(cells.shape[0] * channels, dtype=torch.float32).reshape(-1, channels)
    return SparseTensor(feats, cells, shape)


def at(dense: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """The dense tensor's channels at each voxel's cell."""
    return dense[coords[:, 0].long(), :, coords[:, 1].long(), coords[:, 2].long(),
                 coords[:, 3].long()]


def sorted_rows(coords: torch.Tensor) -> torch.Tensor:
    """Rows in cell order, so two lattices can be compared without minding their row order."""
    key = ((coords[:, 0] * 64 + coords[:, 1]) * 64 + coords[:, 2]) * 64 + coords[:, 3]
    return coords[torch.argsort(key)]


def test_a_lattice_does_not_live_on_two_devices():
    with pytest.raises(ValueError, match="one device"):
        SparseTensor(torch.empty(len(lattice(1)), 2, device="meta"),
                     lattice(1).coords, GRID)


def test_a_coordinate_is_one_number_once_the_grid_is_fixed():
    assert flat_keys(lattice(channels=1).coords, GRID).unique().numel() == len(lattice(1))


def test_the_hash_map_answers_a_miss_with_minus_one():
    x = lattice(channels=1)
    keys = flat_keys(x.coords, x.shape)
    every = torch.cartesian_prod(*(torch.arange(v) for v in x.shape))
    absent = flat_keys(every, x.shape)[~torch.isin(flat_keys(every, x.shape), keys)]
    probe = HashMap(keys, torch.arange(len(x), dtype=torch.int32))

    assert bool((probe.lookup(keys) == torch.arange(len(x), dtype=torch.int32)).all())
    assert absent.numel() and bool((probe.lookup(absent) == -1).all())


def test_the_hash_map_looks_the_same_whatever_its_chunk_is(monkeypatch):
    x = lattice(channels=1)
    keys, values = flat_keys(x.coords, x.shape), torch.arange(len(x), dtype=torch.int32)
    whole = HashMap(keys, values).lookup(keys)
    monkeypatch.setattr(sparse_maps, "CHUNK", 7)
    assert bool((HashMap(keys, values).lookup(keys) == whole).all())


def test_the_neighbour_map_names_the_voxel_at_every_tap():
    x = lattice(channels=1)
    neighbor = x.maps.neighbor((3, 3, 3))
    rows = {tuple(int(v) for v in coord): i for i, coord in enumerate(x.coords)}
    offsets = kernel_offsets((3, 3, 3), (1, 1, 1), torch.device("cpu")).tolist()
    for i, coord in enumerate(x.coords.tolist()):
        for tap, (dx, dy, dz) in enumerate(offsets):
            assert int(neighbor[i, tap]) == rows.get(
                (coord[0], coord[1] + dx, coord[2] + dy, coord[3] + dz), -1)


def test_a_reused_lattice_keeps_its_maps_and_a_new_one_does_not():
    x = lattice(channels=2)
    assert x.replace(torch.zeros_like(x.feats)).maps is x.maps
    assert x.subset(x.feats[:, 0] > 0).maps is not x.maps


@pytest.mark.parametrize("dilation", [1, 2])
def test_a_submanifold_conv_is_a_dense_conv_on_the_same_support(dilation):
    x = lattice(channels=4, seed=dilation)
    conv = SubmanifoldConv3d(4, 6, kernel_size=3, dilation=dilation).eval()
    with torch.no_grad():
        out = conv(x)
        dense = F.conv3d(x.to_dense(), conv.weight, conv.bias, padding=dilation,
                         dilation=dilation)
        assert torch.allclose(out.feats, at(dense, x.coords), atol=1e-6)


def test_a_convolution_leaves_the_lattice_alone():
    x = lattice(channels=3)
    out = SubmanifoldConv3d(3, 8, kernel_size=3)(x)
    assert torch.equal(out.coords, x.coords) and out.shape == x.shape


def test_an_empty_neighbourhood_reads_only_its_own_cell():
    x = SparseTensor(torch.ones(1, 3), torch.tensor([[0, 2, 2, 2]]), (1, 5, 5, 5))
    conv = SubmanifoldConv3d(3, 2, kernel_size=3).eval()
    with torch.no_grad():
        assert torch.allclose(conv(x).feats.T,
                              conv.weight[:, :, 1, 1, 1] @ x.feats.T + conv.bias[:, None],
                              atol=1e-6)


def test_mean_pooling_averages_the_cells_that_share_a_parent():
    x = full_lattice(2)
    pooled, index = sparse_mean_pool3d(x, 2)
    batch, width, height, depth = x.shape
    assert pooled.shape == (batch, -(-width // 2), -(-height // 2), -(-depth // 2))

    sums = pooled.feats.new_zeros(pooled.feats.shape)
    sums.index_add_(0, index, x.feats)
    counts = torch.zeros(len(pooled), 1)
    counts.index_add_(0, index, torch.ones(len(x), 1))
    assert torch.allclose(pooled.feats, sums / counts, atol=1e-6)


def test_nearest_upsample_returns_each_child_to_its_own_parent():
    x = full_lattice(2)
    pooled, index = sparse_mean_pool3d(x, 2)
    assert torch.equal(sparse_upsample3d_nearest(pooled.feats, index), pooled.feats[index])
    assert len(sparse_upsample3d_nearest(pooled.feats, index)) == len(x)


def test_channel_to_spatial_is_a_pixel_shuffle_on_a_full_lattice():
    x = full_lattice(8, BOX)
    up = SparseChannel2Spatial(2)(x)
    dense = pixel_shuffle_3d(x.to_dense(), 2)
    assert len(up) == len(x) * 8 and up.feats.shape[1] == 1
    assert torch.allclose(at(dense, up.coords).reshape(-1), up.feats.reshape(-1))


def test_spatial_to_channel_is_a_pixel_unshuffle_on_a_full_lattice():
    x = full_lattice(1, BOX)
    down = SparseSpatial2Channel(2)(x)
    dense = pixel_unshuffle_3d(x.to_dense(), 2)
    assert len(down) * 8 == len(x) and down.feats.shape[1] == 8
    assert torch.allclose(at(dense, down.coords), down.feats)


def test_a_childs_channel_block_is_x_fastest():
    offsets = child_offsets(2, torch.device("cpu"))
    assert offsets[1].tolist() == [1, 0, 0] and offsets[2].tolist() == [0, 1, 0]
    assert offsets[4].tolist() == [0, 0, 1]


def test_channel_to_spatial_and_spatial_to_channel_are_inverses():
    x = full_lattice(8)
    up = SparseChannel2Spatial(2)(x)
    assert up.shape == (1, x.shape[1] * 2, x.shape[2] * 2, x.shape[3] * 2)
    assert len(up) == len(x) * 8
    back = SparseSpatial2Channel(2)(up)
    assert torch.equal(back.coords, x.coords) and torch.equal(back.feats, x.feats)


def test_a_subdivision_mask_keeps_exactly_the_cells_it_marks():
    x = full_lattice(8)
    keep = torch.zeros(len(x), 8, dtype=torch.bool)
    keep[::3, 0], keep[1::3, 5] = True, True
    up = SparseChannel2Spatial(2)(x, keep)
    assert len(up) == int(keep.sum())

    scale = torch.tensor([1, 2, 2, 2], dtype=torch.int32)
    offsets = child_offsets(2, x.device)

    def shift(offset):
        """A lattice offset with the batch axis left alone."""
        return torch.cat([torch.zeros(1, 1, dtype=torch.int32), offset[None]], dim=1)

    expect = torch.cat([x.coords[::3] * scale + shift(offsets[0]),
                        x.coords[1::3] * scale + shift(offsets[5])])
    assert torch.equal(sorted_rows(up.coords), sorted_rows(expect))


def test_an_incomplete_parent_is_refused_rather_than_invented():
    up = SparseChannel2Spatial(2)(full_lattice(8))
    one_short = torch.ones(len(up), dtype=torch.bool)
    one_short[-1] = False
    with pytest.raises(ValueError, match="children"):
        SparseSpatial2Channel(2)(up.subset(one_short))


def test_pixel_shuffle_and_unshuffle_round_trip():
    dense = torch.randn(2, 3, 6, 4, 8)
    assert torch.equal(pixel_shuffle_3d(pixel_unshuffle_3d(dense, 2), 2), dense)


def test_a_refinement_hands_back_the_parent_split_row_it_used():
    x = full_lattice(16)
    born, source = SparseChannel2Spatial(2).plan(x)
    split = x.feats.reshape(len(x) * 8, -1)
    assert torch.equal(born.feats, split[source])
    assert born.shape == (1, x.shape[1] * 2, x.shape[2] * 2, x.shape[3] * 2)
    assert len(born) == len(x) * 8


def test_a_lattice_round_trips_through_a_dense_grid():
    x = lattice(channels=3)
    again = SparseTensor.from_dense(x.to_dense(), support=None)
    assert len(again) == x.shape[1] * x.shape[2] * x.shape[3]

    kept = SparseTensor.from_dense(x.to_dense(),
                                   support=x.to_dense().abs().sum(dim=1) > 0)
    assert torch.equal(sorted_rows(kept.coords.int()), sorted_rows(x.coords))
    assert torch.allclose(kept.feats.reshape(-1), x.feats.reshape(-1), atol=1e-6)
