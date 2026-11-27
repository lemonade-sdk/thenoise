import pytest
import torch

from thenoise.mesh.dual_grid import corner_ring, dual_grid_mesh
from thenoise.sparse import SparseTensor

GRID = (1, 4, 4, 4)


def filled(shape=GRID) -> SparseTensor:
    cells = torch.cartesian_prod(*(torch.arange(v) for v in shape[1:]))
    coords = torch.cat([torch.zeros(len(cells), 1), cells], dim=1).to(torch.int32)
    return SparseTensor(torch.zeros(len(cells), 1), coords, shape)


def crossings(cells: dict, shape=GRID) -> torch.Tensor:
    """``[M, 3]`` over the whole grid: the axes each listed voxel crosses."""
    flag = torch.zeros(*shape[1:], 3, dtype=torch.bool)
    for (x, y, z), axes in cells.items():
        for axis in axes:
            flag[x, y, z, axis] = True
    return flag.reshape(-1, 3)


def sheet(axis: int, at: int, size: int = 3, shape=(1, 8, 8, 8)) -> dict:
    """A ``size x size`` patch of crossings in the plane ``axis == at``."""
    cells = {}
    for p in range(size):
        for q in range(size):
            cell = [at, at, at]
            cell[(axis + 1) % 3], cell[(axis + 2) % 3] = p, q
            cells[tuple(cell)] = (axis,)
    return cells


def edge_uses(mesh) -> torch.Tensor:
    stride = mesh.vertices.shape[0]
    every = mesh.faces[:, [[0, 1], [1, 2], [2, 0]]]
    undirected = every.min(-1).values * stride + every.max(-1).values
    return torch.bincount(undirected.reshape(-1))


def directed_edge_faults(mesh) -> int:
    """Edges two faces walk the same way: a fold, where a surface turns inside out."""
    stride = mesh.vertices.shape[0]
    steps = mesh.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    return int((torch.bincount(steps[:, 0] * stride + steps[:, 1]) > 1).sum())


def unshared_edges(mesh) -> int:
    return int((edge_uses(mesh) == 1).sum())


def test_a_crossing_owns_the_quad_at_its_corner():
    lattice = filled()
    mesh = dual_grid_mesh(lattice, torch.zeros(len(lattice), 3),
                          crossings({(1, 1, 1): (0,)}))[0]

    assert mesh.faces.shape == (2, 3)
    ring = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 1)]
    rows = {(x * GRID[2] + y) * GRID[3] + z for x, y, z in ring}
    assert set(mesh.faces.flatten().tolist()) == rows


def test_a_quad_whose_corner_is_absent_is_never_made():
    lattice = filled()
    keep = torch.ones(len(lattice), dtype=torch.bool)
    keep[(1 * GRID[2] + 1) * GRID[3] + 2] = False       # the ring's far corner
    mesh = dual_grid_mesh(lattice.subset(keep), torch.zeros(int(keep.sum()), 3),
                          crossings({(1, 1, 1): (0,)})[keep])[0]
    assert mesh.is_empty


def test_nothing_crossed_leaves_nothing_to_draw():
    lattice = filled()
    mesh = dual_grid_mesh(lattice, torch.zeros(len(lattice), 3),
                          torch.zeros(len(lattice), 3, dtype=torch.bool))[0]
    assert mesh.is_empty


def test_a_crossing_at_the_edge_of_the_grid_owns_no_outside_neighbour():
    lattice = filled()
    mesh = dual_grid_mesh(lattice, torch.zeros(len(lattice), 3),
                          crossings({(3, 3, 3): (0, 1, 2)}))[0]
    assert mesh.is_empty


def test_the_corner_weights_choose_the_diagonal():
    lattice = filled()
    flag = crossings({(1, 1, 1): (0,)})
    ring = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 1)]
    rows = torch.tensor([(x * GRID[2] + y) * GRID[3] + z for x, y, z in ring])

    weight = torch.zeros(len(lattice))
    weight[rows[[0, 2]]] = 1.0
    along = dual_grid_mesh(lattice, torch.zeros(len(lattice), 3), flag, weight)[0]
    assert frozenset(along.faces[0].tolist()) & frozenset(along.faces[1].tolist()) \
        == frozenset(rows[[0, 2]].tolist())

    weight = torch.zeros(len(lattice))
    weight[rows[[1, 3]]] = 1.0
    across = dual_grid_mesh(lattice, torch.zeros(len(lattice), 3), flag, weight)[0]
    assert frozenset(across.faces[0].tolist()) & frozenset(across.faces[1].tolist()) \
        == frozenset(rows[[1, 3]].tolist())


def test_the_offset_and_the_bounds_place_the_vertex():
    lattice = filled()
    vertex = torch.zeros(len(lattice), 3)
    vertex[0] = torch.tensor([0.5, -0.25, 1.0])
    mesh = dual_grid_mesh(lattice, vertex, crossings({(0, 0, 0): (0,)}), bounds=2.0)[0]
    step = 4.0 / GRID[1]
    assert torch.allclose(mesh.vertices[0],
                          torch.tensor([-2.0 + 0.5 * step, -2.0 - 0.25 * step,
                                        -2.0 + 1.0 * step]))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_sheet_has_the_area_and_the_rim_it_should(axis):
    lattice = filled((1, 8, 8, 8))
    mesh = dual_grid_mesh(lattice, torch.zeros(len(lattice), 3),
                          crossings(sheet(axis, at=1, size=3), (1, 8, 8, 8)))[0]

    step = 1.0 / 8
    assert mesh.faces.shape == (18, 3)
    assert int(edge_uses(mesh).max()) == 2
    assert unshared_edges(mesh) == 4 * 3
    assert torch.allclose(mesh.face_areas().sum(), torch.tensor(9 * step * step), atol=1e-6)
    normal = mesh.face_normals()
    assert torch.allclose(normal.abs()[:, axis], torch.ones(18), atol=1e-4)
    assert bool((normal[:, axis] > 0).all() or (normal[:, axis] < 0).all())


def test_the_sheets_of_the_three_axes_face_the_same_way():
    cells = {}
    for axis, at in ((0, 1), (1, 3), (2, 5)):
        cells |= sheet(axis, at=at, size=2, shape=(1, 8, 8, 8))
    lattice = filled((1, 8, 8, 8))
    mesh = dual_grid_mesh(lattice, torch.zeros(len(lattice), 3),
                          crossings(cells, (1, 8, 8, 8)))[0]

    assert mesh.faces.shape == (3 * 8, 3)
    every = mesh.face_normals()
    axis = every.abs().argmax(dim=-1)
    assert torch.equal(axis.sort().values, torch.tensor([0, 1, 2]).repeat_interleave(8))
    # One rule for all three: the same sign against the axis each sheet faces.
    assert len({int(torch.sign(row[axis[i]])) for i, row in enumerate(every)}) == 1


def test_three_sheets_meeting_at_a_corner_wind_alongside_each_other():
    lattice = filled((1, 2, 2, 2))
    flag = torch.zeros(len(lattice), 3, dtype=torch.bool)
    flag[0] = True                       # one voxel, crossed along every axis
    mesh = dual_grid_mesh(lattice, torch.zeros(len(lattice), 3), flag)[0]

    assert mesh.faces.shape == (6, 3)
    assert directed_edge_faults(mesh) == 0
    # Eighteen edge uses: three quad diagonals, three edges where sheets meet — each twice —
    # and the six rim edges of a corner that nothing continues past.
    assert unshared_edges(mesh) == 6


def test_a_reused_ring_folds_the_surface(monkeypatch):
    lattice = filled((1, 2, 2, 2))
    flag = torch.zeros(len(lattice), 3, dtype=torch.bool)
    flag[0] = True
    vertex = torch.zeros(len(lattice), 3)

    assert directed_edge_faults(dual_grid_mesh(lattice, vertex, flag)[0]) == 0
    ring = corner_ring(lattice.device)
    monkeypatch.setattr("thenoise.mesh.dual_grid.corner_ring",
                        lambda device: ring[[0, 0, 1]].to(device))
    assert directed_edge_faults(dual_grid_mesh(lattice, vertex, flag)[0]) > 0


def test_two_batch_entries_stay_two_meshes():
    one = filled()
    coords = torch.cat([one.coords, one.coords], dim=0)
    coords[len(one):, 0] = 1
    both = SparseTensor(torch.zeros(len(coords), 1), coords, (2, *GRID[1:]))
    flag = torch.zeros(len(coords), 3, dtype=torch.bool)
    flag[0] = True
    flag[len(one)] = True

    first, second = dual_grid_mesh(both, torch.zeros(len(coords), 3), flag)
    assert first.faces.shape == second.faces.shape == (6, 3)
    # Each mesh numbers its own voxels from zero: either stands on its own to be written out.
    assert torch.equal(first.faces, second.faces)
    assert torch.equal(first.vertices, second.vertices)
