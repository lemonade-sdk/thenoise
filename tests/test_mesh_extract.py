import math

import torch

from thenoise.mesh.extract import ORIENTATION, TETRAHEDRA, Mesh, empty_mesh, extract_surface

from thenoise.mesh.octree import sample_sdf

_CUBE_CORNERS = torch.tensor([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)], dtype=torch.float32)


def volume_of(sdf, resolution=64, bounds=1.0, level=0.0, start=8):
    return extract_surface(sample_sdf(sdf, resolution, device="cpu", bounds=bounds,
                                      level=level, start=start))


def sphere(radius=0.4):
    return lambda points: points.norm(dim=-1) - radius


def box(side=0.6):
    return lambda points: points.abs().amax(dim=-1) - side / 2


def edge_multiplicity(faces):
    pairs = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    span = int(faces.max().item()) + 1
    keys = pairs.min(dim=1).values * span + pairs.max(dim=1).values
    return torch.unique(keys, return_counts=True)[1]


def test_the_tetrahedra_tile_the_cube_without_gaps():
    corners = _CUBE_CORNERS[TETRAHEDRA]
    steps = torch.stack((corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0],
                         corners[:, 3] - corners[:, 0]), dim=1)
    determinants = torch.linalg.det(steps)
    assert torch.count_nonzero(determinants) == 6
    assert torch.allclose(determinants.abs().sum() / 6.0, torch.tensor(1.0))
    assert torch.equal(ORIENTATION, torch.sign(determinants).to(torch.long))


def test_a_sphere_comes_back_closed_and_facing_outward():
    mesh = volume_of(sphere())
    assert not mesh.is_empty
    assert torch.equal(edge_multiplicity(mesh.faces),
                       torch.full_like(edge_multiplicity(mesh.faces), 2))
    true_volume = 4.0 / 3.0 * math.pi * 0.4 ** 3
    assert 0.97 * true_volume < mesh.signed_volume() <= true_volume * 1.001


def test_a_cube_comes_back_closed():
    mesh = volume_of(box())
    assert torch.equal(edge_multiplicity(mesh.faces),
                       torch.full_like(edge_multiplicity(mesh.faces), 2))
    assert 0.95 * 0.6 ** 3 < mesh.signed_volume() < 1.05 * 0.6 ** 3


def test_the_level_is_where_the_surface_is():
    shifted = lambda points: points.norm(dim=-1)
    mesh = volume_of(shifted, level=0.25)
    assert 0.97 * 4.0 / 3.0 * math.pi * 0.25 ** 3 < mesh.signed_volume() \
        <= 4.0 / 3.0 * math.pi * 0.25 ** 3


def test_no_face_leaves_the_box():
    mesh = volume_of(sphere(0.9), resolution=48)
    assert mesh.vertices.abs().amax() <= 1.0


def test_every_vertex_lies_on_the_surface_within_a_cell():
    resolution, radius = 64, 0.4
    mesh = volume_of(sphere(radius), resolution=resolution, start=16)
    error = (mesh.vertices.norm(dim=-1) - radius).abs()
    assert error.amax() < 2 * 2.0 * 1.0 / resolution


def test_an_empty_volume_extracts_nothing():
    volume = sample_sdf(lambda points: points.norm(dim=-1) + 1.0, 32, device="cpu", start=8)
    mesh = extract_surface(volume)
    assert mesh.is_empty and mesh.vertices.shape == (0, 3)
    assert empty_mesh().is_empty


def test_extraction_is_deterministic():
    first = volume_of(sphere(), resolution=40)
    second = volume_of(sphere(), resolution=40)
    assert torch.equal(first.vertices, second.vertices)
    assert torch.equal(first.faces, second.faces)


def test_a_shape_smaller_than_a_cell_still_yields_a_closed_hull():
    mesh = volume_of(sphere(0.02), resolution=16, start=4)
    assert mesh.faces.shape[0] >= 4
    assert torch.equal(edge_multiplicity(mesh.faces),
                       torch.full_like(edge_multiplicity(mesh.faces), 2))


def test_signed_volume_of_a_hand_built_tetrahedron():
    corners = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    assert torch.allclose(Mesh(corners, faces).signed_volume(), torch.tensor(1 / 6))


def test_every_pattern_of_every_tetrahedron_faces_out_of_the_solid():
    from thenoise.mesh.extract import _FLIP, _TET_TABLES

    for tet in range(TETRAHEDRA.shape[0]):
        table = _TET_TABLES[_FLIP[tet].item()]
        corners = _CUBE_CORNERS[TETRAHEDRA[tet]]
        for code in range(1, 15):
            inside = [(code >> bit) & 1 for bit in range(4)]
            solid = corners[[i for i, on in enumerate(inside) if on]]
            void = corners[[i for i, on in enumerate(inside) if not on]]
            outward = void.mean(0) - solid.mean(0)
            for slot in range(2):
                edges = table[code, slot]
                if bool((edges < 0).any()):
                    continue
                cut = torch.stack([corners[a] + (corners[b] - corners[a]) * 0.5
                                   for a, b in edges.tolist()])
                normal = torch.linalg.cross(cut[1] - cut[0], cut[2] - cut[0])
                assert (normal * outward).sum() > 0, (tet, code, slot)
