import json
import struct

import numpy as np

import pytest
import torch

from conftest import sphere_cut
from thenoise.mesh.extract import Mesh, NORMAL_CREASE_DEGREES, empty_mesh
from thenoise.mesh.write import write_glb, write_mesh, write_obj, write_stl


def triangle() -> Mesh:
    return Mesh(vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int64))


def test_obj_counts_from_one(tmp_path):
    path = tmp_path / "mesh.obj"
    write_obj(triangle(), path)
    rows = [line.split() for line in path.read_text().splitlines()]
    assert rows[0] == ["v", "0.000000", "0.000000", "0.000000"]
    assert rows[3] == ["vn", "0.000000", "0.000000", "1.000000"]
    assert rows[-1] == ["f", "1//1", "2//2", "3//3"]


def test_glb_is_a_binary_glTF_of_positions_and_triangles(tmp_path):
    path = tmp_path / "shape.glb"
    write_glb(triangle(), path)
    raw = path.read_bytes()
    magic, version, total = struct.unpack("<III", raw[:12])
    assert (magic, version) == (0x46546C67, 2)
    assert total == len(raw)
    json_length, json_type = struct.unpack("<II", raw[12:20])
    assert json_type == 0x4E4F534A and json_length % 4 == 0
    document = json.loads(raw[20:20 + json_length])
    bin_offset = 20 + json_length
    bin_length, bin_type = struct.unpack("<II", raw[bin_offset:bin_offset + 8])
    assert bin_type == 0x004E4942
    assert bin_offset + 8 + bin_length <= total
    assert document["meshes"][0]["primitives"][0]["attributes"] == {"POSITION": 0, "NORMAL": 1}
    assert document["accessors"][0]["type"] == "VEC3"
    assert document["accessors"][0]["count"] == 3
    assert document["accessors"][1]["type"] == "VEC3"
    assert document["accessors"][1]["count"] == 3
    assert document["accessors"][2]["count"] == 3
    positions = raw[bin_offset + 8:bin_offset + 8 + 36]
    assert struct.unpack("<3f", positions[:12]) == (0.0, 0.0, 0.0)
    assert struct.unpack("<3f", raw[bin_offset + 8 + 36:bin_offset + 8 + 48]) == (0.0, 0.0, 1.0)
    assert struct.unpack("<I", raw[bin_offset + 8 + 72:bin_offset + 8 + 76]) == (0,)


def test_glb_promises_the_positions_extent_because_the_format_demands_it(tmp_path):
    path = tmp_path / "s.glb"
    write_glb(Mesh(vertices=torch.tensor([[-1.0, 0.0, 2.0], [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0]]),
                   faces=torch.tensor([[0, 1, 2]])), path)
    raw = path.read_bytes()
    length = struct.unpack("<I", raw[12:16])[0]
    document = json.loads(raw[20:20 + length])
    assert document["accessors"][0]["min"] == [-1.0, 0.0, 0.0]
    assert document["accessors"][0]["max"] == [0.0, 1.0, 2.0]


def test_the_extension_decides_when_nothing_was_said(tmp_path):
    write_mesh(triangle(), tmp_path / "a.obj")
    write_mesh(triangle(), tmp_path / "b.glb")
    assert (tmp_path / "a.obj").read_text().startswith("v ")
    assert (tmp_path / "b.glb").read_bytes()[:4] == b"glTF"


def test_a_format_nobody_writes_is_refused_not_silently_mislabelled(tmp_path):
    with pytest.raises(ValueError, match="no mesh writer for 'ply'"):
        write_mesh(triangle(), tmp_path / "x.ply", "ply")


def test_an_empty_mesh_is_refused_before_it_looks_like_a_success(tmp_path):
    for name in ("empty.glb", "empty.obj"):
        with pytest.raises(ValueError, match="no mesh to write"):
            write_mesh(empty_mesh(), tmp_path / name)


def test_stl_gives_every_triangle_its_own_record(tmp_path):
    path = tmp_path / "shape.stl"
    write_stl(triangle(), path)
    raw = path.read_bytes()
    assert raw[:8] == b"thenoise"
    assert len(raw) == 84 + 50
    assert struct.unpack("<I", raw[80:84])[0] == 1
    record = struct.unpack("<12fH", raw[84:134])
    assert record[:3] == (0.0, 0.0, 1.0)
    assert record[3:6] == (0.0, 0.0, 0.0)
    assert record[6:9] == (1.0, 0.0, 0.0)
    assert record[9:12] == (0.0, 1.0, 0.0)
    assert record[12] == 0


def test_stl_normals_follow_the_winding(tmp_path):
    path = tmp_path / "sphere.stl"
    write_stl(sphere_cut(24), path)
    records = np.frombuffer(path.read_bytes()[84:], dtype=np.dtype([
        ("normal", "<f4", 3), ("vertices", "<f4", 9), ("attributes", "<u2")]))
    centre = np.stack([records["vertices"][:, 0:3], records["vertices"][:, 3:6],
                       records["vertices"][:, 6:9]], axis=1).mean(axis=1)
    away = centre / np.linalg.norm(centre, axis=1, keepdims=True)
    assert ((records["normal"] * away).sum(axis=1) > 0.9).all()


def test_a_faceless_triangle_is_not_written_at_all(tmp_path):
    """A triangle with no area is a spike or a hole to whoever opens the file, and the
    reducer leaves a few thousand of them behind."""
    path = tmp_path / "spike.stl"
    write_stl(Mesh(vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
                   faces=torch.tensor([[0, 1, 2], [0, 1, 3]])), path)
    assert struct.unpack("<I", path.read_bytes()[80:84]) == (1,)


def test_a_surface_of_only_faceless_triangles_is_refused(tmp_path):
    flat = Mesh(vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                faces=torch.tensor([[0, 1, 2]]))
    with pytest.raises(ValueError, match="faceless"):
        write_glb(flat, tmp_path / "flat.glb")


def test_a_thin_triangle_is_written_when_it_still_covers_ground(tmp_path):
    """A face is judged by its shape and never by its size, which is how a facet of
    something flat survives."""
    path = tmp_path / "facet.obj"
    write_obj(_flat(apex=0.5), path)
    text = path.read_text()
    assert text.count("\nf ") == 2


def test_a_needle_is_not_written_though_it_has_area(tmp_path):
    """Thinner than about a third of a degree, and its normal is a coin flip: a reader calls
    that degenerate data, so the face goes and the point only it needed goes with it."""
    path = tmp_path / "needle.obj"
    write_obj(_flat(apex=0.0004), path)
    text = path.read_text()
    assert text.count("\nf ") == 1
    assert len([one for one in text.splitlines() if one.startswith("v ")]) == 3


def _flat(apex: float) -> Mesh:
    """A wide triangle and one stretched along its base to ``apex`` tall, both in the same
    plane: the pair the reducer leaves behind, at whatever scale the shape was written."""
    return Mesh(vertices=torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0],
                                       [0.0, 8.0, 0.0], [0.5, apex, 0.0]]),
                faces=torch.tensor([[0, 1, 2], [1, 3, 0]]))


def test_normals_smooth_a_sphere_toward_the_shape_it_was_cut_from():
    sphere = sphere_cut(48)
    normals = sphere.smooth_normals()
    away = sphere.vertices / sphere.vertices.norm(dim=-1, keepdim=True)
    angle = torch.rad2deg(torch.acos((normals * away).sum(dim=-1).clamp(-1.0, 1.0)))
    assert normals.norm(dim=-1).min() > 0.999999
    assert float(angle.median()) < 2.0
    assert float(angle.max()) < NORMAL_CREASE_DEGREES


def test_normals_do_not_average_across_a_fold():
    """A bisector on every point of a hard edge is what rounds a shape into a pebble."""
    fold = Mesh(vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
                                       [1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
                faces=torch.tensor([[0, 2, 3], [0, 3, 1], [0, 1, 4], [1, 5, 4]]))
    normals = fold.smooth_normals()
    side = torch.linalg.cross(fold.vertices[2] - fold.vertices[0],
                              fold.vertices[3] - fold.vertices[0])
    up = torch.linalg.cross(fold.vertices[1] - fold.vertices[0],
                            fold.vertices[4] - fold.vertices[0])
    side, up = side / side.norm(), up / up.norm()
    assert torch.allclose(normals[2], side, atol=1e-6)
    assert torch.allclose(normals[4], up, atol=1e-6)
    for crease in (0, 1):
        against = torch.stack([(normals[crease] * side).sum(), (normals[crease] * up).sum()])
        assert against.abs().max() > 0.99, "the crease point got a bisector"


def test_a_jittered_cut_still_faces_somewhere():
    """Noise tips faces over and can flatten one, and a zero normal is a black patch."""
    rough = sphere_cut(24)
    drawn = torch.Generator().manual_seed(4)
    rough = Mesh(rough.vertices + torch.randn(rough.vertices.shape, generator=drawn) * 0.02,
                 rough.faces)
    lengths = rough.smooth_normals().norm(dim=-1)
    assert torch.isfinite(lengths).all()
    assert float(lengths.min()) > 0.999999


def test_stl_is_written_by_the_extension_too(tmp_path):
    write_mesh(triangle(), tmp_path / "a.stl")
    assert (tmp_path / "a.stl").read_bytes()[80:84] == struct.pack("<I", 1)


def test_a_mesh_still_graded_writes_the_same_file(tmp_path):
    """A cut comes off the device inside a graph; a writer that forgot one of the two tensors
    would fail the moment they are indexed together."""
    graded = Mesh(triangle().vertices.clone().requires_grad_(), triangle().faces)
    written = write_obj(graded, tmp_path / "graded.obj")
    assert not written.vertices.requires_grad and not written.faces.requires_grad
    assert (tmp_path / "graded.obj").read_text().count("\nf ") == 1


@pytest.mark.parametrize("writer,kind", [(write_glb, "glb"), (write_obj, "obj"),
                                         (write_stl, "stl"), (write_mesh, "glb")])
def test_a_file_is_not_written_from_a_field_that_overflowed(tmp_path, writer, kind):
    """A glTF header cannot carry the ``NaN`` this mesh would put in it."""
    out = tmp_path / f"shape.{kind}"
    with pytest.raises(ValueError, match="not a number"):
        writer(Mesh(vertices=torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0],
                                           [0.0, 1.0, 0.0]]),
                    faces=torch.tensor([[0, 1, 2]])), out)
    assert not out.exists()
