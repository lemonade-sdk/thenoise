import json
import struct

import numpy as np

import pytest
import torch

from thenoise.mesh.extract import Mesh, empty_mesh
from thenoise.mesh.write import write_glb, write_mesh, write_obj, write_stl


def triangle() -> Mesh:
    return Mesh(vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int64))


def test_obj_counts_from_one(tmp_path):
    path = tmp_path / "mesh.obj"
    write_obj(triangle(), path)
    lines = path.read_text().split()
    assert lines[:3] == ["v", "0.000000", "0.000000"]
    assert lines[-4:] == ["f", "1", "2", "3"]


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
    assert document["meshes"][0]["primitives"][0]["attributes"] == {"POSITION": 0}
    assert document["accessors"][0]["type"] == "VEC3"
    assert document["accessors"][0]["count"] == 3
    assert document["accessors"][1]["count"] == 3
    positions = raw[bin_offset + 8:bin_offset + 8 + 36]
    assert struct.unpack("<3f", positions[:12]) == (0.0, 0.0, 0.0)
    assert struct.unpack("<I", raw[bin_offset + 8 + 36:bin_offset + 8 + 40]) == (0,)


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
    from thenoise.mesh.octree import sample_sdf
    from thenoise.mesh.extract import extract_surface

    sphere = extract_surface(sample_sdf(lambda p: p.norm(dim=-1) - 0.4, 24,
                                        device="cpu", start=8))
    path = tmp_path / "sphere.stl"
    write_stl(sphere, path)
    records = np.frombuffer(path.read_bytes()[84:], dtype=np.dtype([
        ("normal", "<f4", 3), ("vertices", "<f4", 9), ("attributes", "<u2")]))
    centre = np.stack([records["vertices"][:, 0:3], records["vertices"][:, 3:6],
                       records["vertices"][:, 6:9]], axis=1).mean(axis=1)
    away = centre / np.linalg.norm(centre, axis=1, keepdims=True)
    assert ((records["normal"] * away).sum(axis=1) > 0.9).all()


def test_a_degenerate_triangle_gets_no_normal_rather_than_a_nan(tmp_path):
    path = tmp_path / "flat.stl"
    write_stl(Mesh(vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                          [2.0, 0.0, 0.0]]),
                   faces=torch.tensor([[0, 1, 2]])), path)
    assert struct.unpack("<3f", path.read_bytes()[84:96]) == (0.0, 0.0, 0.0)


def test_stl_is_written_by_the_extension_too(tmp_path):
    write_mesh(triangle(), tmp_path / "a.stl")
    assert (tmp_path / "a.stl").read_bytes()[80:84] == struct.pack("<I", 1)
