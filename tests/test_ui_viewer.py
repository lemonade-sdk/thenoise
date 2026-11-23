from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from thenoise.mesh.extract import Mesh
from thenoise.mesh.spec import OUTPUT_FORMATS
from thenoise.mesh.write import write_glb, write_obj, write_stl

_UI = Path(__file__).resolve().parent.parent / "thenoise" / "ui" / "js"
node = shutil.which("node")

# A Prelude stands where helpers.js would: the viewer looks its canvas up by id at load.
_SHIM = """
const canvas = {getContext: () => null, addEventListener: () => {},
                clientWidth: 300, clientHeight: 300};
const $ = () => canvas;
const document = {createElement: () => canvas, getElementById: () => canvas};
const localStorage = {getItem: () => null, setItem: () => {}};
const window = {devicePixelRatio: 2, addEventListener: () => {}};
const requestAnimationFrame = () => 0;
const ResizeObserver = class { observe() {} };
const initToggle = () => {};
const fs = require('node:fs');

{viewer}

const read = (file, format) => {
  const buffer = fs.readFileSync(file);
  const bytes = buffer.buffer.slice(buffer.byteOffset,
                                    buffer.byteOffset + buffer.byteLength);
  try {
    const shape = parseMesh(bytes, format);
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < shape.positions.length; i += 3) {
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], shape.positions[i + axis]);
        max[axis] = Math.max(max[axis], shape.positions[i + axis]);
      }
    }
    return {triangles: shape.triangles, vertices: shape.vertices, radius: shape.radius,
            indices: shape.indices ? Array.from(shape.indices) : null, min, max};
  } catch (error) {
    return {error: error.message};
  }
};

// Where the camera would put each corner: view then projection, as the vertex shader does.
const framed = (file, format) => {
  const buffer = fs.readFileSync(file);
  const shape = parseMesh(buffer.buffer.slice(buffer.byteOffset,
                                              buffer.byteOffset + buffer.byteLength),
                          format);
  MESH_VIEW.center = shape.center;
  MESH_VIEW.radius = shape.radius;
  fitMesh();
  const yaw = MESH_REST.yaw, pitch = MESH_REST.pitch, reach = MESH_VIEW.distance;
  const eye = [shape.center[0] + reach * Math.cos(pitch) * Math.sin(yaw),
               shape.center[1] + reach * Math.sin(pitch),
               shape.center[2] + reach * Math.cos(pitch) * Math.cos(yaw)];
  const view = meshLook(eye), proj = meshPerspective(1.6);
  const times = (m, v) => [0, 1, 2, 3].map(row =>
    m[row] * v[0] + m[4 + row] * v[1] + m[8 + row] * v[2] + m[12 + row] * v[3]);
  const ndc = (p) => {
    const seen = times(view, [p[0], p[1], p[2], 1]);
    const clip = times(proj, seen);
    return [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], clip[3]];
  };
  const corners = [];
  for (let i = 0; i < shape.positions.length; i += 3) {
    corners.push(ndc([shape.positions[i], shape.positions[i + 1], shape.positions[i + 2]]));
  }
  return {center: ndc(shape.center), corners, distance: reach};
};

const asked = JSON.parse(process.argv[2]);
const out = {};
for (const [name, [file, format]] of Object.entries(asked)) out[name] = read(file, format);
out.camera = framed(asked.glb[0], asked.glb[1]);
console.log(JSON.stringify(out));
"""


def octahedron() -> Mesh:
    corners = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0],
                            [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    return Mesh(corners + torch.tensor([0.5, -1.5, 2.0]),
                torch.tensor([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                              [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]))


def written(mesh: Mesh, directory: Path) -> dict[str, tuple[Path, str]]:
    write_glb(mesh, directory / "shape.glb")
    write_obj(mesh, directory / "shape.obj")
    write_stl(mesh, directory / "shape.stl")
    corners = mesh.vertices.tolist()
    facets = "".join(
        "  facet normal 0 0 0\n    outer loop\n"
        + "".join("      vertex %.6f %.6f %.6f\n" % tuple(corners[c]) for c in face)
        + "    endloop\n  endfacet\n"
        for face in mesh.faces.tolist())
    (directory / "ascii.stl").write_text("solid shape\n" + facets + "endsolid shape\n")
    # A quad face, corner indices that carry texture and normal fields, a face counting back
    # from the end, and a seven-corner face: none of it comes from us, all of it comes from
    # elsewhere, and the seven corners are what overflows the starting index array.
    (directory / "quad.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n"
        "v 2 0 0\nv 2 1 0\nv 3 0 0\nv 3 1 0\nv 4 0 0\nv 4 1 0\nv 5 0 0\nv 5 1 0\n"
        "f 1/1/1 2/1/1 3/1/1 4/1/1\nf -4 -3 -2 -1\nf 5 6 7 9 10 11 12\n")
    (directory / "junk.glb").write_bytes(b"not a glb at all, not even close")
    (directory / "oob.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 9\n")
    return {"glb": (directory / "shape.glb", "glb"),
            "obj": (directory / "shape.obj", "obj"),
            "stl": (directory / "shape.stl", "stl"),
            "ascii": (directory / "ascii.stl", "stl"),   # a flavour of stl, not a format
            "quad": (directory / "quad.obj", "obj"),
            "junk": (directory / "junk.glb", "glb"),     # refused: not the bytes
            "oob": (directory / "oob.obj", "obj"),      # refused: points past the vertices
            "unknown": (directory / "shape.obj", "ply")}  # refused: not a format we view

def parsed(tmp_path: Path) -> dict:
    if node is None:
        pytest.skip("no node on this machine to run the viewer's parsers")
    asked = {name: [str(path), output_format]
             for name, (path, output_format) in written(octahedron(), tmp_path).items()}
    shim = tmp_path / "shim.js"
    shim.write_text(_SHIM.replace("{viewer}", (_UI / "viewer.js").read_text()))
    res = subprocess.run([node, str(shim), json.dumps(asked)], capture_output=True, text=True,
                         timeout=120)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_viewer_views_every_format_the_writer_has():
    script = (_UI / "viewer.js").read_text()
    declared = re.search(r"const MESH_VIEWED_FORMATS = \[([^\]]*)\]", script)
    assert declared, "viewer.js no longer lists the formats it reads"
    offered = [one.strip().strip("'\"") for one in declared[1].split(",")]
    assert offered == list(OUTPUT_FORMATS), offered


def test_viewer_reads_the_bytes_the_writer_makes(tmp_path):
    got = parsed(tmp_path)
    bounds = {"min": [-0.5, -2.5, 1.0], "max": [1.5, -0.5, 3.0]}
    for fmt in ("glb", "obj"):
        assert got[fmt]["triangles"] == 8, got[fmt]
        assert got[fmt]["vertices"] == 6, got[fmt]
        assert got[fmt]["indices"] is not None and len(got[fmt]["indices"]) == 24
        assert got[fmt]["min"] == pytest.approx(bounds["min"], abs=1e-5)
        assert got[fmt]["max"] == pytest.approx(bounds["max"], abs=1e-5)
        assert got[fmt]["radius"] == pytest.approx(1.0, abs=1e-5)


def test_viewer_keeps_an_stl_a_soup(tmp_path):
    got = parsed(tmp_path)
    for stl in ("stl", "ascii"):
        assert got[stl]["triangles"] == 8, got[stl]
        assert got[stl]["vertices"] == 24, got[stl]
        assert got[stl]["indices"] is None
        assert got[stl]["min"] == pytest.approx([-0.5, -2.5, 1.0], abs=1e-5)


def test_viewer_fans_a_face_it_was_never_written(tmp_path):
    got = parsed(tmp_path)["quad"]
    assert got["vertices"] == 12
    assert got["triangles"] == 2 + 2 + 5
    assert got["indices"][:12] == [0, 1, 2, 0, 2, 3, 8, 9, 10, 8, 10, 11]
    assert got["indices"][-6:] == [4, 9, 10, 4, 10, 11]


def test_viewer_says_no_in_words_the_stage_can_show(tmp_path):
    got = parsed(tmp_path)
    assert "not a glb" in got["junk"]["error"]
    assert "nothing here views" in got["unknown"]["error"]
    # A face past the end of the vertex list draws nothing and would say nothing at all.
    assert "not there" in got["oob"]["error"]


def test_viewer_frames_the_shape_it_was_given(tmp_path):
    camera = parsed(tmp_path)["camera"]
    assert len(camera["corners"]) == 6, "nothing reached the camera to be judged"
    assert camera["center"][:2] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert camera["distance"] > 0
    for corner in camera["corners"]:
        assert corner[3] > 0, f"{corner} would be behind the eye"
        assert abs(corner[0]) < 1 and abs(corner[1]) < 1, f"{corner} would be off the canvas"
        assert -1 < corner[2] < 1, f"{corner} would be clipped by the depth range"
    depth = [corner[2] for corner in camera["corners"]]
    assert max(depth) - min(depth) > 1e-4, "near and far cannot tell the corners apart"
    spread = [abs(corner[0]) for corner in camera["corners"]]
    assert max(spread) < 0.95, "the fit leaves nothing around the shape"
