"""Mesh writers: OBJ, binary glTF, STL, with no third-party dependency."""
from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np
import torch

from .extract import Mesh

GLB_MAGIC = 0x46546C67
_GLTF_POSITION_TARGET = 34962
_GLTF_INDEX_TARGET = 34963

logger = logging.getLogger(__name__)

_SPAN = 1 << 20


def _spans(count: int):
    for low in range(0, count, _SPAN):
        yield low, min(low + _SPAN, count)


def _host(mesh: Mesh) -> Mesh:
    """Every writer starts here: on the host, finite, and without the faces no reader can draw."""
    if mesh.is_empty:
        raise ValueError("there is no mesh to write")
    if not bool(torch.isfinite(mesh.vertices).all()):
        raise ValueError("the sampled field left a position that is not a number")
    local = mesh.cpu().without_degenerate()
    if local.is_empty:
        raise ValueError("every triangle was faceless: there is no surface to write")
    if local.faces.shape[0] != mesh.faces.shape[0]:
        logger.info("write: %d of %d triangles have no shape to draw and were left out",
                    mesh.faces.shape[0] - local.faces.shape[0], mesh.faces.shape[0])
    return local


def write_obj(mesh: Mesh, path: str | Path) -> Mesh:
    """``v`` and ``vn`` lines then ``f v//vn`` lines, 1-based as OBJ counts."""
    local = _host(mesh)
    normals = local.smooth_normals()
    corners = (local.faces + 1).tolist()
    with open(path, "w", encoding="utf-8") as text:
        for low, high in _spans(local.vertices.shape[0]):
            text.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n"
                               for x, y, z in local.vertices[low:high].tolist()))
        for low, high in _spans(normals.shape[0]):
            text.write("".join(f"vn {x:.6f} {y:.6f} {z:.6f}\n"
                               for x, y, z in normals[low:high].tolist()))
        for low, high in _spans(len(corners)):
            text.write("".join(f"f {a}//{a} {b}//{b} {c}//{c}\n"
                               for a, b, c in corners[low:high]))
    return local


def write_glb(mesh: Mesh, path: str | Path) -> Mesh:
    """One mesh, one primitive, no materials; glTF demands a POSITION min and max."""
    local = _host(mesh)
    # The body is written in this order, so the header's offsets cannot drift from the bytes.
    arrays = {"POSITION": local.vertices.to(torch.float32).contiguous(),
              "NORMAL": local.smooth_normals().to(torch.float32).contiguous()}
    indices = local.faces.to(torch.int32).contiguous().reshape(-1)

    views, accessors, reached = [], [], 0
    for place, values in enumerate(arrays.values()):
        views.append({"buffer": 0, "byteOffset": reached, "byteLength": values.numel() * 4,
                      "target": _GLTF_POSITION_TARGET})
        accessors.append({"bufferView": place, "componentType": 5126,
                          "count": values.shape[0], "type": "VEC3",
                          "min": values.amin(dim=0).tolist(),
                          "max": values.amax(dim=0).tolist()})
        reached += values.numel() * 4
    views.append({"buffer": 0, "byteOffset": reached, "byteLength": indices.numel() * 4,
                  "target": _GLTF_INDEX_TARGET})
    accessors.append({"bufferView": len(arrays), "componentType": 5125,
                      "count": indices.shape[0], "type": "SCALAR"})
    body = reached + indices.numel() * 4

    document = {
        "asset": {"version": "2.0", "generator": "thenoise"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {name: place
                                                   for place, name in enumerate(arrays)},
                                    "indices": len(accessors) - 1}]}],
        "buffers": [{"byteLength": body}],
        "bufferViews": views,
        "accessors": accessors,
    }
    head = json.dumps(document, separators=(",", ":")).encode("utf-8")
    # Every length the header carries is known from the mesh, so the body streams after it.
    head_pad, body_pad = -len(head) % 4, -body % 4
    with open(path, "wb") as out:
        out.write(struct.pack("<III", GLB_MAGIC, 2,
                              12 + 8 + len(head) + head_pad + 8 + body + body_pad))
        out.write(struct.pack("<II", len(head) + head_pad, 0x4E4F534A) + head
                  + b" " * head_pad)
        out.write(struct.pack("<II", body + body_pad, 0x004E4942))
        for values in (*arrays.values(), indices):
            for low, high in _spans(values.shape[0]):
                out.write(values[low:high].numpy().tobytes())
        out.write(b"\0" * body_pad)
    return local


_STL_DTYPE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", 9), ("attributes", "<u2")])


def write_stl(mesh: Mesh, path: str | Path) -> Mesh:
    """Binary STL: independent triangles with their own normals, which is what a cut is.
    Units are whatever the model's space means to whoever opens it."""
    local = _host(mesh)
    faces = local.faces.shape[0]
    positions = local.vertices.to(torch.float32)
    with open(path, "wb") as out:
        out.write(b"thenoise".ljust(80, b"\0") + struct.pack("<I", faces))
        for low, high in _spans(faces):
            corners = positions[local.faces[low:high]]
            # Rebuilt per chunk rather than taken off the mesh: no whole-mesh temporary here.
            cross = torch.linalg.cross(corners[:, 1] - corners[:, 0],
                                       corners[:, 2] - corners[:, 0])
            records = np.empty(corners.shape[0], dtype=_STL_DTYPE)
            records["attributes"] = 0
            records["normal"] = (cross / cross.norm(dim=-1, keepdim=True)
                                 .clamp(min=1e-12)).numpy()
            records["vertices"] = corners.reshape(-1, 9).numpy()
            records.tofile(out)
    return local


def write_mesh(mesh: Mesh, path: str | Path,
               output_format: str | None = None) -> Mesh:
    """The mesh in ``output_format``, or in the format ``path``'s extension promises."""
    if output_format is None:
        output_format = Path(path).suffix[1:].lower()
    if output_format == "glb":
        return write_glb(mesh, path)
    if output_format == "obj":
        return write_obj(mesh, path)
    if output_format == "stl":
        return write_stl(mesh, path)
    raise ValueError(f"no mesh writer for {output_format!r}")


__all__ = ["write_glb", "write_mesh", "write_obj", "write_stl"]
