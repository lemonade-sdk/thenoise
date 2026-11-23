"""Mesh writers: OBJ, binary glTF, STL, with no third-party dependency."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

from .extract import Mesh

GLB_MAGIC = 0x46546C67
_GLTF_POSITION_TARGET = 34962
_GLTF_INDEX_TARGET = 34963


def _host(mesh: Mesh) -> Mesh:
    """Every writer starts here, and a glTF header cannot carry a ``NaN`` position."""
    if not bool(torch.isfinite(mesh.vertices).all()):
        raise ValueError("the sampled field left a position that is not a number")
    return mesh.cpu()


def write_obj(mesh: Mesh, path: str | Path) -> None:
    """``v`` lines then ``f`` lines, 1-based as OBJ counts."""
    if mesh.is_empty:
        raise ValueError("there is no mesh to write")
    local = _host(mesh)
    vertices = local.vertices.tolist()
    faces = local.faces.tolist()
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    Path(path).write_text("\n".join(lines) + "\n")


def _glb(mesh: Mesh) -> bytes:
    """One mesh, one primitive, no materials; glTF demands a POSITION min and max."""
    local = _host(mesh)
    positions = local.vertices.to(torch.float32).contiguous()
    indices = local.faces.to(torch.int32).contiguous().reshape(-1)
    blob = positions.numpy().tobytes() + indices.numpy().tobytes()
    index_offset = positions.numel() * 4
    bounds = positions.amin(dim=0).tolist()
    ceiling = positions.amax(dim=0).tolist()
    document = {
        "asset": {"version": "2.0", "generator": "thenoise"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": index_offset,
             "target": _GLTF_POSITION_TARGET},
            {"buffer": 0, "byteOffset": index_offset,
             "byteLength": len(blob) - index_offset, "target": _GLTF_INDEX_TARGET},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": positions.shape[0],
             "type": "VEC3", "min": bounds, "max": ceiling},
            {"bufferView": 1, "componentType": 5125, "count": indices.shape[0],
             "type": "SCALAR"},
        ],
    }
    head = json.dumps(document, separators=(",", ":")).encode("utf-8")
    head += b" " * (-len(head) % 4)
    tail = blob + b"\0" * (-len(blob) % 4)
    chunks = (struct.pack("<II", len(head), 0x4E4F534A) + head
              + struct.pack("<II", len(tail), 0x004E4942) + tail)
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(chunks)) + chunks


def write_glb(mesh: Mesh, path: str | Path) -> None:
    if mesh.is_empty:
        raise ValueError("there is no mesh to write")
    Path(path).write_bytes(_glb(mesh))


_STL_DTYPE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", 9), ("attributes", "<u2")])


def write_stl(mesh: Mesh, path: str | Path) -> None:
    """Binary STL: independent triangles with their own normals, which is what a cut is.
    Units are whatever the model's space means to whoever opens it."""
    if mesh.is_empty:
        raise ValueError("there is no mesh to write")
    local = _host(mesh)
    corners = local.vertices.to(torch.float32)[local.faces]
    span = corners[:, 1] - corners[:, 0]
    reach = corners[:, 2] - corners[:, 0]
    faces = corners.shape[0]
    records = np.zeros(faces, dtype=_STL_DTYPE)
    cross = torch.linalg.cross(span, reach)
    records["normal"] = (cross / cross.norm(dim=-1, keepdim=True).clamp(min=1e-12)).numpy()
    records["vertices"] = corners.reshape(faces, 9).numpy()
    Path(path).write_bytes(b"thenoise".ljust(80, b"\0") + struct.pack("<I", faces)
                           + records.tobytes())


def write_mesh(mesh: Mesh, path: str | Path, output_format: str | None = None) -> None:
    """The mesh in ``output_format``, or in the format ``path``'s extension promises."""
    if output_format is None:
        output_format = Path(path).suffix[1:].lower()
    if output_format == "glb":
        write_glb(mesh, path)
    elif output_format == "obj":
        write_obj(mesh, path)
    elif output_format == "stl":
        write_stl(mesh, path)
    else:
        raise ValueError(f"no mesh writer for {output_format!r}")


__all__ = ["Mesh", "write_glb", "write_mesh", "write_obj", "write_stl"]
