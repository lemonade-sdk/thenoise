"""The shape after it is cut: the declared finish chain, and what each filter does.

The assertions are made against spheres and detached tetrahedra cut by the real
extractor, so a filter that quietly opened the surface or ate the shape shows up here
rather than in a file somebody tries to print.
"""
import subprocess
import sys

import numpy as np
import pytest
import torch
from test_mesh_controller import DEFAULTS

from thenoise.mesh.extract import Mesh, extract_surface
from thenoise.mesh.finish import (CHAIN, DEFAULTS as OFF, FILTERS, FINISH_NAMES,
                                  MeshFinishOptions, filter_specs, finish_mesh,
                                  flat_options, options_from)
from thenoise.mesh.finish.cleanup import drop_islands
from thenoise.mesh.finish.simplify import reduce_to
from thenoise.mesh.octree import sample_sdf
from thenoise.mesh.spec import MeshSpec
from thenoise.utils.options import options_schema


def sphere(resolution=64, radius=0.4) -> Mesh:
    return extract_surface(sample_sdf(lambda points: points.norm(dim=-1) - radius,
                                      resolution, device="cpu", bounds=1.0, level=0.0,
                                      start=8))


def tetra(centre=(3.0, 3.0, 3.0), size=0.05) -> Mesh:
    """A closed piece of its own, nowhere near the origin."""
    corners = torch.tensor([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
                           dtype=torch.float32) * size
    corners += torch.tensor(centre, dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=torch.long)
    return Mesh(corners, faces)


def beside_the_shape(shape: Mesh, piece: Mesh) -> Mesh:
    return Mesh(torch.cat((shape.vertices, piece.vertices)),
                torch.cat((shape.faces, piece.faces + len(shape.vertices))))


def edge_multiplicity(faces) -> torch.Tensor:
    pairs = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    span = int(faces.max().item()) + 1
    keys = pairs.min(dim=1).values * span + pairs.max(dim=1).values
    return torch.unique(keys, return_counts=True)[1]


def pieces(mesh: Mesh) -> int:
    """Counted by hand, so the filter is not measured with its own algorithm."""
    parent = {}

    def find(v):
        while parent.setdefault(v, v) != v:
            v = parent[v]
        return v

    for face in mesh.faces.tolist():
        for a, b in zip(face, face[1:] + face[:1]):
            parent[find(a)] = find(b)
    return len({find(v) for v in parent})


def area(mesh: Mesh) -> float:
    corners = mesh.vertices.numpy()[mesh.faces.numpy()]
    return float(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                         corners[:, 2] - corners[:, 0]), axis=1).sum()
                 / 2.0)


class _NothingAsked:
    """What the chain gets when the request left every knob alone."""

    def __init__(self):
        self.stages = []

    def check(self) -> None:
        pass

    def progress(self, stage, *rest) -> None:
        self.stages.append(stage)


# -------------------------------------------------------------- the declaration


def test_the_chain_runs_in_the_order_it_is_declared():
    assert [spec.name for spec in CHAIN] == ["floaters", "simplify"]
    assert [spec.name for spec in filter_specs()] == [spec.name for spec in CHAIN]


def test_the_knobs_a_filter_reads_are_the_ones_it_declared():
    assert [spec.options for spec in CHAIN] == [("floaters",), ("triangles",)]
    assert {spec.name for spec in CHAIN} == set(FILTERS)


def test_no_knob_asks_for_anything_by_default():
    assert OFF == {"floaters": 0.0, "triangles": 1.0}
    assert options_schema(MeshFinishOptions)[0]["label"] == "Drop floaters"


def test_a_knob_outside_its_own_range_is_refused_by_name():
    for knob, bad in [("triangles", 1.5), ("triangles", 0.0), ("floaters", 2.0)]:
        with pytest.raises(ValueError, match=knob):
            MeshFinishOptions(**{knob: bad})


def test_the_declaration_needs_no_compiled_library_to_build_the_flags():
    """A wheel that is only needed to reduce must not be needed to offer reducing."""
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys; import thenoise.cli; "
         "print('fast_simplification' in sys.modules)"],
        capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "False"


def test_the_pane_is_told_the_words_the_bounds_and_the_defaults():
    described = {entry["name"]: entry for entry in options_schema(MeshFinishOptions)}
    assert described["triangles"] == {"name": "triangles", "label": "Triangles",
                                     "type": "number", "default": 1.0,
                                     "min": 0.001, "max": 1.0, "step": 0.01}
    assert described["floaters"]["max"] == 1.0
    assert described["floaters"]["ui_max"] == 0.02


# ------------------------------------------------------------------ the chain


def test_a_chain_nobody_asked_anything_of_hands_back_the_very_mesh():
    cut, observer = sphere(32), _NothingAsked()
    assert finish_mesh(cut, MeshFinishOptions(), observer.check, observer.progress) is cut
    assert observer.stages == []


def test_both_filters_run_and_the_declared_order_is_kept(monkeypatch):
    seen = []

    def record(name):
        def adapter(mesh, **_knobs):
            seen.append(name)
            return mesh

        return adapter

    for name in FILTERS:
        monkeypatch.setitem(FILTERS, name, record(name))
    cut = sphere(32)
    finish_mesh(cut, MeshFinishOptions(floaters=0.01, triangles=0.5),
                (observer := _NothingAsked()).check, observer.progress)
    assert seen == ["floaters", "simplify"]
    assert observer.stages == ["finishing", "finishing"]


# ---------------------------------------------------------------- floaters


def test_a_piece_that_is_not_the_shape_is_cut_away():
    """The share is the whole contract: below a piece's own share of the area it stays,
    above it the piece is gone and the shape is exactly what it was."""
    shape = sphere(48)
    with_piece = beside_the_shape(shape, tetra())
    junk = area(tetra()) / area(with_piece)
    assert pieces(with_piece) == 2

    assert pieces(drop_islands(with_piece, junk / 2)) == 2
    kept = drop_islands(with_piece, junk * 2)
    assert pieces(kept) == 1
    assert len(kept.faces) == len(shape.faces)
    assert len(kept.vertices) == len(shape.vertices)
    assert area(kept) == pytest.approx(area(shape), rel=1e-6)


def test_the_share_is_of_area_so_a_second_shape_survives():
    """Two spheres of one request are two pieces of comparable area: neither is junk."""
    two = beside_the_shape(sphere(48), sphere(48))
    assert len(drop_islands(two, 0.4).faces) == len(two.faces)


def test_zero_keeps_every_piece_because_only_the_request_knows():
    with_piece = beside_the_shape(sphere(32), tetra())
    assert drop_islands(with_piece, 0.0) is with_piece


def test_a_share_that_leaves_nothing_is_refused_rather_than_writing_nothing():
    with pytest.raises(ValueError, match="no shape left"):
        drop_islands(beside_the_shape(sphere(32), tetra()), 1.0)


def test_an_empty_cut_needs_no_components():
    empty = Mesh(torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.long))
    assert drop_islands(empty, 0.01) is empty


# ------------------------------------------------------------- the reduction


def test_the_budget_is_met_and_the_surface_stays_closed():
    cut, keep = sphere(64), 0.25
    reduced = reduce_to(cut, keep)
    assert len(reduced.faces) == max(4, round(len(cut.faces) * keep))
    assert torch.equal(edge_multiplicity(reduced.faces),
                       torch.full_like(edge_multiplicity(reduced.faces), 2))
    assert reduced.signed_volume() == pytest.approx(cut.signed_volume(), rel=0.01)


def test_a_hundredth_of_the_triangles_still_holds_a_closed_shape():
    """The floor of the ladder: a million triangles to ten thousand still encloses."""
    cut = sphere(128)
    reduced = reduce_to(cut, 0.01)
    assert torch.equal(edge_multiplicity(reduced.faces),
                       torch.full_like(edge_multiplicity(reduced.faces), 2))
    assert reduced.signed_volume() == pytest.approx(cut.signed_volume(), rel=0.05)


def test_keeping_everything_hands_back_the_very_mesh():
    cut = sphere(32)
    assert reduce_to(cut, 1.0) is cut


def test_the_mesh_comes_back_the_kind_the_cut_was():
    """The library works in float64 on the host; the writers see what they saw before."""
    cut = sphere(32)
    reduced = reduce_to(cut, 0.5)
    assert (reduced.vertices.dtype, reduced.faces.dtype) == (cut.vertices.dtype,
                                                            cut.faces.dtype)
    assert reduced.vertices.device == cut.vertices.device


# ---------------------------------------------------------------- the request


def test_the_finish_knobs_come_in_flat_and_leave_as_one_object():
    resolved = MeshSpec(image=b"x", floaters=0.01, triangles=0.25).resolve(DEFAULTS)
    assert resolved.finish == MeshFinishOptions(floaters=0.01, triangles=0.25)


def test_a_request_that_said_nothing_leaves_the_mesh_alone():
    assert MeshSpec(image=b"x").resolve(DEFAULTS).finish == MeshFinishOptions()


def test_a_number_the_field_cannot_hold_is_refused_before_anything_loads():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MeshSpec(image=b"x", triangles=3.0)


def test_the_knobs_read_off_anything_that_carries_them():
    class Holder:
        floaters = 0.02
        triangles = 0.4

    assert flat_options(Holder()) == {"floaters": 0.02, "triangles": 0.4}
    assert set(FINISH_NAMES) == {"floaters", "triangles"}
