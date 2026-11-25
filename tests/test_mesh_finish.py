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
from thenoise.mesh.finish import cleanup
from thenoise.mesh.finish.cleanup import drop_islands
from thenoise.mesh.finish.simplify import reduce_to
from thenoise.mesh.octree import sample_sdf
from thenoise.mesh.spec import MeshSpec
from thenoise.postprocess.options import PostprocessOptions
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


def diagonal(mesh: Mesh) -> float:
    """The diagonal of the space a mesh occupies: what `floaters` measures pieces against."""
    return float((mesh.vertices.amax(dim=0) - mesh.vertices.amin(dim=0)).norm())


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
    # 0.001 stays the floor a request may ask for; 0.01 is where the slider starts, so that
    # its steps land on 1.0 instead of stopping at 0.991.
    assert described["triangles"] == {"name": "triangles", "label": "Triangles",
                                      "type": "number", "default": 1.0, "min": 0.001,
                                      "ui_min": 0.01, "max": 1.0, "step": 0.01}
    assert described["floaters"] == {"name": "floaters", "label": "Drop floaters",
                                     "type": "number", "default": 0.0, "min": 0.0,
                                     "max": 1.0, "step": 0.01}


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
    assert observer.stages == ["finishing: floaters", "finishing: simplify"]


def test_a_chain_that_never_returns_still_names_the_filter_it_entered(caplog):
    """The one piece of evidence a hung job leaves is the last line it wrote, so each filter
    says what it was handed before it starts, not only what it gave back when it finishes."""
    cut, keep = sphere(32), MeshFinishOptions(floaters=0.01, triangles=0.5)
    with caplog.at_level("INFO", logger="thenoise.mesh.finish.chain"):
        finish_mesh(cut, keep, (observer := _NothingAsked()).check, observer.progress)
    lines = [record.getMessage() for record in caplog.records]
    faces = len(cut.faces)
    assert [line for line in lines if line.endswith("triangles")] == [
        f"finishing floaters over {faces} triangles",
        f"finishing simplify over {faces} triangles"]
    assert [line.split(" in ")[0] for line in lines if " triangles in " in line] == [
        f"finishing floaters: {faces} -> {faces} triangles",
        f"finishing simplify: {faces} -> {round(faces * keep.triangles)} triangles"]


# ---------------------------------------------------------------- floaters


def test_a_piece_that_is_not_the_shape_is_cut_away():
    """Below a piece's own reach it stays, above it the piece is gone, and the shape is exactly
    what it was."""
    shape, junk = sphere(48), tetra()
    with_piece = beside_the_shape(shape, junk)
    small = diagonal(junk) / diagonal(shape)
    assert pieces(with_piece) == 2
    assert 0.1 < small < 0.2

    assert pieces(drop_islands(with_piece, small / 2)) == 2
    kept = drop_islands(with_piece, small * 2)
    assert pieces(kept) == 1
    assert len(kept.faces) == len(shape.faces)
    assert len(kept.vertices) == len(shape.vertices)
    assert area(kept) == pytest.approx(area(shape), rel=1e-6)


def test_a_thin_part_is_not_dust_how_little_area_it_holds():
    """The ruler is a diagonal because a piece can span the shape and hold almost no area: this
    sheet is 0.02% of it and reaches 58% as far, so an area share would have eaten it."""
    shape = sphere(48)
    corners = torch.tensor([[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.001]],
                           dtype=torch.float32)
    sheet = Mesh(corners, torch.tensor([[0, 1, 2]]))
    with_sheet = beside_the_shape(shape, sheet)
    assert area(sheet) < 0.0005 * area(shape)
    assert pieces(drop_islands(with_sheet, 0.5)) == 2
    assert pieces(drop_islands(with_sheet, 0.6)) == 1


def test_two_shapes_reaching_the_same_far_are_both_kept():
    """Two spheres of one request reach equally far: neither is junk at any threshold."""
    two = beside_the_shape(sphere(48), sphere(48))
    assert len(drop_islands(two, 0.9).faces) == len(two.faces)


def test_zero_keeps_every_piece_because_only_the_request_knows():
    with_piece = beside_the_shape(sphere(32), tetra())
    assert drop_islands(with_piece, 0.0) is with_piece


def test_the_whole_ruler_keeps_the_widest_piece_and_only_that():
    """The top of the range is a setting, not an error: one piece always ties the widest."""
    shape = sphere(32)
    kept = drop_islands(beside_the_shape(shape, tetra()), 1.0)
    assert pieces(kept) == 1
    assert len(kept.faces) == len(shape.faces)


def test_an_empty_cut_needs_no_components():
    empty = Mesh(torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.long))
    assert drop_islands(empty, 0.01) is empty


def strip(triangles: int) -> Mesh:
    """Triangles in a row, neighbours sharing an edge: one piece, and a label entering at one
    end has the whole strip to cross before it reaches the other."""
    vertices = torch.tensor([[float(one // 2), float(one % 2), 0.0]
                             for one in range(triangles + 2)], dtype=torch.float32)
    return Mesh(vertices, torch.tensor([[one, one + 1, one + 2] for one in range(triangles)]))



def test_a_strip_long_enough_to_crawl_is_still_one_piece(monkeypatch):
    """The pathology the rounds cannot be trusted with: a chain of triangles, where a label
    advances a little each round. Bounded, and the pieces are still exactly right."""
    strand = strip(2000)
    assert pieces(strand) == 1
    monkeypatch.setattr(cleanup, "ROUNDS", 3)
    assert pieces(drop_islands(strand, 0.0001)) == 1
    assert pieces(drop_islands(strand, 0.5)) == 1


def test_the_round_budget_costs_the_partition_nothing(monkeypatch):
    """The bound on the rounds is a promise about time, not about the answer: whatever the
    rounds left apart is merged exactly, so one round and thirty give the same pieces."""
    shape, junk = sphere(48), tetra()
    with_piece = beside_the_shape(shape, junk)
    share = diagonal(junk) / diagonal(shape) * 2
    monkeypatch.setattr(cleanup, "ROUNDS", 1)
    assert pieces(drop_islands(with_piece, share)) == 1
    assert len(drop_islands(with_piece, share).faces) == len(shape.faces)



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


def test_every_filter_knob_lands_where_a_slider_can_put_it():
    """The pane counts whole ticks of the step, so both edges of a knob's range have to sit on
    that grid: a min of 0.001 with step 0.01 would tick to a tenth of a step, and a rounded min
    is a slider offering a value the server rejects."""
    for holder in (MeshFinishOptions, PostprocessOptions):
        for entry in options_schema(holder):
            if entry["type"] != "number":
                continue
            ticks = 10 ** min(6, len(str(entry["step"]).partition(".")[2]))
            low = entry.get("ui_min", entry["min"])
            for value in (entry["default"], low, entry.get("ui_max", entry["max"])):
                assert abs(value * ticks - round(value * ticks)) < 1e-9, (
                    f"{holder.__name__}.{entry['name']} cannot be ticked to {value}")
