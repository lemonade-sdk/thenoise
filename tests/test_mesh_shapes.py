"""What a sampled shape leaves behind, and what a later request can do with it."""
import os
from pathlib import Path

import pytest
import torch
from test_mesh_controller import DEFAULTS, FakeShapeModel, PNG

from thenoise.jobs import Cancellation
from thenoise.mesh.controller import MeshPipelineController
from thenoise.mesh.shapes import SHAPES_LIMIT, ShapeStore
from thenoise.mesh.spec import MeshSpec


class _Watching(FakeShapeModel):
    """The fake, keeping the latents the codec was handed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cut_from: list[torch.Tensor] = []

    def decode_mesh(self, latents, params):
        self.cut_from.append(latents.clone())
        return super().decode_mesh(latents, params)


def _spec(**overrides) -> dict:
    return {"seed": 7, **overrides}


def _render(model, store, name="job_alpha", output_format="glb", **overrides):
    controller = MeshPipelineController(model)
    spec = controller.resolve(MeshSpec(output_format=output_format, **_spec(**overrides)))
    target = store.root / f"{name}.{output_format}"
    how = controller.render(spec, Cancellation(), lambda *_: None, target, store)
    return spec, target, how


def test_a_sampled_shape_is_written_beside_the_file_it_asked_for(tmp_path):
    store = ShapeStore(tmp_path)
    model = _Watching()
    _render(model, store, image=PNG)
    assert store.path("job_alpha").exists()
    assert torch.equal(store.load("job_alpha"), model.cut_from[0].cpu())


def test_naming_a_shape_cuts_it_again_without_picture_or_solver(tmp_path):
    store = ShapeStore(tmp_path)
    _render(_Watching(), store, image=PNG)
    kept = torch.full((1, 3, 4), 7.0)
    torch.save({"latents": kept}, store.path("job_alpha"))

    model = _Watching()
    spec, target, how = _render(model, store, name="job_beta", octree_resolution=32,
                           from_shape="job_alpha")
    assert spec.mode == "shape"
    assert "encode" not in model.events and "init" not in model.events
    assert "step" not in model.events
    assert torch.equal(model.cut_from[0], kept)
    assert target.exists()
    assert "job_alpha left behind" in how


def test_a_shape_that_was_pruned_is_refused_with_the_window_it_fell_out_of(tmp_path):
    store = ShapeStore(tmp_path)
    with pytest.raises(ValueError, match=f"newest {SHAPES_LIMIT}"):
        _render(_Watching(), store, from_shape="job_never_sampled")


def test_recutting_needs_a_place_where_shapes_are_kept(tmp_path):
    model = _Watching()
    controller = MeshPipelineController(model)
    spec = controller.resolve(MeshSpec(**_spec(from_shape="job_alpha")))
    with pytest.raises(ValueError, match="jobs directory"):
        controller.render(spec, Cancellation(), lambda *_: None, tmp_path / "shape.glb")


def test_only_the_newest_shapes_are_kept_and_nothing_else_is_touched(tmp_path):
    store = ShapeStore(tmp_path, limit=3)
    stranger = tmp_path / "someone_elses.pt"
    stranger.write_bytes(b"not ours")
    for number in range(5):
        store.save(f"job_{number}", torch.zeros(1, 2, 2))
        os.utime(store.path(f"job_{number}"), (number, number))
    store.prune()

    assert [one for one, _ in store.shapes()] == ["job_4", "job_3", "job_2"]
    assert stranger.exists()


def test_a_shape_is_addressed_by_a_job_id_and_nothing_that_walks(tmp_path):
    store = ShapeStore(tmp_path)
    for address in ("../../etc/passwd", "shape", "job../x", ""):
        with pytest.raises(ValueError, match="job id"):
            store.path(address)


def test_the_same_image_and_seed_are_sampled_once(tmp_path):
    """Another resolution costs the volume only; another seed costs the solve, not the image."""
    store = ShapeStore(tmp_path)
    model = _Watching()
    controller = MeshPipelineController(model)
    first = controller.resolve(MeshSpec(**_spec(image=PNG)))
    said = controller.render(first, Cancellation(), lambda *_: None,
                             store.root / "job_16.glb", store)
    solved = model.events.count("step")
    assert solved > 0
    assert "sampled in" in said

    again = controller.render(first, Cancellation(), lambda *_: None,
                              store.root / "job_32.glb", store)
    assert model.events.count("encode") == 1
    assert model.events.count("step") == solved, "the second cut reuses the sampled shape"
    assert model.events.count("decode") == 2
    assert "nothing sampled" in again, "a tenth of a second has to say why it was quick"

    other = controller.resolve(MeshSpec(**_spec(image=PNG, seed=8)))
    controller.render(other, Cancellation(), lambda *_: None,
                      store.root / "job_other_seed.glb", store)
    assert model.events.count("step") > solved, "another seed is another shape"
    assert model.events.count("decode") == 3


def test_a_shape_and_its_picture_are_refused_together():
    with pytest.raises(ValueError, match="either image or from_shape"):
        MeshSpec(image=PNG, from_shape="job_alpha").resolve(DEFAULTS)


def test_the_echo_names_the_shape_a_request_was_cut_from():
    resolved = MeshSpec(from_shape="job_alpha").resolve(DEFAULTS)
    assert resolved.mode == "shape"
    assert resolved.public()["from_shape"] == "job_alpha"
    assert MeshSpec().resolve(DEFAULTS).public()["from_shape"] is None


def test_a_model_that_could_not_be_asked_for_the_mode_still_recuts(tmp_path):
    model = _Watching()
    model.caps = {"i2d": True}
    controller = MeshPipelineController(model)
    assert controller.resolve(MeshSpec(**_spec(from_shape="job_alpha"))).mode == "shape"
    with pytest.raises(ValueError, match="multi-view"):
        controller.resolve(MeshSpec(seed=7, image=[PNG, PNG]))


def test_a_recut_still_finishes_and_writes_what_the_request_asked_for(tmp_path):
    store = ShapeStore(tmp_path)
    _render(_Watching(), store, image=PNG)
    model = _Watching()
    _, target, how = _render(model, store, name="job_stl", output_format="stl",
                        from_shape="job_alpha", octree_resolution=8, scale=2.0)
    assert target.read_bytes().startswith(b"thenoise"), "the request's format, not the default"
    assert "left behind" in how
