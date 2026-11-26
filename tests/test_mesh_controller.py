import io
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from PIL import Image

from thenoise.jobs import Cancellation, JobCancelled
from thenoise.mesh.controller import MeshPipelineController, _sampling_bytes
from thenoise.mesh.memory import volume_bytes
from thenoise.mesh.extract import Mesh
from thenoise.mesh.spec import MeshSpec
from thenoise.models.base import Conditioning, DiffusionModel
from thenoise.utils.attention import ATTENTION_METHODS

PNG = io.BytesIO()
Image.new("RGB", (24, 16), (3, 4, 5)).save(PNG, format="PNG")
PNG = PNG.getvalue()

OTHER = io.BytesIO()
Image.new("RGB", (24, 16), (9, 9, 9)).save(OTHER, format="PNG")
OTHER = OTHER.getvalue()

DEFAULTS = {
    # The base's own prefs, so a preference added to every medium needs no line here.
    **DiffusionModel.DEFAULT_PREFS,
    "output_format": "glb",
    "num_latents": 6,
    "steps": 3,
    "guidance_scale": 5.0,
    "sampler": "euler",
    "octree_resolution": 16,
    "bounds": 1.0,
    "mc_level": 0.0,
}


def picture(bytes_=PNG):
    return {"image": bytes_}


def triangle_mesh() -> Mesh:
    return Mesh(vertices=torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]),
                faces=torch.tensor([[0, 1, 2]]))


class FakeMemory:
    """Which weights a stage held, and when: a step outside its stage is on evicted memory."""

    def __init__(self, events=None):
        self.held: list[tuple[str, ...]] = []
        self.sized: list[tuple[tuple[str, ...], int, int]] = []
        self.events = events if events is not None else []

    @contextmanager
    def stage(self, *names, pixels=None, reserve=None):
        assert pixels or reserve, "a stage must size its transient"
        self.sized.append((names, pixels, reserve))
        self.held.append(names)
        self.events.append("hold:" + "+".join(names))
        try:
            yield type("S", (), {"run": staticmethod(lambda work, *a, **k: work(*a, **k))})
        finally:
            self.events.append("release:" + "+".join(names))


class FakeShapeModel:
    name = "fake-shape"
    MEDIUM = "mesh"
    ATTENTION_METHODS = ATTENTION_METHODS
    resolve_attention = DiffusionModel.resolve_attention
    attention_routing_blocked = staticmethod(lambda: None)
    lora_dir = "/loras"
    default_prefs = DEFAULTS
    dtype = torch.float32
    device = "cpu"

    def __init__(self, cancel_during=None, caps=None):
        self.events: list[str] = []
        self.memory = FakeMemory(self.events)
        self.dit = object()
        self.lora_calls: list[tuple] = []
        self.seen_args = []
        self.decode_params = []
        self.cancel_during = cancel_during
        self.token = None
        # A tower that numbers views is the exception, so a test says which model it means.
        self.caps = caps if caps is not None else {"i2d": True, "mv2d": 4, "t2d": False}

    def capability(self, medium):
        return dict(self.caps)

    def lora_components(self):
        return {"dit": self.dit}

    def list_loras(self):
        return []

    def switch_loras(self, specs, component, role):
        self.events.append(f"loras:{role}")
        self.lora_calls.append((role, component, tuple(specs or ())))

    def encode_prompt(self, args):
        self.events.append("encode")
        self.seen_args.append(args)
        return Conditioning(cond=torch.zeros(1, 4, 8), null=torch.ones(1, 4, 8))

    def init_latents(self, params):
        self.events.append("init")
        return torch.zeros(1, params.num_latents, 4)

    def fuse_text(self, cond):
        return cond

    def begin_sampling(self, schedule):
        pass

    def schedule(self, params):
        from thenoise.samplers import sigma_steps
        return sigma_steps([1.0, 0.0], torch.device("cpu"), torch.float32)

    def denoise_step(self, latents, t, cond, guidance_scale, i):
        self.events.append("step")
        if self.cancel_during == "step":
            self.token.cancel()
        return torch.zeros_like(latents)

    def decode_mesh(self, latents, params):
        self.events.append("decode")
        self.decode_params.append(params)
        return triangle_mesh()


def render(model, progress=lambda *a: None, **overrides):
    controller = MeshPipelineController(model)
    resolved = controller.resolve(MeshSpec(**{**picture(), **overrides}))
    out = Path(f"/tmp/thenoise-mesh-{len(model.events)}.{resolved.output_format}")
    out.unlink(missing_ok=True)
    model.token = Cancellation()
    controller.render(resolved, model.token, progress, out)
    return out


def test_a_shape_reports_the_parts_of_itself_that_would_otherwise_be_silent(tmp_path):
    """A job that says nothing while it cuts the volume is a hang to whoever watches it."""
    seen = []
    model = FakeShapeModel()
    render(model, progress=lambda stage, *_: seen.append(stage))
    assert seen[0] == "conditioning"
    assert seen[-1] == "cutting", "the volume must not be reported as the solver's last step"
    assert "sampling" in seen[1:-1]



def test_a_shape_stage_asks_for_room_in_the_units_it_actually_moves():
    """A volume is not an image: only the conditioning stage may speak in pixels."""
    model = FakeShapeModel()
    render(model, octree_resolution=64, latents=512)
    sized = {names: (pixels, reserve) for names, pixels, reserve in model.memory.sized}
    assert sized[("codec",)] == (None, volume_bytes(64))
    assert sized[("dit",)] == (None, _sampling_bytes(512))
    pixels, reserve = sized[("conditioner",)]
    assert pixels and reserve is None, "a picture is the one thing measured in pixels"


def test_the_three_components_are_brought_in_the_order_the_pipeline_needs_them():
    model = FakeShapeModel()
    render(model)
    assert model.memory.held == [("conditioner",), ("dit",), ("codec",)]
    assert model.events == ["hold:conditioner", "encode", "release:conditioner",
                            "init", "hold:dit", "loras:dit", "step", "release:dit",
                            "hold:codec", "decode", "release:codec"]


def test_a_resolved_request_reaches_the_volume_knobs():
    model = FakeShapeModel()
    render(model, octree_resolution=48, bounds=1.25, mc_level=0.01)
    assert model.decode_params[0].octree_resolution == 48
    assert model.decode_params[0].bounds == 1.25
    assert model.decode_params[0].mc_level == 0.01


def test_the_pictures_reach_the_conditioner_as_one_request():
    model = FakeShapeModel()
    render(model)
    args = model.seen_args[0]
    assert [image.width for image in args.image] == [24]


def test_views_reach_the_conditioner_in_the_order_they_were_sent():
    model = FakeShapeModel()
    render(model, image=[PNG, OTHER])
    assert [tuple(image.getpixel((0, 0))) for image in model.seen_args[0].image] \
        == [(3, 4, 5), (9, 9, 9)]


def test_more_views_than_the_model_numbers_are_refused():
    model = FakeShapeModel()
    with pytest.raises(ValueError, match="at most 4 views"):
        MeshPipelineController(model).resolve(
            MeshSpec(image=[PNG] * 5))


def test_a_model_with_no_multi_view_tower_refuses_the_second_picture():
    """A quietly dropped image is a shape nobody asked for: refused, not ignored."""
    one_picture = FakeShapeModel(caps={"i2d": True, "t2d": False})
    with pytest.raises(ValueError, match="no multi-view tower"):
        MeshPipelineController(one_picture).resolve(MeshSpec(image=[PNG, OTHER]))


def test_the_file_is_the_container_the_request_asked_for(tmp_path):
    controller = MeshPipelineController(FakeShapeModel())
    for fmt, head in (("glb", b"glTF"), ("obj", b"v ")):
        resolved = controller.resolve(MeshSpec(output_format=fmt, **picture()))
        out = tmp_path / f"shape.{fmt}"
        controller.render(resolved, Cancellation(), lambda *a: None, out)
        assert out.read_bytes().startswith(head)


def test_a_cancelled_request_never_reaches_the_volume():
    model = FakeShapeModel(cancel_during="step")
    with pytest.raises(JobCancelled):
        render(model)
    assert "decode" not in model.events


def test_the_conditioning_of_a_repeated_picture_is_encoded_once():
    model = FakeShapeModel()
    controller = MeshPipelineController(model)
    for _ in range(2):
        controller.render(controller.resolve(MeshSpec(**picture(), seed=1)),
                          Cancellation(), lambda *a: None, Path("/tmp/mesh-a.glb"))
    assert model.events.count("encode") == 1


def test_a_different_picture_is_not_the_previous_ones_conditioning():
    model = FakeShapeModel()
    controller = MeshPipelineController(model)
    for data in (PNG, OTHER):
        controller.render(controller.resolve(MeshSpec(**picture(data))),
                          Cancellation(), lambda *a: None, Path("/tmp/mesh-b.glb"))
    assert model.events.count("encode") == 2


def test_a_shape_needs_a_picture_because_there_is_nothing_else_to_condition_on():
    controller = MeshPipelineController(FakeShapeModel())
    with pytest.raises(ValueError, match="does not support t2d"):
        controller.resolve(MeshSpec())


def test_the_requests_own_checks_run_before_anything_is_loaded():
    controller = MeshPipelineController(FakeShapeModel())
    with pytest.raises(ValueError, match="sampler"):
        controller.resolve(MeshSpec(sampler="nope", **picture()))
    with pytest.raises(ValueError, match="text_encoder"):
        controller.resolve(MeshSpec(lora_specs={"text_encoder": ["x.safetensors"]},
                                    **picture()))


class _TwoPieceModel(FakeShapeModel):
    """A codec whose cut has a piece of its own next to the shape."""

    def decode_mesh(self, latents, params):
        from thenoise.mesh.extract import extract_surface
        from thenoise.mesh.octree import sample_sdf

        self.events.append("decode")
        shape = extract_surface(sample_sdf(lambda points: points.norm(dim=-1) - 0.4, 32,
                                           device="cpu", bounds=1.0, level=0.0, start=8))
        corners = torch.tensor([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
                               dtype=torch.float32) * 0.05
        corners += torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
        piece = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
        self.cut = Mesh(torch.cat((shape.vertices, corners)),
                        torch.cat((shape.faces, piece + len(shape.vertices))))
        return self.cut


def _obj_counts(path: Path) -> tuple[int, int]:
    lines = path.read_text().splitlines()
    return (sum(1 for line in lines if line.startswith("v ")),
            sum(1 for line in lines if line.startswith("f ")))


def test_the_file_carries_the_finished_mesh_not_the_cut(tmp_path):
    """The junk goes before the budget is taken, and the file is what is left."""
    model = _TwoPieceModel()
    controller = MeshPipelineController(model)
    resolved = controller.resolve(MeshSpec(floaters=0.15, triangles=0.5,
                                           output_format="obj", **picture()))
    out = tmp_path / "shape.obj"
    seen: list[str] = []
    controller.render(resolved, Cancellation(), lambda stage, *rest: seen.append(stage), out)

    shape_faces = len(model.cut.faces) - 4
    written_vertices, written_faces = _obj_counts(out)
    assert written_faces == max(4, round(shape_faces * 0.5))
    assert written_vertices < len(model.cut.vertices)
    assert [one for one in seen if one == "finish"] == ["finish", "finish"]


def test_a_named_shape_lora_reaches_the_transformer_that_can_wear_it():
    """`lora_specs` are keyed by role, so a lookup by another name loads nothing."""
    model = FakeShapeModel()
    render(model, lora_specs=["x.safetensors"])
    assert model.lora_calls == [("dit", model.dit, ("x.safetensors",))]


def test_a_lora_is_switched_while_the_transformer_is_held():
    """Switching bakes weights in and may requantize, so the stage must be holding them."""
    model = FakeShapeModel()
    render(model, lora_specs=["x.safetensors"])
    events = model.events
    assert events.index("hold:dit") < events.index("loras:dit") < events.index("step")


def test_a_shape_with_no_lora_clears_the_previous_shapes():
    """A model instance is reused between jobs, so empty roles are offered none."""
    model = FakeShapeModel()
    render(model)
    assert model.lora_calls == [("dit", model.dit, ())]


def test_loras_without_a_directory_to_read_them_from_are_refused():
    model = FakeShapeModel()
    model.lora_dir = None
    controller = MeshPipelineController(model)
    resolved = controller.resolve(MeshSpec(lora_specs={"dit": ["x.safetensors"]},
                                           **picture()))
    with pytest.raises(ValueError, match="--lora-dir"):
        controller.render(resolved, Cancellation(), lambda *a: None,
                          Path("/tmp/mesh-lora.glb"))
