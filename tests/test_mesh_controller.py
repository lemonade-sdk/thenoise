import io
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from PIL import Image

from thenoise.jobs import Cancellation, JobCancelled
from thenoise.mesh.controller import MeshPipelineController
from thenoise.mesh.extract import Mesh
from thenoise.mesh.spec import MeshSpec
from thenoise.models.base import Conditioning

PNG = io.BytesIO()
Image.new("RGB", (24, 16), (3, 4, 5)).save(PNG, format="PNG")
PNG = PNG.getvalue()

OTHER = io.BytesIO()
Image.new("RGB", (24, 16), (9, 9, 9)).save(OTHER, format="PNG")
OTHER = OTHER.getvalue()

DEFAULTS = {
    "negative_prompt": "",
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
    """The one thing a controller asks of residency: which weights a stage held."""

    def __init__(self):
        self.held: list[tuple[str, ...]] = []

    @contextmanager
    def stage(self, *names, pixels):
        assert pixels > 0
        self.held.append(names)
        yield type("S", (), {"run": staticmethod(lambda work, *a, **k: work(*a, **k))})


class FakeShapeModel:
    name = "fake-shape"
    MEDIUM = "mesh"
    lora_dir = "/loras"
    default_prefs = DEFAULTS
    dtype = torch.float32
    device = "cpu"

    def __init__(self, cancel_during=None):
        self.memory = FakeMemory()
        self.events: list[str] = []
        self.seen_args = []
        self.decode_params = []
        self.cancel_during = cancel_during
        self.token = None

    def capability(self, medium):
        return {"i2d": True, "t2d": False}

    def lora_components(self):
        return ("dit",)

    def list_loras(self):
        return []

    def switch_loras(self, specs, component, stage):
        self.events.append(f"loras:{stage}")

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


def render(model, **overrides):
    controller = MeshPipelineController(model)
    resolved = controller.resolve(MeshSpec(**{**picture(), **overrides}))
    out = Path(f"/tmp/thenoise-mesh-{len(model.events)}.{resolved.output_format}")
    out.unlink(missing_ok=True)
    model.token = Cancellation()
    controller.render(resolved, model.token, lambda *a: None, out)
    return out



def test_the_three_components_are_brought_in_the_order_the_pipeline_needs_them():
    model = FakeShapeModel()
    render(model)
    assert model.memory.held == [("conditioner",), ("dit",), ("codec",)]
    assert model.events == ["encode", "loras:sampling", "init", "step", "decode"]


def test_a_resolved_request_reaches_the_volume_knobs():
    model = FakeShapeModel()
    render(model, octree_resolution=48, bounds=1.25, mc_level=0.01)
    assert model.decode_params[0].octree_resolution == 48
    assert model.decode_params[0].bounds == 1.25
    assert model.decode_params[0].mc_level == 0.01


def test_the_picture_reaches_the_conditioner_whole():
    model = FakeShapeModel()
    render(model)
    args = model.seen_args[0]
    assert args.image is not None and args.image.width == 24


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


def test_a_shape_has_no_text_to_condition_on_so_a_prompt_alone_is_refused():
    controller = MeshPipelineController(FakeShapeModel())
    with pytest.raises(ValueError, match="does not support t2d"):
        controller.resolve(MeshSpec(prompt="a chair"))


def test_the_requests_own_checks_run_before_anything_is_loaded():
    controller = MeshPipelineController(FakeShapeModel())
    with pytest.raises(ValueError, match="sampler"):
        controller.resolve(MeshSpec(sampler="nope", **picture()))
    with pytest.raises(ValueError, match="text_encoder"):
        controller.resolve(MeshSpec(lora_specs={"text_encoder": ["x.safetensors"]},
                                    **picture()))


def test_loras_without_a_directory_to_read_them_from_are_refused():
    model = FakeShapeModel()
    model.lora_dir = None
    controller = MeshPipelineController(model)
    resolved = controller.resolve(MeshSpec(lora_specs={"dit": ["x.safetensors"]},
                                           **picture()))
    with pytest.raises(ValueError, match="--lora-dir"):
        controller.render(resolved, Cancellation(), lambda *a: None,
                          Path("/tmp/mesh-lora.glb"))
