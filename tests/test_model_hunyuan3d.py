import pytest
import torch
from PIL import Image
from test_dit_hunyuan3d import TINY
from thenoise.dit.hunyuan3d import Hunyuan3DShapeParams

from thenoise.models.base import Conditioning
from thenoise.models.config import EncodePromptArgs, MeshDecodeParams, MeshSamplingParams
from thenoise.models.hunyuan3d import Hunyuan3DShapeModel


class RecordingDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, latents, t, context):
        self.seen.append((t.clone(), context.shape))
        return torch.full_like(latents, 2.0 + context.mean())


class StubConditioner:
    def __init__(self):
        self.calls = []

    def __call__(self, images, dtype):
        self.calls.append(len(images))
        return torch.zeros(len(images), 5, 8, dtype=dtype)

    def unconditional(self, like):
        return torch.full_like(like, 7.0)


class StubCodec(torch.nn.Module):
    def __init__(self, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.token_calls = 0
        self.sdf_calls = []

    def tokens(self, latents):
        self.token_calls += 1
        self.last = latents
        return torch.zeros(latents.shape[0], latents.shape[1], 4)

    def sdf(self, tokens, points):
        self.sdf_calls.append(tuple(points.shape))
        radius = points[0].norm(dim=-1)
        return (radius - 0.5)[None]


def bare(dtype=torch.float32) -> Hunyuan3DShapeModel:
    """An adapter with stubbed components: the kernels, without a checkpoint."""
    model = object.__new__(Hunyuan3DShapeModel)
    model.device = "cpu"
    model.dtype = dtype
    model.checkpoint_prefs = {}
    model.shape_params = Hunyuan3DShapeParams(**TINY)
    model.dit = RecordingDiT()
    model.conditioner = StubConditioner()
    model.codec = StubCodec()
    return model


def params(steps=4, num_latents=6, seed=3) -> MeshSamplingParams:
    return MeshSamplingParams(height=0, width=0, steps=steps, seed=seed,
                              guidance_scale=5.0, sampler="euler",
                              num_latents=num_latents)


def test_the_shared_shape_file_is_the_only_component():
    assert Hunyuan3DShapeModel.REQUIRED_COMPONENTS == ()
    assert Hunyuan3DShapeModel.MEDIUM == "mesh"
    assert Hunyuan3DShapeModel.REF_METHODS == ("vision",)


def test_the_dits_t_is_the_complement_of_the_solver_sigma():
    model = bare()
    latents = torch.zeros(1, 3, TINY["in_channels"])
    with torch.no_grad():
        out = model._velocity(latents, torch.tensor(0.25), torch.zeros(1, 5, TINY["context_dim"]))
    t, _ = model.dit.seen[0]
    assert torch.allclose(t, torch.tensor(0.75))
    assert torch.equal(out, torch.full_like(latents, -2.0))


def test_the_schedule_descends_and_sums_to_one():
    steps = bare().schedule(params(steps=5))
    assert len(steps) == 5
    assert float(steps[0].t) == 1.0
    assert all(float(step.delta) > 0 for step in steps)
    assert abs(sum(float(step.delta) for step in steps) - 1.0) < 1e-6


def test_a_requested_sigma_grid_replaces_the_uniform_one():
    model = bare()
    model.checkpoint_prefs = {"sigmas": [1.0, 0.5, 0.0]}
    steps = model.schedule(params(steps=9))
    assert len(steps) == 2
    assert [float(s.t) for s in steps] == [1.0, 0.5]


def test_latents_are_unstructured_tokens_and_seed_fixed():
    first = bare().init_latents(params(steps=1, num_latents=7, seed=11))
    again = bare().init_latents(params(steps=1, num_latents=7, seed=11))
    assert first.shape == (1, 7, TINY["in_channels"])
    assert torch.equal(first, again)


def test_cfg_mixes_the_two_branches():
    model = bare()
    cond = Conditioning(cond=torch.zeros(1, 5, TINY["context_dim"]),
                        null=torch.ones(1, 5, TINY["context_dim"]))
    latents = torch.zeros(1, 3, TINY["in_channels"])
    with torch.no_grad():
        guided = model.denoise_step(latents, torch.tensor(0.5), cond, 2.0, 0)
    assert torch.equal(guided, torch.full_like(latents, -1.0))
    assert len(model.dit.seen) == 2


def test_no_second_branch_without_guidance():
    model = bare()
    cond = Conditioning(cond=torch.zeros(1, 5, TINY["context_dim"]))
    with torch.no_grad():
        model.denoise_step(torch.zeros(1, 3, TINY["in_channels"]), torch.tensor(0.5),
                           cond, 1.0, 0)
    assert len(model.dit.seen) == 1


def test_the_picture_is_the_conditioning_and_the_null_branch_is_zeros_like_it():
    model = bare()
    out = model.encode_prompt(EncodePromptArgs(prompt="a chair", guidance_scale=5.0,
                                                  image=[Image.new("RGB", (32, 32))]))
    assert out.cond.shape == (1, 5, 8)
    assert torch.equal(out.null, torch.full(out.cond.shape, 7.0))
    assert model.conditioner.calls == [1]


def test_a_prompt_without_a_picture_is_refused():
    with pytest.raises(ValueError, match="image is required"):
        bare().encode_prompt(EncodePromptArgs(prompt="a chair", guidance_scale=5.0))


def test_the_scale_factor_comes_off_before_the_codec_is_queried():
    model = bare()
    latents = torch.full((1, 4, TINY["in_channels"]), 4.0)
    model.decode_mesh(latents, MeshDecodeParams(octree_resolution=8, bounds=1.0,
                                                chunk=64))
    assert model.codec.token_calls == 1
    assert torch.allclose(model.codec.last, torch.full((1, 4, TINY["in_channels"]), 2.0))


def test_the_tokens_are_refined_once_for_the_whole_volume():
    model = bare()
    model.decode_mesh(torch.zeros(1, 4, TINY["in_channels"]),
                      MeshDecodeParams(octree_resolution=16, bounds=1.0, chunk=32))
    assert model.codec.token_calls == 1
    assert len(model.codec.sdf_calls) > 1


def test_a_mesh_comes_out_of_the_decode():
    model = bare()
    mesh = model.decode_mesh(torch.zeros(1, 4, TINY["in_channels"]),
                            MeshDecodeParams(octree_resolution=32, bounds=1.0, chunk=512))
    assert not mesh.is_empty
    assert mesh.signed_volume() > 0


def test_references_are_fitted_to_whole_patches_inside_the_towers_canvas():
    sizing = Hunyuan3DShapeModel.REFERENCE_SIZING
    width, height = sizing.target_size(1920, 1080)
    assert width % 14 == 0 and height % 14 == 0
    assert width * height <= 518 * 518 + 14 * 14
    assert width / height == pytest.approx(1920 / 1080, rel=0.05)


class Handle:
    def __init__(self, *keys):
        self._keys = list(keys)

    def keys(self):
        return self._keys


def test_detection_uses_the_dits_own_fingerprint():
    assert Hunyuan3DShapeModel.detect(Handle(
        "model.t_embedder.mlp.2.weight", "model.blocks.0.attn1.k_norm.weight"))
    assert not Hunyuan3DShapeModel.detect(Handle(
        "model.t_embedder.mlp.0.weight", "model.blocks.0.attn1.k_norm.weight"))
    assert Hunyuan3DShapeModel.detect(Handle(
        "model.diffusion_model.model.t_embedder.mlp.2.weight",
        "model.diffusion_model.model.blocks.0.attn1.k_norm.weight"))
    assert not Hunyuan3DShapeModel.detect(Handle(
        "model.diffusion_model.txt_in.weight", "model.diffusion_model.img_in.weight"))
